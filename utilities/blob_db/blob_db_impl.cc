//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under the BSD-style license found in the
//  LICENSE file in the root directory of this source tree. An additional grant
//  of patent rights can be found in the PATENTS file in the same directory.
#ifndef ROCKSDB_LITE

#include "utilities/blob_db/blob_db_impl.h"
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <ctime>
#include <iomanip>
#include <limits>
#include <memory>

#include "db/db_impl.h"
#include "db/filename.h"
#include "db/write_batch_internal.h"
#include "rocksdb/convenience.h"
#include "rocksdb/env.h"
#include "rocksdb/iterator.h"
#include "rocksdb/utilities/stackable_db.h"
#include "rocksdb/utilities/transaction.h"
#include "table/block.h"
#include "table/block_based_table_builder.h"
#include "table/block_builder.h"
#include "table/meta_blocks.h"
#include "util/crc32c.h"
#include "util/file_reader_writer.h"
#include "util/instrumented_mutex.h"
#include "util/timer_queue.h"
#include "utilities/transactions/optimistic_transaction_db_impl.h"
#include "utilities/transactions/optimistic_transaction_impl.h"

namespace {
int kBlockBasedTableVersionFormat = 2;

void extendTTL(rocksdb::blobstorage::ttlrange_t* ttl_range, uint32_t ttl) {
  ttl_range->first = std::min(ttl_range->first, ttl);
  ttl_range->second = std::max(ttl_range->second, ttl);
}

void extendTimestamps(rocksdb::blobstorage::tsrange_t* ts_range, uint64_t ts) {
  ts_range->first = std::min(ts_range->first, ts);
  ts_range->second = std::max(ts_range->second, ts);
}

void extendSN(rocksdb::blobstorage::snrange_t* sn_range,
              rocksdb::SequenceNumber sn) {
  sn_range->first = std::min(sn_range->first, sn);
  sn_range->second = std::max(sn_range->second, sn);
}
}  // end namespace

namespace rocksdb {

namespace blobstorage {

// BlobHandle is a pointer to the blob that is stored in the LSM
class BlobHandle {
 public:
  BlobHandle()
      : file_number_(std::numeric_limits<uint64_t>::max()),
        offset_(std::numeric_limits<uint64_t>::max()),
        size_(std::numeric_limits<uint64_t>::max()),
        compression_(kNoCompression) {}

  uint64_t filenumber() const { return file_number_; }
  void set_filenumber(uint64_t fn) { file_number_ = fn; }

  // The offset of the block in the file.
  uint64_t offset() const { return offset_; }
  void set_offset(uint64_t _offset) { offset_ = _offset; }

  // The size of the stored block
  uint64_t size() const { return size_; }
  void set_size(uint64_t _size) { size_ = _size; }

  CompressionType compression() const { return compression_; }
  void set_compression(CompressionType t) { compression_ = t; }

  void EncodeTo(std::string* dst) const;
  Status DecodeFrom(Slice* input);

  void clear();

 private:
  uint64_t file_number_;
  uint64_t offset_;
  uint64_t size_;
  CompressionType compression_;
};

void BlobHandle::EncodeTo(std::string* dst) const {
  // Sanity check that all fields have been set
  assert(offset_ != std::numeric_limits<uint64_t>::max());
  assert(size_ != std::numeric_limits<uint64_t>::max());
  assert(file_number_ != std::numeric_limits<uint64_t>::max());

  dst->reserve(30);
  PutVarint64(dst, file_number_);
  PutVarint64(dst, offset_);
  PutVarint64(dst, size_);
  dst->push_back(static_cast<unsigned char>(compression_));
}

void BlobHandle::clear() {
  file_number_ = std::numeric_limits<uint64_t>::max();
  offset_ = std::numeric_limits<uint64_t>::max();
  size_ = std::numeric_limits<uint64_t>::max();
  compression_ = kNoCompression;
}

Status BlobHandle::DecodeFrom(Slice* input) {
  if (GetVarint64(input, &file_number_) && GetVarint64(input, &offset_) &&
      GetVarint64(input, &size_)) {
    compression_ = static_cast<CompressionType>(input->data()[0]);
    return Status::OK();
  } else {
    clear();
    return Status::Corruption("bad blob handle");
  }
}

Random blob_rgen(static_cast<uint32_t>(time(nullptr)));

void BlobDBFlushBeginListener::OnFlushBegin(DB* db, const FlushJobInfo& info) {
  if (impl_) impl_->OnFlushBeginHandler(db, info);
}

WalFilter::WalProcessingOption BlobReconcileWalFilter::LogRecordFound(
    unsigned long long log_number, const std::string& log_file_name,
    const WriteBatch& batch, WriteBatch* new_batch, bool* batch_changed) {
  return WalFilter::WalProcessingOption::kContinueProcessing;
}

bool blobf_compare_ttl::operator()(const std::shared_ptr<BlobFile>& lhs,
                                   const std::shared_ptr<BlobFile>& rhs) const {
  if (lhs->ttl_range_.first < rhs->ttl_range_.first) return true;

  if (lhs->ttl_range_.first > rhs->ttl_range_.first) return false;

  return lhs->BlobFileNumber() > rhs->BlobFileNumber();
}

void EvictAllVersionsFilter::Callback(int level, const Slice& key,
                                      ValueType value_type,
                                      const Slice& existing_value,
                                      const SequenceNumber& sn,
                                      bool is_new) const {
  if (!is_new && value_type == CompactionFilter::kValue) {
    BlobHandle handle;
    Slice lsmval(existing_value);
    Status s = handle.DecodeFrom(&lsmval);
    if (s.ok()) {
      if (impl_->debug_level_ >= 3)
        Log(InfoLogLevel::INFO_LEVEL, impl_->db_options_.info_log,
            "CALLBACK COMPACTED OUT KEY: %s SN: %d "
            "NEW: %d FN: %d OFFSET: %d SIZE: %d",
            key.ToString().c_str(), sn, is_new, handle.filenumber(),
            handle.offset(), handle.size());

      impl_->override_vals_q_.enqueue({handle.filenumber(), key.size(),
                                       handle.offset(), handle.size(), sn});
    }
  } else {
    if (impl_->debug_level_ >= 3)
      Log(InfoLogLevel::INFO_LEVEL, impl_->db_options_.info_log,
          "CALLBACK NEW KEY: %s SN: %d NEW: %d", key.ToString().c_str(), sn,
          is_new);
  }
}

std::unique_ptr<CompactionFilter>
EvictAllVersionsFilterFactory::CreateCompactionFilter(
    const CompactionFilter::Context& context) {
  if (context.is_manual_compaction) {
    return std::unique_ptr<CompactionFilter>(new EvictAllVersionsFilter(impl_));
  } else {
    return std::unique_ptr<CompactionFilter>(nullptr);
  }
}

Status BlobDB::DestroyBlobDB(const std::string& dbname, const Options& options,
                             const BlobDBOptions& bdb_options) {
  const ImmutableDBOptions soptions(SanitizeOptions(dbname, options));
  Env* env = soptions.env;

  Status result;
  std::string blobdir;
  blobdir = (bdb_options.path_relative) ? dbname + "/" + bdb_options.blob_dir
                                        : bdb_options.blob_dir;

  std::vector<std::string> filenames;
  Status status = env->GetChildren(blobdir, &filenames);

  for (const auto& f : filenames) {
    uint64_t number;
    FileType type;
    if (ParseFileName(f, &number, &type) && type == kBlobFile) {
      Status del = env->DeleteFile(blobdir + "/" + f);
      if (result.ok() && !del.ok()) {
        result = del;
      }
    }
  }

  env->DeleteDir(blobdir);
  return result;
}

BlobDBImpl::BlobDBImpl(const std::string& dbname,
                       const BlobDBOptions& blob_db_options,
                       const DBOptions& db_options)
    : BlobDB(nullptr),
      db_impl_(nullptr),
      myenv_(db_options.env),
      wo_set_(false),
      bdb_options_(blob_db_options),
      db_options_(db_options),
      env_options_(db_options),
      dir_change_(false),
      next_file_number_(1),
      epoch_of_(0),
      shutdown_(false),
      current_epoch_(0),
      open_file_count_(0),
      last_period_write_(0),
      last_period_ampl_(0),
      total_periods_write_(0),
      total_periods_ampl_(0),
      total_blob_space_(0),
      open_p1_done_(false),
      debug_level_(0) {
  blob_dir_ = (bdb_options_.path_relative)
                  ? dbname + "/" + bdb_options_.blob_dir
                  : bdb_options_.blob_dir;

  if (bdb_options_.default_ttl_extractor) {
    bdb_options_.extract_ttl_fn = &BlobDBImpl::extractTTLFromBlob;
  }
}

Status BlobDBImpl::LinkToBaseDB(DB* db) {
  assert(db_ == nullptr);
  assert(open_p1_done_);

  db_ = db;
  db_impl_ = dynamic_cast<DBImpl*>(db);

  myenv_ = db_->GetEnv();

  opt_db_ = std::make_shared<OptimisticTransactionDBImpl>(db, false);

  Status s = myenv_->CreateDirIfMissing(blob_dir_);
  if (!s.ok()) {
    Log(InfoLogLevel::WARN_LEVEL, db_options_.info_log,
        "Failed to create blob directory: %s status: '%s'", blob_dir_.c_str(),
        s.ToString().c_str());
  }
  s = myenv_->NewDirectory(blob_dir_, &dir_ent_);
  if (!s.ok()) {
    Log(InfoLogLevel::WARN_LEVEL, db_options_.info_log,
        "Failed to open blob directory: %s status: '%s'", blob_dir_.c_str(),
        s.ToString().c_str());
  }

  startBackgroundTasks();
  return s;
}

BlobDBImpl::BlobDBImpl(DB* db, const BlobDBOptions& blob_db_options)
    : BlobDB(db),
      db_impl_(dynamic_cast<DBImpl*>(db)),
      opt_db_(new OptimisticTransactionDBImpl(db, false)),
      wo_set_(false),
      bdb_options_(blob_db_options),
      db_options_(db->GetOptions()),
      env_options_(db_->GetOptions()),
      dir_change_(false),
      next_file_number_(1),
      epoch_of_(0),
      shutdown_(false),
      current_epoch_(0),
      open_file_count_(0),
      last_period_write_(0),
      last_period_ampl_(0),
      total_periods_write_(0),
      total_periods_ampl_(0),
      total_blob_space_(0) {
  assert(db_impl_ != nullptr);
  if (!bdb_options_.blob_dir.empty())
    blob_dir_ = (bdb_options_.path_relative)
                    ? db_->GetName() + "/" + bdb_options_.blob_dir
                    : bdb_options_.blob_dir;

  if (bdb_options_.default_ttl_extractor) {
    bdb_options_.extract_ttl_fn = &BlobDBImpl::extractTTLFromBlob;
  }
}

BlobDBImpl::~BlobDBImpl() {
  CancelAllBackgroundWork(db_, true);

  shutdown();
}

Status BlobDBImpl::openPhase1() {
  assert(db_ == nullptr);
  if (blob_dir_.empty())
    return Status::NotSupported("No blob directory in options");

  std::unique_ptr<Directory> dir_ent;
  Status s = myenv_->NewDirectory(blob_dir_, &dir_ent);
  if (!s.ok()) {
    Log(InfoLogLevel::WARN_LEVEL, db_options_.info_log,
        "Failed to open blob directory: %s status: '%s'", blob_dir_.c_str(),
        s.ToString().c_str());
    open_p1_done_ = true;
    return Status::OK();
  }

  s = openAllFiles();
  open_p1_done_ = true;
  return s;
}

void BlobDBImpl::startBackgroundTasks() {
  // store a call to a member function and object
  tqueue_.add(
      bdb_options_.reclaim_of_period,
      std::bind(&BlobDBImpl::reclaimOpenFiles, this, std::placeholders::_1));
  tqueue_.add(bdb_options_.gc_check_period,
              std::bind(&BlobDBImpl::runGC, this, std::placeholders::_1));
  tqueue_.add(
      bdb_options_.deletion_check_period,
      std::bind(&BlobDBImpl::evictDeletions, this, std::placeholders::_1));
  tqueue_.add(
      bdb_options_.deletion_check_period,
      std::bind(&BlobDBImpl::evictCompacted, this, std::placeholders::_1));
  tqueue_.add(
      bdb_options_.delete_obsf_period,
      std::bind(&BlobDBImpl::deleteObsFiles, this, std::placeholders::_1));
  tqueue_.add(bdb_options_.sanity_check_period,
              std::bind(&BlobDBImpl::sanityCheck, this, std::placeholders::_1));
  tqueue_.add(bdb_options_.wa_stats_period,
              std::bind(&BlobDBImpl::waStats, this, std::placeholders::_1));
  tqueue_.add(bdb_options_.fsync_files_period,
              std::bind(&BlobDBImpl::fsyncFiles, this, std::placeholders::_1));
  tqueue_.add(
      bdb_options_.check_seqf_period,
      std::bind(&BlobDBImpl::checkSeqFiles, this, std::placeholders::_1));
}

void BlobDBImpl::shutdown() { shutdown_.store(true); }

void BlobDBImpl::OnFlushBeginHandler(DB* db, const FlushJobInfo& info) {
  if (shutdown_.load()) return;

  // a callback that happens too soon needs to be ignored
  if (!db_) return;

  fsyncFiles(false);
}

Status BlobDBImpl::getAllLogFiles(
    std::set<std::pair<uint64_t, std::string>>* file_nums) {
  std::vector<std::string> all_files;
  Status status = myenv_->GetChildren(blob_dir_, &all_files);
  if (!status.ok()) {
    return status;
  }

  for (const auto& f : all_files) {
    uint64_t number;
    FileType type;
    bool psucc = ParseFileName(f, &number, &type);
    if (psucc && type == kBlobFile) {
      file_nums->insert(std::make_pair(number, f));
    } else {
      Log(InfoLogLevel::WARN_LEVEL, db_options_.info_log,
          "Skipping file in blob directory %s parse: %d type: %d", f.c_str(),
          psucc, ((psucc) ? type : -1));
    }
  }

  return status;
}

Status BlobDBImpl::openAllFiles() {
  WriteLock wl(&mutex_);

  std::set<std::pair<uint64_t, std::string>> file_nums;
  Status status = getAllLogFiles(&file_nums);

  if (!status.ok()) {
    Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
        "Failed to collect files from blob dir: %s status: '%s'",
        blob_dir_.c_str(), status.ToString().c_str());
    return status;
  }

  Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
      "BlobDir files path: %s count: %d min: %d max: %d", blob_dir_.c_str(),
      file_nums.size(), (file_nums.empty()) ? -1 : (file_nums.begin())->first,
      (file_nums.empty()) ? -1 : (file_nums.end())->first);

  if (!file_nums.empty())
    next_file_number_.store((file_nums.rbegin())->first + 1);

  for (auto f_iter : file_nums) {
    std::string bfpath = BlobFileName(blob_dir_, f_iter.first);
    uint64_t size_bytes;
    Status s1 = myenv_->GetFileSize(bfpath, &size_bytes);
    if (!s1.ok()) {
      Log(InfoLogLevel::WARN_LEVEL, db_options_.info_log,
          "Unable to get size of %s. File skipped from open status: '%s'",
          bfpath.c_str(), s1.ToString().c_str());
      continue;
    }

    if (debug_level_ >= 1)
      Log(InfoLogLevel::WARN_LEVEL, db_options_.info_log,
          "Blob File open: %s size: %d", bfpath.c_str(), size_bytes);

    std::shared_ptr<BlobFile> bfptr =
        std::make_shared<BlobFile>(this, blob_dir_, f_iter.first);

    bfptr->setFileSize(size_bytes);

    WriteLock wl_f(&bfptr->mutex_);

    // read header
    std::shared_ptr<Reader> reader;
    reader = bfptr->openSequentialReader(myenv_, db_options_, env_options_);
    s1 = reader->ReadHeader(&bfptr->header_);
    if (!s1.ok()) {
      Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
          "Failure to read header for blob-file %s "
          "status: '%s' size: %d",
          bfpath.c_str(), s1.ToString().c_str(), size_bytes);
      continue;
    }
    bfptr->header_valid_ = true;

    std::shared_ptr<RandomAccessFileReader> ra_reader =
        openRandomAccess_locked(bfptr, myenv_, env_options_);

    BlobLogFooter bf;
    s1 = bfptr->readFooter_locked(&bf);
    bfptr->closeRandomAccess_locked();
    if (s1.ok()) {
      s1 = bfptr->setFromFooter_locked(bf);
      if (!s1.ok()) {
        Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
            "Header Footer mismatch for blob-file %s "
            "status: '%s' size: %d",
            bfpath.c_str(), s1.ToString().c_str(), size_bytes);
        continue;
      }
    } else {
      Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
          "File found incomplete (w/o footer) %s", bfpath.c_str());

      // sequentially iterate over the file and read all the records
      ttlrange_t ttl_range(std::numeric_limits<uint32_t>::max(),
                           std::numeric_limits<uint32_t>::min());
      tsrange_t ts_range(std::numeric_limits<uint32_t>::max(),
                         std::numeric_limits<uint32_t>::min());
      snrange_t sn_range(std::numeric_limits<SequenceNumber>::max(),
                         std::numeric_limits<SequenceNumber>::min());

      uint64_t blob_count = 0;
      BlobLogRecord record;
      Reader::READ_LEVEL shallow = Reader::kReadLevelHdrFooterKey;

      // TODO(arahut) - when we detect corruption, we should truncate
      while (reader->ReadRecord(&record, shallow).ok()) {
        ++blob_count;
        if (bfptr->HasTTL()) {
          extendTTL(&ttl_range, record.GetTTL());
        }
        if (bfptr->HasTimestamp()) {
          extendTimestamps(&ts_range, record.GetTimeVal());
        }
        extendSN(&sn_range, record.GetSN());
      }

      if (!blob_count) {
        Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
            "BlobCount = 0 in file %s", bfpath.c_str());
        continue;
      }

      bfptr->setBlobCount(blob_count);
      bfptr->setSNRange(sn_range);

      if (bfptr->HasTimestamp()) bfptr->setTimeRange(ts_range);

      Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
          "Blob File: %s blob_count: %" PRIu64
          " size_bytes: %d sn_range: (%d, %d) ts: %d ttl: %d",
          bfpath.c_str(), blob_count, size_bytes, sn_range.first,
          sn_range.second, bfptr->HasTimestamp(), bfptr->HasTTL());

      if (bfptr->HasTTL()) {
        ttl_range.second =
            std::max(ttl_range.second,
                     ttl_range.first + (uint32_t)bdb_options_.ttl_range);
        bfptr->setTTLRange(ttl_range);

        std::time_t epoch_now = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        if (ttl_range.second < epoch_now) {
          Status fstatus = createWriter_locked(bfptr);
          if (fstatus.ok()) fstatus = bfptr->writeFooterAndClose_locked();
          if (!fstatus.ok()) {
            Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
                "Failed to close Blob File: %s status: '%s'. Skipped",
                bfpath.c_str(), fstatus.ToString().c_str());
            continue;
          } else {
            Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
                "Blob File Closed: %s now: %d ttl_range: (%d, %d)",
                bfpath.c_str(), epoch_now, ttl_range.first, ttl_range.second);
          }
        } else {
          open_blob_files_.insert(bfptr);
        }
      }
    }

    blob_files_.insert(std::make_pair(f_iter.first, bfptr));
  }

  return status;
}

Status BlobFile::readFooter_locked(BlobLogFooter* bf) {
  if (file_size_ < (BlobLogHeader::kHeaderSize + BlobLogFooter::kFooterSize))
    return Status::IOError("File does not have footer", PathName());

  uint64_t footer_offset = file_size_ - BlobLogFooter::kFooterSize;
  // assume that ra_file_reader_ is valid before we enter this
  assert(ra_file_reader_);

  Slice result;
  char scratch[BlobLogFooter::kFooterSize + 10];
  Status s = ra_file_reader_->Read(footer_offset, BlobLogFooter::kFooterSize,
                                   &result, scratch);

  if (!s.ok()) return s;

  s = bf->DecodeFrom(&result);
  return s;
}

void BlobDBImpl::closeRandomAccess_locked(
    const std::shared_ptr<BlobFile>& bfile) {
  bfile->closeRandomAccess_locked();
  open_file_count_--;
}

std::shared_ptr<RandomAccessFileReader> BlobDBImpl::openRandomAccess_locked(
    const std::shared_ptr<BlobFile>& bfile, Env* env,
    const EnvOptions& env_options) {
  bool fresh_open = false;
  auto rar = bfile->openRandomAccess_locked(env, env_options, &fresh_open);
  if (fresh_open) open_file_count_++;
  return rar;
}

std::shared_ptr<BlobFile> BlobDBImpl::newBlobFile(const std::string& reason) {
  uint64_t file_num = next_file_number_++;
  auto bfile = std::make_shared<BlobFile>(this, blob_dir_, file_num);
  Log(InfoLogLevel::DEBUG_LEVEL, db_options_.info_log,
      "New blob file created: %s reason='%s'", bfile->PathName().c_str(),
      reason.c_str());
  LogFlush(db_options_.info_log);
  return bfile;
}

Status BlobDBImpl::createWriter_locked(const std::shared_ptr<BlobFile>& bfile) {
  std::string fpath(bfile->PathName());
  std::unique_ptr<WritableFile> wfile;

  Status s = myenv_->ReopenWritableFile(fpath, &wfile, env_options_);
  if (!s.ok()) {
    Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
        "Failed to open blob file for write: %s status: '%s'"
        " exists: '%s'",
        fpath.c_str(), s.ToString().c_str(),
        myenv_->FileExists(fpath).ToString().c_str());
    return s;
  }

  std::unique_ptr<WritableFileWriter> fwriter;
  fwriter.reset(new WritableFileWriter(std::move(wfile), env_options_));

  uint64_t boffset = bfile->GetFileSize();
  if (debug_level_ >= 2 && boffset) {
    Log(InfoLogLevel::DEBUG_LEVEL, db_options_.info_log,
        "Open blob file: %s with offset: %d", fpath.c_str(), boffset);
  }

  Writer::ELEM_TYPE et = Writer::ET_NONE;
  if (bfile->file_size_ == BlobLogHeader::kHeaderSize)
    et = Writer::ET_FILE_HDR;
  else if (bfile->file_size_ > BlobLogHeader::kHeaderSize)
    et = Writer::ET_FOOTER;
  else if (bfile->file_size_) {
    Log(InfoLogLevel::WARN_LEVEL, db_options_.info_log,
        "Open blob file: %s with wrong size: %d", fpath.c_str(), boffset);
    return Status::Corruption("Invalid blob file size");
  }

  bfile->log_writer_ = std::make_shared<Writer>(
      std::move(fwriter), bfile->file_number_, bdb_options_.bytes_per_sync,
      db_options_.use_fsync, boffset);
  bfile->log_writer_->last_elem_type_ = et;

  return s;
}

std::shared_ptr<BlobFile> BlobDBImpl::findBlobFile_locked(
    uint32_t expiration) const {
  if (open_blob_files_.empty()) return nullptr;

  std::shared_ptr<BlobFile> tmp = std::make_shared<BlobFile>();
  tmp->ttl_range_ = std::make_pair(expiration, 0);

  auto citr = open_blob_files_.equal_range(tmp);
  if (citr.first == open_blob_files_.end()) {
    assert(citr.second == open_blob_files_.end());

    std::shared_ptr<BlobFile> check = *(open_blob_files_.rbegin());
    return (check->ttl_range_.second < expiration) ? nullptr : check;
  }

  if (citr.first != citr.second) return *(citr.first);

  auto finditr = citr.second;
  if (finditr != open_blob_files_.begin()) --finditr;

  bool b2 = (*finditr)->ttl_range_.second < expiration;
  bool b1 = (*finditr)->ttl_range_.first > expiration;

  return (b1 || b2) ? nullptr : (*finditr);
}

std::shared_ptr<Writer> BlobDBImpl::checkOrCreateWriter_locked(
    const std::shared_ptr<BlobFile>& bfile) {
  std::shared_ptr<Writer> writer = bfile->GetWriter();
  if (writer) return writer;

  Status s = createWriter_locked(bfile);
  if (!s.ok()) return nullptr;

  writer = bfile->GetWriter();
  return writer;
}

void BlobDBImpl::updateWriteOptions(const WriteOptions& options) {
  if (!wo_set_.load(std::memory_order_relaxed)) {
    // DCLP
    WriteLock wl(&mutex_);
    if (!wo_set_.load(std::memory_order_acquire)) {
      wo_set_.store(true, std::memory_order_release);
      write_options_ = options;
    }
  }
}

std::shared_ptr<BlobFile> BlobDBImpl::selectBlobFile() {
  uint32_t val = blob_rgen.Next();
  {
    ReadLock rl(&mutex_);
    if (open_simple_files_.size() == bdb_options_.num_concurrent_simple_blobs)
      return open_simple_files_[val % bdb_options_.num_concurrent_simple_blobs];
  }

  std::shared_ptr<BlobFile> bfile = newBlobFile("selectBlobFile");
  assert(bfile);

  // file not visible, hence no lock
  std::shared_ptr<Writer> writer = checkOrCreateWriter_locked(bfile);
  if (!writer) {
    Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
        "Failed to get writer from blob file: %s", bfile->PathName().c_str());
    return nullptr;
  }

  bfile->file_size_ = BlobLogHeader::kHeaderSize;
  bfile->header_.compression_ = bdb_options_.compression;
  bfile->header_valid_ = true;

  // CHECK again
  WriteLock wl(&mutex_);
  if (open_simple_files_.size() == bdb_options_.num_concurrent_simple_blobs) {
    return open_simple_files_[val % bdb_options_.num_concurrent_simple_blobs];
  }

  Status s = writer->WriteHeader(bfile->header_);
  if (!s.ok()) {
    Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
        "Failed to write header to new blob file: %s"
        " status: '%s'",
        bfile->PathName().c_str(), s.ToString().c_str());
    return nullptr;
  }

  dir_change_.store(true);
  blob_files_.insert(std::make_pair(bfile->BlobFileNumber(), bfile));
  open_simple_files_.push_back(bfile);
  return bfile;
}

std::shared_ptr<BlobFile> BlobDBImpl::selectBlobFileTTL(uint32_t expiration) {
  uint64_t epoch_read = 0;
  std::shared_ptr<BlobFile> bfile;
  {
    ReadLock rl(&mutex_);
    bfile = findBlobFile_locked(expiration);
    epoch_read = epoch_of_.load();
  }

  if (bfile) {
    assert(!bfile->Immutable());
    return bfile;
  }

  uint32_t exp_low =
      (expiration / bdb_options_.ttl_range) * bdb_options_.ttl_range;
  uint32_t exp_high = exp_low + bdb_options_.ttl_range;
  ttlrange_t ttl_guess = std::make_pair(exp_low, exp_high);

  bfile = newBlobFile("selectBlobFileTTL");
  assert(bfile);

  Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
      "New blob file TTL range: %s %d %d", bfile->PathName().c_str(), exp_low,
      exp_high);
  LogFlush(db_options_.info_log);

  // we don't need to take lock as no other thread is seeing bfile yet
  std::shared_ptr<Writer> writer = checkOrCreateWriter_locked(bfile);
  if (!writer) {
    Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
        "Failed to get writer from blob file with TTL: %s",
        bfile->PathName().c_str());
    return nullptr;
  }

  bfile->header_.setTTLGuess(ttl_guess);
  bfile->header_.compression_ = bdb_options_.compression;
  bfile->header_valid_ = true;
  bfile->file_size_ = BlobLogHeader::kHeaderSize;

  // set the first value of the range, since that is
  // concrete at this time.  also necessary to add to open_blob_files_
  bfile->ttl_range_ = ttl_guess;

  WriteLock wl(&mutex_);
  // in case the epoch has shifted in the interim, then check
  // check condition again - should be rare.
  if (epoch_of_.load() != epoch_read) {
    auto bfile2 = findBlobFile_locked(expiration);
    if (bfile2) return bfile2;
  }

  Status s = writer->WriteHeader(bfile->header_);
  if (!s.ok()) {
    Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
        "Failed to write header to new blob file: %s"
        " status: '%s'",
        bfile->PathName().c_str(), s.ToString().c_str());
    return nullptr;
  }

  dir_change_.store(true);
  blob_files_.insert(std::make_pair(bfile->BlobFileNumber(), bfile));
  open_blob_files_.insert(bfile);
  epoch_of_++;

  return bfile;
}

bool BlobDBImpl::extractTTLFromBlob(const Slice& value, Slice* newval,
                                    int32_t* ttl_val) {
  *newval = value;
  *ttl_val = -1;
  if (value.size() <= BlobDB::kTTLSuffixLength) return false;

  int32_t ttl_tmp =
      DecodeFixed32(value.data() + value.size() - sizeof(int32_t));
  std::string ttl_exp(value.data() + value.size() - BlobDB::kTTLSuffixLength,
                      4);
  if (ttl_exp != "ttl:") return false;

  newval->remove_suffix(BlobDB::kTTLSuffixLength);
  *ttl_val = ttl_tmp;
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// A specific pattern is looked up at the end of the value part.
// ttl:TTLVAL . if this pattern is found, PutWithTTL is called, otherwise
// regular Put is called.
////////////////////////////////////////////////////////////////////////////////
Status BlobDBImpl::Put(const WriteOptions& options,
                       ColumnFamilyHandle* column_family, const Slice& key,
                       const Slice& value) {
  Slice newval;
  int32_t ttl_val;
  if (bdb_options_.extract_ttl_fn) {
    bdb_options_.extract_ttl_fn(value, &newval, &ttl_val);
    return PutWithTTL(options, column_family, key, newval, ttl_val);
  }

  return PutWithTTL(options, column_family, key, value, -1);
}

Status BlobDBImpl::Delete(const WriteOptions& options,
                          ColumnFamilyHandle* column_family, const Slice& key) {
  SequenceNumber lsn = db_impl_->GetLatestSequenceNumber();
  Status s = db_->Delete(options, column_family, key);

  // add deleted key to list of keys that have been deleted for book-keeping
  delete_keys_q_.enqueue({column_family, key.ToString(), lsn});
  return s;
}

Status BlobDBImpl::SingleDelete(const WriteOptions& wopts,
                                ColumnFamilyHandle* column_family,
                                const Slice& key) {
  SequenceNumber lsn = db_impl_->GetLatestSequenceNumber();
  Status s = db_->SingleDelete(wopts, column_family, key);

  delete_keys_q_.enqueue({column_family, key.ToString(), lsn});
  return s;
}

Status BlobDBImpl::Write(const WriteOptions& opts, WriteBatch* updates) {
  class Handler1 : public WriteBatch::Handler {
   public:
    explicit Handler1(BlobDBImpl* i) : impl(i), previous_put(false) {}

    BlobDBImpl* impl;
    WriteBatch updates_blob;
    Status batch_rewrite_status;
    std::shared_ptr<BlobFile> last_file;
    bool previous_put;

    virtual Status PutCF(uint32_t column_family_id, const Slice& key,
                         const Slice& value_unc) override {
      Slice newval;
      int32_t ttl_val = -1;
      if (impl->bdb_options_.extract_ttl_fn)
        impl->bdb_options_.extract_ttl_fn(value_unc, &newval, &ttl_val);
      else
        newval = value_unc;

      int32_t expiration = -1;
      if (ttl_val != -1) {
        std::time_t cur_t = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        expiration = ttl_val + static_cast<int32_t>(cur_t);
      }
      std::shared_ptr<BlobFile> bfile =
          (ttl_val != -1) ? impl->selectBlobFileTTL(expiration)
                          : impl->selectBlobFile();
      if (last_file && last_file != bfile) {
        batch_rewrite_status = Status::NotFound("too many blob files");
        return batch_rewrite_status;
      }

      if (!bfile) {
        batch_rewrite_status = Status::NotFound("blob file not found");
        return batch_rewrite_status;
      }

      Slice value = value_unc;
      std::string compression_output;
      if (impl->bdb_options_.compression != kNoCompression) {
        CompressionType ct = impl->bdb_options_.compression;
        CompressionOptions compression_opts;
        value = CompressBlock(value_unc, compression_opts, &ct,
                              kBlockBasedTableVersionFormat, Slice(),
                              &compression_output);
      }

      std::string headerbuf;
      Writer::ConstructBlobHeader(&headerbuf, key, value, expiration, -1);

      if (previous_put) {
        impl->appendSN(last_file, -1);
        previous_put = false;
      }

      last_file = bfile;

      std::string index_entry;
      Status st = impl->appendBlob(bfile, headerbuf, key, value, &index_entry);

      if (expiration != -1)
        extendTTL(&(bfile->ttl_range_), (uint32_t)expiration);

      if (!st.ok()) {
        batch_rewrite_status = st;
      } else {
        previous_put = true;
        WriteBatchInternal::Put(&updates_blob, column_family_id, key,
                                index_entry);
      }
      return Status::OK();
    }

    virtual Status MergeCF(uint32_t column_family_id, const Slice& key,
                           const Slice& value) override {
      batch_rewrite_status =
          Status::NotSupported("Not supported operation in blob db.");
      return batch_rewrite_status;
    }

    virtual Status DeleteCF(uint32_t column_family_id,
                            const Slice& key) override {
      WriteBatchInternal::Delete(&updates_blob, column_family_id, key);
      return Status::OK();
    }

    virtual void LogData(const Slice& blob) override {
      updates_blob.PutLogData(blob);
    }

   private:
  };

  Handler1 handler1(this);
  updates->Iterate(&handler1);

  Status s;
  SequenceNumber lsn = db_impl_->GetLatestSequenceNumber();

  if (!handler1.batch_rewrite_status.ok()) {
    return handler1.batch_rewrite_status;
  } else {
    s = db_->Write(opts, &(handler1.updates_blob));
  }

  if (!s.ok()) return s;

  if (handler1.previous_put) {
    // this is the sequence number of the write.
    SequenceNumber sn = WriteBatchInternal::Sequence(&handler1.updates_blob);
    appendSN(handler1.last_file, sn);

    closeIf(handler1.last_file);
  }

  // add deleted key to list of keys that have been deleted for book-keeping
  class Handler2 : public WriteBatch::Handler {
   public:
    explicit Handler2(BlobDBImpl* i, const SequenceNumber& sn)
        : impl(i), lsn(sn) {}

    virtual Status DeleteCF(uint32_t column_family_id,
                            const Slice& key) override {
      ColumnFamilyHandle* cfh =
          impl->db_impl_->GetColumnFamilyHandleUnlocked(column_family_id);

      impl->delete_keys_q_.enqueue({cfh, key.ToString(), lsn});
      return Status::OK();
    }

   private:
    BlobDBImpl* impl;
    SequenceNumber lsn;
  };

  // add deleted key to list of keys that have been deleted for book-keeping
  Handler2 handler2(this, lsn);
  updates->Iterate(&handler2);

  return Status::OK();
}

Status BlobDBImpl::PutWithTTL(const WriteOptions& options,
                              ColumnFamilyHandle* column_family,
                              const Slice& key, const Slice& value,
                              int32_t ttl) {
  return PutUntil(
      options, column_family, key, value,
      (ttl != -1)
          ? ttl + static_cast<int32_t>(std::chrono::system_clock::to_time_t(
                      std::chrono::system_clock::now()))
          : -1);
}

Status BlobDBImpl::PutUntil(const WriteOptions& options,
                            ColumnFamilyHandle* column_family, const Slice& key,
                            const Slice& value_unc, int32_t expiration) {
  updateWriteOptions(options);

  std::shared_ptr<BlobFile> bfile =
      (expiration != -1) ? selectBlobFileTTL(expiration) : selectBlobFile();

  if (!bfile) return Status::NotFound("Blob file not found");

  Slice value = value_unc;
  std::string compression_output;
  if (bdb_options_.compression != kNoCompression) {
    CompressionType ct = bdb_options_.compression;
    CompressionOptions compression_opts;
    value = CompressBlock(value_unc, compression_opts, &ct,
                          kBlockBasedTableVersionFormat, Slice(),
                          &compression_output);
  }

  std::string headerbuf;
  Writer::ConstructBlobHeader(&headerbuf, key, value, expiration, -1);

  // this is another more safer way to do it, where you keep the writeLock
  // for the entire write path. this will increase latency and reduce
  // throughput
  // WriteLock lockbfile_w(&bfile->mutex_);
  // std::shared_ptr<Writer> writer =
  // checkOrCreateWriter_locked(bfile);

  if (debug_level_ >= 3)
    Log(InfoLogLevel::DEBUG_LEVEL, db_options_.info_log,
        ">Adding KEY FILE: %s: KEY: %s VALSZ: %d", bfile->PathName().c_str(),
        key.ToString().c_str(), value.size());

  std::string index_entry;
  Status s = appendBlob(bfile, headerbuf, key, value, &index_entry);
  if (!s.ok()) {
    Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
        "Failed to append blob to FILE: %s: KEY: %s VALSZ: %d"
        " status: '%s' blob_file: '%s'",
        bfile->PathName().c_str(), key.ToString().c_str(), value.size(),
        s.ToString().c_str(), bfile->DumpState().c_str());
    // Fallback just write to the LSM and get going
    WriteBatch batch;
    batch.Put(column_family, key, value);
    return db_->Write(options, &batch);
  }

  WriteBatch batch;
  batch.Put(column_family, key, index_entry);

  // this goes to the base db and can be expensive
  s = db_->Write(options, &batch);

  // this is the sequence number of the write.
  SequenceNumber sn = WriteBatchInternal::Sequence(&batch);

  if (debug_level_ >= 3)
    Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
        "<Adding KEY FILE: %s: KEY: %s SN: %d", bfile->PathName().c_str(),
        key.ToString().c_str(), sn);

  s = appendSN(bfile, sn);
  if (!s.ok()) {
    Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
        "Failed to append SN to FILE: %s: KEY: %s VALSZ: %d"
        " status: '%s' blob_file: '%s'",
        bfile->PathName().c_str(), key.ToString().c_str(), value.size(),
        s.ToString().c_str(), bfile->DumpState().c_str());
  }

  if (expiration != -1) extendTTL(&(bfile->ttl_range_), (uint32_t)expiration);

  closeIf(bfile);

  return s;
}

Status BlobDBImpl::appendBlob(const std::shared_ptr<BlobFile>& bfile,
                              const std::string& headerbuf, const Slice& key,
                              const Slice& value, std::string* index_entry) {
  Status s;

  uint64_t blob_offset = 0;
  uint64_t key_offset = 0;
  {
    WriteLock lockbfile_w(&bfile->mutex_);
    std::shared_ptr<Writer> writer = checkOrCreateWriter_locked(bfile);
    if (!writer) return Status::IOError("Failed to create blob writer");

    // write the blob to the blob log.
    s = writer->EmitPhysicalRecord(headerbuf, key, value, &key_offset,
                                   &blob_offset);
  }

  if (!s.ok()) {
    Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
        "Invalid status in appendBlob: %s status: '%s'",
        bfile->PathName().c_str(), s.ToString().c_str());
    return s;
  }

  // increment blob count
  bfile->blob_count_++;
  auto size_put = BlobLogRecord::kHeaderSize + key.size() + value.size();

  bfile->file_size_ += size_put;
  last_period_write_ += size_put;
  total_blob_space_ += size_put;

  // PutVarint64(index_entry, bfile->BlobFileNumber());
  BlobHandle handle;
  handle.set_filenumber(bfile->BlobFileNumber());
  handle.set_size(value.size());
  handle.set_offset(blob_offset);
  handle.set_compression(bdb_options_.compression);
  handle.EncodeTo(index_entry);

  if (debug_level_ >= 3)
    Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
        ">Adding KEY FILE: %s: BC: %d OFFSET: %d SZ: %d",
        bfile->PathName().c_str(), bfile->blob_count_.load(), blob_offset,
        value.size());

  return s;
}

Status BlobDBImpl::appendSN(const std::shared_ptr<BlobFile>& bfile,
                            const SequenceNumber& sn) {
  Status s;
  {
    WriteLock lockbfile(&bfile->mutex_);
    std::shared_ptr<Writer> writer = checkOrCreateWriter_locked(bfile);
    if (!writer) return Status::IOError("Failed to create blob writer");

    s = writer->AddRecordFooter(sn);
    if (!s.ok()) {
      Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
          "Invalid status in appendSN: %s status: '%s'",
          bfile->PathName().c_str(), s.ToString().c_str());
      return s;
    }

    if (sn != std::numeric_limits<SequenceNumber>::max())
      extendSN(&(bfile->sn_range_), sn);
  }

  bfile->file_size_ += BlobLogRecord::kFooterSize;
  last_period_write_ += BlobLogRecord::kFooterSize;
  total_blob_space_ += BlobLogRecord::kFooterSize;
  return s;
}

std::vector<Status> BlobDBImpl::MultiGet(
    const ReadOptions& options,
    const std::vector<ColumnFamilyHandle*>& column_family,
    const std::vector<Slice>& keys, std::vector<std::string>* values) {
  auto statuses = db_->MultiGet(options, column_family, keys, values);

  for (size_t i = 0; i < keys.size(); ++i) {
    if (!statuses[i].ok()) continue;

    BlockHandle handle;
    Slice index_entry_slice((*values)[i]);
    uint64_t file_number;
    if (!GetVarint64(&index_entry_slice, &file_number)) {
      statuses[i] = Status::Corruption();
      continue;
    }

    statuses[i] = handle.DecodeFrom(&index_entry_slice);
    if (!statuses[i].ok()) continue;

    std::shared_ptr<BlobFile> bfile;
    {
      ReadLock l(&mutex_);
      auto hitr = blob_files_.find(file_number);

      // file was deleted
      if (hitr == blob_files_.end()) {
        statuses[i] = Status::NotFound("Blob Not Found");
        continue;
      }

      bfile = hitr->second;
    }

    std::shared_ptr<RandomAccessFileReader> reader;
    {
      WriteLock lockbfile_w(&bfile->mutex_);
      reader = openRandomAccess_locked(bfile, myenv_, env_options_);
    }

    // allocate the buffer
    (*values)[i].resize(handle.size());
    char* buffer = &((*values)[i])[0];

    Slice blob_value;
    statuses[i] =
        reader->Read(handle.offset(), handle.size(), &blob_value, buffer);
  }
  return statuses;
}

Status BlobDBImpl::Get(const ReadOptions& options,
                       ColumnFamilyHandle* column_family, const Slice& key,
                       std::string* value) {
  auto cfh = reinterpret_cast<ColumnFamilyHandleImpl*>(column_family);
  auto cfd = cfh->cfd();

  Status s;
  std::string index_entry;
  s = db_->Get(options, column_family, key, &index_entry);
  if (!s.ok()) {
    if (debug_level_ >= 3)
      Log(InfoLogLevel::WARN_LEVEL, db_options_.info_log,
          "Get Failed on LSM KEY: %s status: '%s'", key.ToString().c_str(),
          s.ToString().c_str());
    return s;
  }

  BlobHandle handle;
  Slice index_entry_slice(index_entry);
  s = handle.DecodeFrom(&index_entry_slice);
  if (!s.ok()) return s;

  // offset has to have certain min, as we will read CRC
  // later from the Blob Header, which needs to be also a
  // valid offset.
  if (handle.offset() <
      (BlobLogHeader::kHeaderSize + BlobLogRecord::kHeaderSize + key.size())) {
    if (debug_level_ >= 2) {
      Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
          "Invalid blob handle file_number: %" PRIu64 " blob_offset: %" PRIu64
          " blob_size: %" PRIu64 " key: %s",
          handle.filenumber(), handle.offset(), handle.size(), key.data());
    }
    return Status::NotFound("Blob Not Found, although found in LSM");
  }

  std::shared_ptr<BlobFile> bfile;
  {
    ReadLock l(&mutex_);
    auto hitr = blob_files_.find(handle.filenumber());

    // file was deleted
    if (hitr == blob_files_.end()) {
      return Status::NotFound("Blob Not Found as blob file missing");
    }

    bfile = hitr->second;
  }

  std::shared_ptr<RandomAccessFileReader> reader;
  {
    WriteLock lockbfile_w(&bfile->mutex_);
    reader = openRandomAccess_locked(bfile, myenv_, env_options_);
  }

  std::string *valueptr = value;
  std::string value_c;
  if (bdb_options_.compression != kNoCompression) {
    valueptr = &value_c;
  }

  // allocate the buffer. This is safe in C++11
  valueptr->resize(handle.size());
  char* buffer = &(*valueptr)[0];

  Slice blob_value;
  s = reader->Read(handle.offset(), handle.size(), &blob_value, buffer);
  if (!s.ok()) {
    if (debug_level_ >= 2) {
      Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
          "Failed to read blob from file: %s blob_offset: %" PRIu64
          " blob_size: %" PRIu64 " key: %s status: '%s'",
          bfile->PathName().c_str(), handle.offset(), handle.size(), key.data(),
          s.ToString().c_str());
    }
    return Status::NotFound("Blob Not Found as couldnt retrieve CRC");
  }

  Slice crc_slice;
  uint32_t crc_exp;
  std::string crc_str;
  crc_str.resize(sizeof(uint32_t));
  char* crc_buffer = &(crc_str[0]);
  s = reader->Read(handle.offset() - (key.size() + sizeof(uint32_t)),
                   sizeof(uint32_t), &crc_slice, crc_buffer);
  if (!s.ok() || !GetFixed32(&crc_slice, &crc_exp)) {
    if (debug_level_ >= 2) {
      Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
          "Failed to fetch blob crc file: %s blob_offset: %" PRIu64
          " blob_size: %" PRIu64 " key: %s status: '%s'",
          bfile->PathName().c_str(), handle.offset(), handle.size(), key.data(),
          s.ToString().c_str());
    }
    return Status::NotFound("Blob Not Found as couldnt retrieve CRC");
  }

  uint32_t crc = crc32c::Extend(0, blob_value.data(), blob_value.size());
  crc = crc32c::Mask(crc);  // Adjust for storage
  if (crc != crc_exp) {
    if (debug_level_ >= 2) {
      Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
          "Blob crc mismatch file: %s blob_offset: %" PRIu64
          " blob_size: %" PRIu64 " key: %s status: '%s'",
          bfile->PathName().c_str(), handle.offset(), handle.size(), key.data(),
          s.ToString().c_str());
    }
    return Status::NotFound("Corruption. Blob CRC mismatch");
  }

  if (bdb_options_.compression != kNoCompression) {
    BlockContents contents;
    s = UncompressBlockContentsForCompressionType(
        blob_value.data(), blob_value.size(),
        &contents, kBlockBasedTableVersionFormat,
        Slice(), bdb_options_.compression, *(cfd->ioptions()));
    *value = std::move(contents.data.ToString());
  }

  return s;
}

std::pair<bool, int64_t> BlobDBImpl::sanityCheck(bool aborted) {
  if (aborted) return std::make_pair(false, -1);

  Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log, "Starting Sanity Check");

  Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
      "Number of files %" PRIu64, blob_files_.size());

  Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
      "Number of open files %" PRIu64, open_blob_files_.size());

  for (auto bfile : open_blob_files_) {
    assert(!bfile->Immutable());
  }

  std::time_t epoch_now =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

  for (auto bfile_pair : blob_files_) {
    auto bfile = bfile_pair.second;
    Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
        "Blob File %s %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %d",
        bfile->PathName().c_str(), bfile->GetFileSize(), bfile->BlobCount(),
        bfile->deleted_count_, bfile->deleted_size_,
        (bfile->ttl_range_.second - epoch_now));
  }

  // reschedule
  return std::make_pair(true, -1);
}

std::pair<bool, int64_t> BlobDBImpl::closeSeqWrite(
    std::shared_ptr<BlobFile> bfile, bool aborted) {
  {
    WriteLock wl(&mutex_);

    // this prevents others from picking up this file
    open_blob_files_.erase(bfile);

    auto findit =
        std::find(open_simple_files_.begin(), open_simple_files_.end(), bfile);
    if (findit != open_simple_files_.end()) open_simple_files_.erase(findit);
  }

  if (!bfile->closed_.load()) {
    WriteLock wl_f(&bfile->mutex_);
    bfile->writeFooterAndClose_locked();
  }

  return std::make_pair(false, -1);
}

void BlobDBImpl::closeIf(const std::shared_ptr<BlobFile>& bfile) {
  // atomic read
  bool close = bfile->GetFileSize() > bdb_options_.blob_file_size;
  if (!close) return;

  if (debug_level_ >= 2) {
    Log(InfoLogLevel::DEBUG_LEVEL, db_options_.info_log,
        "Scheduling file for close %s fsize: %" PRIu64 " limit: %" PRIu64,
        bfile->PathName().c_str(), bfile->GetFileSize(),
        bdb_options_.blob_file_size);
  }

  {
    WriteLock wl(&mutex_);

    open_blob_files_.erase(bfile);
    auto findit =
        std::find(open_simple_files_.begin(), open_simple_files_.end(), bfile);
    if (findit != open_simple_files_.end())
      open_simple_files_.erase(findit);
    else
      Log(InfoLogLevel::WARN_LEVEL, db_options_.info_log,
          "File not found while closing %s fsize: %" PRIu64
          " Multithreaded Writes?",
          bfile->PathName().c_str(), bfile->GetFileSize());
  }

  tqueue_.add(0, std::bind(&BlobDBImpl::closeSeqWrite, this, bfile,
                           std::placeholders::_1));
}

bool BlobDBImpl::FileDeleteOk_SnapshotCheck_locked(
    const std::shared_ptr<BlobFile>& bfile) {
  assert(bfile->Obsolete());

  SequenceNumber esn = bfile->GetSNRange().first;

  // this is not correct.
  // you want to check that there are no snapshots in the
  bool notok = db_impl_->HasActiveSnapshotLaterThanSN(esn);
  if (notok) {
    Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
        "Could not delete file due to snapshot failure %s",
        bfile->PathName().c_str());
    return false;
  } else {
    Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
        "Will delete file due to snapshot success %s",
        bfile->PathName().c_str());
    return true;
  }
}

bool BlobDBImpl::findFileAndEvictABlob(uint64_t file_number, uint64_t key_size,
                                       uint64_t blob_offset,
                                       uint64_t blob_size) {
  (void)blob_offset;
  std::shared_ptr<BlobFile> bfile;
  ReadLock l(&mutex_);
  auto hitr = blob_files_.find(file_number);

  // file was deleted
  if (hitr == blob_files_.end()) {
    Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
        "Could not find file_number %" PRIu64, file_number);
    return false;
  }

  bfile = hitr->second;
  bfile->deleted_count_++;
  bfile->deleted_size_ += key_size + blob_size + BlobLogRecord::kHeaderSize +
                          BlobLogRecord::kFooterSize;
  return true;
}

bool BlobDBImpl::markBlobDeleted(const Slice& key, const Slice& lsmValue) {
  Slice val(lsmValue);
  BlobHandle handle;
  Status s = handle.DecodeFrom(&val);
  if (!s.ok()) {
    Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
        "Could not parse lsm val in markBlobDeleted %s",
        lsmValue.ToString().c_str());
    return false;
  }
  bool succ = findFileAndEvictABlob(handle.filenumber(), key.size(),
                                    handle.offset(), handle.size());
  return succ;
}

std::pair<bool, int64_t> BlobDBImpl::evictCompacted(bool aborted) {
  if (aborted) return std::make_pair(false, -1);

  override_packet_t packet;
  while (override_vals_q_.dequeue(&packet)) {
    bool succ = findFileAndEvictABlob(packet.file_number_, packet.key_size_,
                                      packet.blob_offset_, packet.blob_size_);

    if (!succ)
      Log(InfoLogLevel::DEBUG_LEVEL, db_options_.info_log,
          "EVICT COMPACTION FAILURE SN: %d FN: %d OFFSET: %d SIZE: %d",
          packet.dsn_, packet.file_number_, packet.blob_offset_,
          packet.blob_size_);

    if (debug_level_ >= 3)
      Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
          "EVICT COMPACTED SN: %d FN: %d OFFSET: %d SIZE: %d SUCC: %d",
          packet.dsn_, packet.file_number_, packet.blob_offset_,
          packet.blob_size_, succ);
  }
  return std::make_pair(true, -1);
}

std::pair<bool, int64_t> BlobDBImpl::evictDeletions(bool aborted) {
  if (aborted) return std::make_pair(false, -1);

  ColumnFamilyHandle* last_cfh = nullptr;
  Options last_op;

  Arena arena;
  ScopedArenaIterator iter;

  // we will use same RangeDelAggregator for all cf's.
  // essentially we do not support Range Deletes now
  std::unique_ptr<RangeDelAggregator> range_del_agg;
  delete_packet_t dpacket;
  while (delete_keys_q_.dequeue(&dpacket)) {
    if (last_cfh != dpacket.cfh_) {
      if (!range_del_agg) {
        auto cfhi = reinterpret_cast<ColumnFamilyHandleImpl*>(dpacket.cfh_);
        auto cfd = cfhi->cfd();
        range_del_agg.reset(new RangeDelAggregator(cfd->internal_comparator(),
                                                   kMaxSequenceNumber));
      }

      // this can be expensive
      last_cfh = dpacket.cfh_;
      last_op = db_impl_->GetOptions(last_cfh);
      iter.set(db_impl_->NewInternalIterator(&arena, range_del_agg.get(),
                                             dpacket.cfh_));
      // this will not work for multiple CF's.
    }

    Slice user_key(dpacket.key_);
    InternalKey target(user_key, dpacket.dsn_, kTypeValue);

    Slice eslice = target.Encode();
    iter->Seek(eslice);

    if (!iter->status().ok()) {
      Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
          "Invalid iterator seek %s", dpacket.key_.c_str());
      continue;
    }

    const Comparator* bwc = BytewiseComparator();
    while (iter->Valid()) {
      if (!bwc->Equal(ExtractUserKey(iter->key()), ExtractUserKey(eslice)))
        break;

      ParsedInternalKey ikey(Slice(), 0, kTypeValue);
      if (!ParseInternalKey(iter->key(), &ikey)) {
        continue;
      }

      // once you hit a DELETE, assume the keys below have been
      // processed previously
      if (ikey.type == kTypeDeletion || ikey.type == kTypeSingleDeletion) break;

      Slice val = iter->value();
      markBlobDeleted(ikey.user_key, val);

      iter->Next();
    }
  }
  return std::make_pair(true, -1);
}

std::pair<bool, int64_t> BlobDBImpl::checkSeqFiles(bool aborted) {
  if (aborted) return std::make_pair(false, -1);

  std::vector<std::shared_ptr<BlobFile>> process_files;
  {
    ReadLock l(&mutex_);
    for (auto bfile : open_blob_files_) {
      {
        ReadLock rl_f(&bfile->mutex_);

        std::time_t epoch_now = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        if (bfile->ttl_range_.second > epoch_now) continue;
        process_files.push_back(bfile);
      }
    }
  }

  for (auto bfile : process_files) closeSeqWrite(bfile, false);

  return std::make_pair(true, -1);
}

std::pair<bool, int64_t> BlobDBImpl::fsyncFiles(bool aborted) {
  if (aborted) return std::make_pair(false, -1);

  std::vector<std::shared_ptr<BlobFile>> process_files;
  {
    ReadLock l(&mutex_);
    for (auto fitr : open_blob_files_) {
      if (fitr->NeedsFsync(true, bdb_options_.bytes_per_sync))
        process_files.push_back(fitr);
    }

    for (auto fitr : open_simple_files_) {
      if (fitr->NeedsFsync(true, bdb_options_.bytes_per_sync))
        process_files.push_back(fitr);
    }
  }

  for (auto fitr : process_files) {
    if (fitr->NeedsFsync(true, bdb_options_.bytes_per_sync)) fitr->Fsync();
  }

  bool expected = true;
  if (dir_change_.compare_exchange_weak(expected, false)) dir_ent_->Fsync();

  return std::make_pair(true, -1);
}

std::pair<bool, int64_t> BlobDBImpl::reclaimOpenFiles(bool aborted) {
  if (aborted) return std::make_pair(false, -1);

  if (open_file_count_.load() < bdb_options_.open_files_trigger)
    return std::make_pair(true, -1);

  // in the future, we should sort by last_access_
  // instead of closing every file
  ReadLock l(&mutex_);
  for (auto const& ent : blob_files_) {
    auto bfile = ent.second;
    if (bfile->last_access_.load() == -1) continue;

    WriteLock lockbfile_w(&bfile->mutex_);
    closeRandomAccess_locked(bfile);
  }

  return std::make_pair(true, -1);
}

std::pair<bool, int64_t> BlobDBImpl::waStats(bool aborted) {
  if (aborted) return std::make_pair(false, -1);

  WriteLock wl(&mutex_);

  if (all_periods_write_.size() < bdb_options_.wa_num_stats_periods) {
    total_periods_write_ -= (*all_periods_write_.begin());
    total_periods_ampl_ = (*all_periods_ampl_.begin());

    all_periods_write_.pop_front();
    all_periods_ampl_.pop_front();
  }

  uint64_t val1 = last_period_write_.load();
  uint64_t val2 = last_period_ampl_.load();

  all_periods_write_.push_back(val1);
  all_periods_ampl_.push_back(val2);

  last_period_write_ = 0;
  last_period_ampl_ = 0;

  total_periods_write_ += val1;
  total_periods_ampl_ += val2;

  return std::make_pair(true, -1);
}

////////////////////////////////////////////////////////////////////////////////
// iterate over the blobs sequentially and check if the blob sequence number
// is the latest. If it is the latest, preserve it, otherwise delete it
// if it is TTL based, and the TTL has expired, then
// we can blow the entity if the key is still the latest or the Key is not
// found
// WHAT HAPPENS IF THE KEY HAS BEEN OVERRIDEN. Then we can drop the blob
// without doing anything if the earliest snapshot is not
// referring to that sequence number, i.e. it is later than the sequence number
// of the new key
//
// if it is not TTL based, then we can blow the key if the key has been
// DELETED in the LSM
////////////////////////////////////////////////////////////////////////////////
Status BlobDBImpl::writeBatchOfDeleteKeys(
    const std::shared_ptr<BlobFile>& bfptr, std::time_t tt) {
  // ensure that a sequential reader is available
  std::shared_ptr<Reader> reader;
  {
    WriteLock lockbfile_w(&(bfptr->mutex_));
    // sequentially iterate over the file and read all the records
    reader = bfptr->openSequentialReader(myenv_, db_options_, env_options_);
    if (!reader) {
      // report something here.
      Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
          "File sequential reader could not be opened",
          bfptr->PathName().c_str());
      return Status::IOError("failed to create sequential reader");
    }
  }

  ReadLock rlock(&bfptr->mutex_);

  BlobLogHeader header;
  Status s = reader->ReadHeader(&header);
  if (!s.ok()) {
    Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
        "Failure to read header for blob-file %s", bfptr->PathName().c_str());
    return s;
  }

  ColumnFamilyHandle* cfh = bfptr->GetColumnFamily(db_);
  auto cfhi = reinterpret_cast<ColumnFamilyHandleImpl*>(cfh);
  auto cfd = cfhi->cfd();
  bool has_ttl = header.HasTTL();

  uint64_t blob_count = 0;
  BlobLogRecord record;

  // this reads the key but skips the blob
  Reader::READ_LEVEL shallow = Reader::kReadLevelHdrFooterKey;

  SuperVersion* sv = db_impl_->GetAndRefSuperVersion(cfd);
  if (sv == nullptr) {
    Status result = Status::InvalidArgument("Could not access column family 0");
    return result;
  }

  assert(opt_db_);
  Transaction* txn = static_cast<OptimisticTransactionDB*>(opt_db_.get())
                         ->BeginTransaction(write_options_);

  OptimisticTransactionImpl* otxn =
      dynamic_cast<OptimisticTransactionImpl*>(txn);
  assert(otxn != nullptr);

  bool no_relocation_ttl = (has_ttl && tt > bfptr->GetTTLRange().second);
  bool no_relocation_lsmdel =
      (bfptr->GetFileSize() ==
       (BlobLogHeader::kHeaderSize + bfptr->deleted_size_ +
        BlobLogFooter::kFooterSize));

  bool no_relocation = no_relocation_ttl || no_relocation_lsmdel;

  std::shared_ptr<BlobFile> newfile;
  std::shared_ptr<Writer> new_writer;

  if (!no_relocation) {
    std::string reason("GC of ");
    reason += bfptr->PathName();
    newfile = newBlobFile(reason);

    new_writer = checkOrCreateWriter_locked(newfile);
    newfile->header_ = std::move(header);
    newfile->header_valid_ = true;
    newfile->file_size_ = BlobLogHeader::kHeaderSize;
    s = new_writer->WriteHeader(newfile->header_);

    if (!s.ok()) {
      Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
          "File: %s - header writing failed", newfile->PathName().c_str());
      no_relocation = true;
      // should be a much severe error
    }

    // read the blob because you have to write it back to
    // new file
    shallow = Reader::kReadLevelHdrFooterKeyBlob;
  }

  uint32_t num_deletes = 0;

  while (reader->ReadRecord(&record, shallow).ok()) {
    ++blob_count;

    // this particular TTL has expired
    if (no_relocation_ttl || (has_ttl && tt > record.GetTTL())) {
      txn->Delete(cfh, record.Key());
      num_deletes++;
      continue;
    }

    SequenceNumber seq = kMaxSequenceNumber;
    bool found_record_for_key = false;
    s = db_impl_->GetLatestSequenceForKey(sv, record.Key(), false, &seq,
                                          &found_record_for_key);

    bool del_this = s.ok() && (!found_record_for_key || seq != record.GetSN());

    if (del_this) {
      // stil could have a TOCTOU
      txn->Delete(cfh, record.Key());
      num_deletes++;
      continue;
    }

    {
      std::string index_entry;
      PutVarint64(&index_entry, newfile->BlobFileNumber());

      BlockHandle handle;
      auto raw_block_size = record.Blob().size();
      handle.set_size(raw_block_size);

      // these are returned values
      uint64_t blob_offset = 0;
      uint64_t key_offset = 0;
      // write the blob to the blob log.
      s = new_writer->AddRecord(record.Key(), record.Blob(), &key_offset,
                                &blob_offset, record.GetTTL());

      new_writer->AddRecordFooter(record.GetSN());
      newfile->blob_count_++;
      newfile->file_size_ += BlobLogRecord::kHeaderSize + record.Key().size() +
                             record.Blob().size() + BlobLogRecord::kFooterSize;

      handle.set_offset(blob_offset);
      handle.EncodeTo(&index_entry);

      txn->Put(cfh, record.Key(), index_entry);
    }
  }

  if (!no_relocation) total_blob_space_ += newfile->file_size_;

  if (num_deletes != 0) {
    Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
        "File: %s Number of deletes %d", bfptr->PathName().c_str(),
        num_deletes);
  }

  // Now write the
  db_impl_->ReturnAndCleanupSuperVersion(cfd, sv);

  s = txn->Commit();

  delete txn;

  // if this fails, we should try to preserve the write batch.
  if (s.IsBusy()) {
    Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
        "Optimistic transaction failed: %s", bfptr->PathName().c_str());
    return s;
  }

  if (s.ok()) {
    Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
        "Successfully added deletes back into LSM: %s",
        bfptr->PathName().c_str());
    // we are done.
  }

  return s;
}

bool BlobDBImpl::shouldGCFile_locked(std::shared_ptr<BlobFile> bfile,
                                     std::time_t tt, uint64_t last_id,
                                     std::string* reason) {
  if (bfile->HasTTL()) {
    ttlrange_t ttl_range = bfile->GetTTLRange();
    if (tt > ttl_range.second) {
      *reason = "entire file ttl expired";
      return true;
    }

    if (bdb_options_.ttl_range < bdb_options_.partial_expiration_gc_range) {
      *reason = "has ttl but partial expiration not turned on";
      return false;
    }

    if (!bfile->file_size_.load()) {
      Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
          "Invalid file size = 0 %s", bfile->PathName().c_str());
      *reason = "file is empty";
      return false;
    }

    bool ret = ((bfile->deleted_size_ * 100.0 / bfile->file_size_.load()) >
                bdb_options_.partial_expiration_pct);
    if (ret)
      *reason = "deleted blobs beyond threshold";
    else
      *reason = "deleted blobs below threshold";
    return ret;
  }

  if ((bfile->deleted_size_ * 100.0 / bfile->file_size_.load()) >
      bdb_options_.partial_expiration_pct) {
    *reason = "deleted simple blobs beyond threshold";
    return true;
  }

  // if we haven't reached limits of disk space, don't DELETE
  if (total_blob_space_.load() < bdb_options_.blob_dir_size) {
    *reason = "disk space not exceeded";
    return false;
  }

  bool ret = bfile->BlobFileNumber() == last_id;
  if (ret)
    *reason = "eligible last simple blob file";
  else
    *reason = "not eligible since not last simple blob file";
  return ret;
}

std::pair<bool, int64_t> BlobDBImpl::deleteObsFiles(bool aborted) {
  if (aborted) return std::make_pair(false, -1);

  {
    ReadLock rl(&mutex_);
    if (obsolete_files_.empty()) return std::make_pair(true, -1);
  }

  std::list<std::shared_ptr<BlobFile>> tobsolete;
  {
    WriteLock wl(&mutex_);
    tobsolete.swap(obsolete_files_);
  }

  bool file_deleted = false;
  for (auto iter = tobsolete.begin(); iter != tobsolete.end();) {
    auto bfile = *iter;
    {
      ReadLock rl(&bfile->mutex_);
      if (!FileDeleteOk_SnapshotCheck_locked(bfile)) {
        ++iter;
        continue;
      }
    }

    Status s = myenv_->DeleteFile(bfile->PathName());
    if (!s.ok()) {
      Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
          "File failed to be deleted as obsolete %s",
          bfile->PathName().c_str());
      ++iter;
      continue;
    }

    file_deleted = true;
    total_blob_space_ -= bfile->file_size_;
    Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
        "File deleted as obsolete from blob dir %s", bfile->PathName().c_str());

    iter = tobsolete.erase(iter);
  }

  // directory change. Fsync
  if (file_deleted) dir_ent_->Fsync();

  // put files back into obsolete if for some reason, delete failed
  if (!tobsolete.empty()) {
    WriteLock wl(&mutex_);
    for (auto bfile : tobsolete) obsolete_files_.push_front(bfile);
  }

  if (aborted)
    return std::make_pair(false, -1);
  else
    return std::make_pair(true, -1);
}

bool BlobDBImpl::callbackEvicts_Impl(std::shared_ptr<BlobFile> bfile) {
  std::shared_ptr<Reader> reader;
  {
    WriteLock lockbfile_w(&(bfile->mutex_));
    // sequentially iterate over the file and read all the records
    reader = bfile->openSequentialReader(myenv_, db_options_, env_options_);
    if (!reader) {
      // report something here.
      Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
          "File sequential reader could not be opened for evict callback: %s",
          bfile->PathName().c_str());
      return false;
    }
  }

  ReadLock rlock(&bfile->mutex_);

  BlobLogHeader header;
  Status s = reader->ReadHeader(&header);
  if (!s.ok()) {
    Log(InfoLogLevel::ERROR_LEVEL, db_options_.info_log,
        "Failure to read header for blob-file during evict callback %s",
        bfile->PathName().c_str());
    return false;
  }

  ColumnFamilyHandle* cfh = bfile->GetColumnFamily(db_);
  BlobLogRecord record;
  Reader::READ_LEVEL full = Reader::kReadLevelHdrFooterKeyBlob;
  while (reader->ReadRecord(&record, full).ok()) {
    bdb_options_.gc_evict_cb_fn(cfh, record.Key(), record.Blob());
  }

  return true;
}

std::pair<bool, int64_t> BlobDBImpl::removeTimerQ(TimerQueue* tq,
                                                  bool aborted) {
  WriteLock wl(&mutex_);
  for (auto itr = cb_threads_.begin(); itr != cb_threads_.end(); ++itr) {
    if ((*itr).get() != tq) continue;

    cb_threads_.erase(itr);
    break;
  }
  return std::make_pair(false, -1);
}

std::pair<bool, int64_t> BlobDBImpl::callbackEvicts(
    TimerQueue* tq, std::shared_ptr<BlobFile> bfile, bool aborted) {
  if (aborted) return std::make_pair(false, -1);
  bool succ = callbackEvicts_Impl(bfile);
  if (succ) {
    Log(InfoLogLevel::DEBUG_LEVEL, db_options_.info_log,
        "Eviction callbacks completed %s", bfile->PathName().c_str());
  }

  WriteLock wl(&mutex_);
  bfile->setCanBeDeleted();
  obsolete_files_.push_front(bfile);
  if (tq) {
    // all of the callbacks have been processed
    tqueue_.add(0, std::bind(&BlobDBImpl::removeTimerQ, this, tq,
                             std::placeholders::_1));
  }
  return std::make_pair(false, -1);
}

std::pair<bool, int64_t> BlobDBImpl::runGC(bool aborted) {
  if (aborted) return std::make_pair(false, -1);

  current_epoch_++;
  // collect the ID of the last regular file, in case
  // we need to GC it.
  uint64_t last_id = std::numeric_limits<uint64_t>::max();
  std::vector<std::shared_ptr<BlobFile>> blob_files;
  {
    // take a copy
    ReadLock l(&mutex_);
    blob_files.reserve(blob_files_.size());
    for (auto const& ent : blob_files_) {
      blob_files.push_back(ent.second);

      // has ttl is immutable, once set, hence no locks
      if (!ent.second->HasTTL()) last_id = ent.second->BlobFileNumber();
    }
  }

  if (!blob_files.size()) return std::make_pair(true, -1);

  // 100.0 / 15.0 = 7
  uint64_t next_epoch_increment = static_cast<uint64_t>(
      std::ceil(100 / static_cast<double>(bdb_options_.gc_file_pct)));

  // 15% of files
  size_t files_to_collect =
      (bdb_options_.gc_file_pct * blob_files.size()) / 100;

  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  std::time_t tt = std::chrono::system_clock::to_time_t(now);

  size_t files_processed = 0;

  std::vector<std::shared_ptr<BlobFile>> to_process;
  for (auto bfile : blob_files) {
    // if this is the first time processing the file
    // i.e. gc_epoch == -1, process it.
    // else process the file if its processing epoch matches
    // the current epoch. Typically the #of epochs should be
    // around 5-10
    if (!(bfile->gc_epoch_ == -1 ||
          (uint64_t)bfile->gc_epoch_ == current_epoch_)) {
      continue;
    }

    files_processed++;
    // reset the epoch
    bfile->gc_epoch_ = current_epoch_ + next_epoch_increment;
    to_process.push_back(bfile);
    if (files_processed >= files_to_collect) break;
  }

  // in this collect the set of files, which became obsolete
  std::vector<std::shared_ptr<BlobFile>> obsoletes;
  for (auto bfile : to_process) {
    // File can be obsolete. File can be Open for writes, or closed
    std::string pn = bfile->PathName();

    {
      ReadLock rl_f(&bfile->mutex_);
      // in a previous pass, this file was marked obsolete
      // or this file is still active for appends.
      if (bfile->Obsolete() || !bfile->Immutable()) continue;

      std::string reason;
      bool shouldgc = shouldGCFile_locked(bfile, tt, last_id, &reason);
      if (!shouldgc) {
        Log(InfoLogLevel::DEBUG_LEVEL, db_options_.info_log,
            "File has been skipped for GC ttl %s %d %d reason='%s'", pn.c_str(),
            tt, bfile->GetTTLRange().second, reason.c_str());
        continue;
      }

      Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
          "File has been chosen for GC ttl %s %d %d reason='%s'", pn.c_str(),
          tt, bfile->GetTTLRange().second, reason.c_str());
    }

    Status s = writeBatchOfDeleteKeys(bfile, tt);
    if (!s.ok()) continue;

    obsoletes.push_back(bfile);
  }

  if (!obsoletes.empty()) {
    bool evict_cb = (!!bdb_options_.gc_evict_cb_fn);
    std::shared_ptr<TimerQueue> tq;
    if (evict_cb) tq = std::make_shared<TimerQueue>();

    // if evict callback is present, first schedule
    // the callback thread
    WriteLock wl(&mutex_);
    for (auto bfile : obsoletes) {
      bool last_file = (bfile == obsoletes.back());
      // remove from global list so writers
      blob_files_.erase(bfile->BlobFileNumber());

      if (!evict_cb) {
        bfile->setCanBeDeleted();
        obsolete_files_.push_front(bfile);
      } else {
        tq->add(0, std::bind(&BlobDBImpl::callbackEvicts, this,
                             (last_file) ? tq.get() : nullptr, bfile,
                             std::placeholders::_1));
      }
    }
    if (evict_cb) cb_threads_.emplace_back(tq);
  }

  // reschedule
  return std::make_pair(true, -1);
}

BlobFile::BlobFile()
    : parent_(nullptr),
      file_number_(0),
      blob_count_(0),
      gc_epoch_(-1),
      file_size_(0),
      deleted_count_(0),
      deleted_size_(0),
      closed_(false),
      can_be_deleted_(false),
      ttl_range_(std::make_pair(0, 0)),
      time_range_(std::make_pair(0, 0)),
      sn_range_(std::make_pair(0, 0)),
      last_access_(-1),
      last_fsync_(0),
      header_valid_(false) {}

BlobFile::BlobFile(const BlobDBImpl* p, const std::string& bdir, uint64_t fn)
    : parent_(p),
      path_to_dir_(bdir),
      file_number_(fn),
      blob_count_(0),
      gc_epoch_(-1),
      file_size_(0),
      deleted_count_(0),
      deleted_size_(0),
      closed_(false),
      can_be_deleted_(false),
      ttl_range_(std::make_pair(0, 0)),
      time_range_(std::make_pair(0, 0)),
      sn_range_(std::make_pair(0, 0)),
      last_access_(-1),
      last_fsync_(0),
      header_valid_(false) {}

BlobFile::~BlobFile() {
  if (can_be_deleted_) {
    std::string pn(PathName());
    Status s = Env::Default()->DeleteFile(PathName());
    if (!s.ok()) {
      // Log(InfoLogLevel::INFO_LEVEL, db_options_.info_log,
      // "File could not be deleted %s", pn.c_str());
    }
  }
}

std::string BlobFile::PathName() const {
  return BlobFileName(path_to_dir_, file_number_);
}

std::shared_ptr<Reader> BlobFile::openSequentialReader(
    Env* env, const DBOptions& db_options,
    const EnvOptions& env_options) const {
  std::unique_ptr<SequentialFile> sfile;
  Status s = env->NewSequentialFile(PathName(), &sfile, env_options);
  if (!s.ok()) {
    // report something here.
    return nullptr;
  }

  std::unique_ptr<SequentialFileReader> sfile_reader;
  sfile_reader.reset(new SequentialFileReader(std::move(sfile)));

  std::shared_ptr<Reader> log_reader =
      std::make_shared<Reader>(db_options.info_log, std::move(sfile_reader));

  return log_reader;
}

std::string BlobFile::DumpState() const {
  char str[1000];
  std::snprintf(str, sizeof(str),
                "path: %s fn: %" PRIu64 " blob_count: %" PRIu64
                " gc_epoch: %" PRIu64 " file_size: %" PRIu64
                " deleted_count: %" PRIu64 " deleted_size: %" PRIu64
                " closed: %d can_be_deleted: %d ttl_range: (%d, %d)"
                " sn_range: (%" PRIu64 " %" PRIu64 "), writer: %d reader: %d",
                path_to_dir_.c_str(), file_number_, blob_count_.load(),
                gc_epoch_.load(), file_size_.load(), deleted_count_,
                deleted_size_, closed_.load(), can_be_deleted_.load(),
                ttl_range_.first, ttl_range_.second, sn_range_.first,
                sn_range_.second, (!!log_writer_), (!!ra_file_reader_));
  return str;
}

bool BlobFile::NeedsFsync(bool hard, uint64_t bytes_per_sync) const {
  assert(last_fsync_ <= file_size_);
  return (hard) ? file_size_ > last_fsync_
                : (file_size_ - last_fsync_) >= bytes_per_sync;
}

Status BlobFile::writeFooterAndClose_locked() {
  Log(InfoLogLevel::INFO_LEVEL, parent_->db_options_.info_log,
      "File is being closed after footer %s", PathName().c_str());

  BlobLogFooter footer;
  footer.blob_count_ = blob_count_;
  if (HasTTL()) footer.setTTLRange(ttl_range_);

  footer.sn_range_ = sn_range_;
  if (HasTimestamp()) footer.setTimeRange(time_range_);

  // this will close the file and reset the Writable File Pointer.
  Status s = log_writer_->AppendFooter(footer);
  if (s.ok()) {
    closed_ = true;
    file_size_ += BlobLogFooter::kFooterSize;
  } else {
    Log(InfoLogLevel::ERROR_LEVEL, parent_->db_options_.info_log,
        "Failure to read Header for blob-file %s", PathName().c_str());
  }
  // delete the sequential writer
  log_writer_.reset();
  return s;
}

Status BlobFile::setFromFooter_locked(const BlobLogFooter& footer) {
  if (footer.HasTTL() != header_.HasTTL()) {
    return Status::Corruption("has_ttl mismatch");
  }
  if (footer.HasTimestamp() != header_.HasTimestamp()) {
    return Status::Corruption("has_ts mismatch");
  }

  // assume that file has been fully fsync'd
  last_fsync_.store(file_size_);
  blob_count_ = footer.GetBlobCount();
  ttl_range_ = footer.GetTTLRange();
  time_range_ = footer.GetTimeRange();
  sn_range_ = footer.GetSNRange();
  closed_ = true;

  return Status::OK();
}

void BlobFile::Fsync() {
  if (log_writer_.get()) {
    log_writer_->Sync();
    last_fsync_.store(file_size_.load());
  }
}

void BlobFile::closeRandomAccess_locked() {
  ra_file_reader_.reset();
  last_access_ = -1;
}

std::shared_ptr<RandomAccessFileReader> BlobFile::openRandomAccess_locked(
    Env* env, const EnvOptions& env_options, bool* fresh_open) {
  *fresh_open = false;
  last_access_ =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  if (ra_file_reader_) return ra_file_reader_;

  std::unique_ptr<RandomAccessFile> rfile;
  Status s = env->NewRandomAccessFile(PathName(), &rfile, env_options);
  if (!s.ok()) {
    Log(InfoLogLevel::ERROR_LEVEL, parent_->db_options_.info_log,
        "Failed to open blob file for random-read: %s status: '%s'"
        " exists: '%s'",
        PathName().c_str(), s.ToString().c_str(),
        env->FileExists(PathName()).ToString().c_str());
    return nullptr;
  }

  ra_file_reader_ = std::make_shared<RandomAccessFileReader>(std::move(rfile));
  *fresh_open = true;
  return ra_file_reader_;
}

ColumnFamilyHandle* BlobFile::GetColumnFamily(DB* db) {
  return db->DefaultColumnFamily();
}

}  // namespace blobstorage
}  // namespace rocksdb
#endif  // ROCKSDB_LITE
