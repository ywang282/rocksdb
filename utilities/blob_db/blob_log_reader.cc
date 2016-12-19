//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under the BSD-style license found in the
//  LICENSE file in the root directory of this source tree. An additional grant
//  of patent rights can be found in the PATENTS file in the same directory.
//
#ifndef ROCKSDB_LITE

#include "utilities/blob_db/blob_log_reader.h"

#include <cstdio>
#include "rocksdb/env.h"
#include "util/coding.h"
#include "util/crc32c.h"
#include "util/file_reader_writer.h"

namespace rocksdb {
namespace blobstorage {

Reader::Reader(std::shared_ptr<Logger> info_log,
               unique_ptr<SequentialFileReader>&& _file)
    : info_log_(info_log),
      file_(std::move(_file)),
      backing_store_(new char[kBlockSize]),
      buffer_(),
      next_byte_(0) {}

Reader::~Reader() {}

Status Reader::ReadHeader(BlobLogHeader* header) {
  assert(file_.get() != nullptr);
  assert(next_byte_ == 0);
  Status status =
      file_->Read(BlobLogHeader::kHeaderSize, &buffer_, backing_store_.get());
  next_byte_ += BlobLogHeader::kHeaderSize;
  if (!status.ok()) return status;

  status = header->DecodeFrom(&buffer_);
  return status;
}

Status Reader::ReadRecord(BlobLogRecord* record, READ_LEVEL level,
                          WALRecoveryMode wal_recovery_mode) {
  record->clear();
  buffer_.clear();
  backing_store_.get()[0] = '\0';

  Status status =
      file_->Read(BlobLogRecord::kHeaderSize, &buffer_, backing_store_.get());
  next_byte_ += BlobLogRecord::kHeaderSize;
  if (!status.ok()) return status;

  status = record->DecodeHeaderFrom(buffer_);
  if (!status.ok()) return status;

  uint32_t header_crc = 0;
  uint32_t blob_crc = 0;
  header_crc =
      crc32c::Extend(header_crc, buffer_.data(),
                     BlobLogRecord::kHeaderSize - 2 * sizeof(uint32_t));

  uint64_t kb_size = record->GetKeySize() + record->GetBlobSize();
  switch (level) {
    case kReadLevelHdrFooter:
      file_->Skip(kb_size);
      status = file_->Read(BlobLogRecord::kFooterSize, &buffer_,
                           backing_store_.get());
      record->sn_ = DecodeFixed64(buffer_.data());
      next_byte_ += (kb_size + BlobLogRecord::kFooterSize);
      return status;

    case kReadLevelHdrFooterKey:
      record->resizeKeyBuffer(record->GetKeySize());
      status = file_->Read(record->GetKeySize(), &record->key_,
                           record->getKeyBuffer());
      header_crc =
          crc32c::Extend(header_crc, record->key_.data(), record->GetKeySize());
      header_crc = crc32c::Mask(header_crc);
      if (header_crc != record->header_cksum_)
        return Status::Corruption("Record Checksum mismatch: header_cksum");

      file_->Skip(record->GetBlobSize());
      status = file_->Read(BlobLogRecord::kFooterSize, &buffer_,
                           backing_store_.get());
      record->sn_ = DecodeFixed64(buffer_.data());
      next_byte_ += (kb_size + BlobLogRecord::kFooterSize);
      return status;

    case kReadLevelHdrFooterKeyBlob:
      record->resizeKeyBuffer(record->GetKeySize());
      status = file_->Read(record->GetKeySize(), &record->key_,
                           record->getKeyBuffer());

      header_crc =
          crc32c::Extend(header_crc, record->key_.data(), record->GetKeySize());
      header_crc = crc32c::Mask(header_crc);
      if (header_crc != record->header_cksum_)
        return Status::Corruption("Record Checksum mismatch: header_cksum");

      record->resizeBlobBuffer(record->GetBlobSize());
      status = file_->Read(record->GetBlobSize(), &record->blob_,
                           record->getBlobBuffer());

      blob_crc =
          crc32c::Extend(blob_crc, record->blob_.data(), record->blob_.size());
      blob_crc = crc32c::Mask(blob_crc);
      if (blob_crc != record->checksum_)
        return Status::Corruption("Blob Checksum mismatch");

      status = file_->Read(BlobLogRecord::kFooterSize, &buffer_,
                           backing_store_.get());
      record->sn_ = DecodeFixed64(buffer_.data());
      next_byte_ += (kb_size + BlobLogRecord::kFooterSize);
      return status;
    default:
      assert(0);
      return status;
  }
}

}  // namespace blobstorage
}  // namespace rocksdb
#endif  // ROCKSDB_LITE
