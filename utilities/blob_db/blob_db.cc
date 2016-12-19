//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under the BSD-style license found in the
//  LICENSE file in the root directory of this source tree. An additional grant
//  of patent rights can be found in the PATENTS file in the same directory.
//
#ifndef ROCKSDB_LITE

#include "utilities/blob_db/blob_db.h"
#include "db/filename.h"
#include "db/write_batch_internal.h"
#include "rocksdb/compaction_filter.h"
#include "rocksdb/convenience.h"
#include "rocksdb/env.h"
#include "rocksdb/iterator.h"
#include "rocksdb/utilities/stackable_db.h"
#include "table/block.h"
#include "table/block_based_table_builder.h"
#include "table/block_builder.h"
#include "util/cf_options.h"
#include "util/crc32c.h"
#include "util/file_reader_writer.h"
#include "util/instrumented_mutex.h"
#include "utilities/blob_db/blob_db_impl.h"

namespace rocksdb {

namespace blobstorage {
port::Mutex listener_mutex;
typedef std::shared_ptr<BlobDBFlushBeginListener> FlushBegin_t;
typedef std::shared_ptr<BlobReconcileWalFilter> ReconcileWal_t;
typedef std::shared_ptr<EvictAllVersionsFilterFactory> EvictAllVersions_t;

// to ensure the lifetime of the listeners
std::vector<FlushBegin_t> all_flush_begins;
std::vector<ReconcileWal_t> all_wal_filters;

Status BlobDB::OpenAndLoad(const Options& options,
                           const BlobDBOptions& bdb_options,
                           const std::string& dbname, BlobDB** blob_db,
                           Options* changed_options) {
  *changed_options = options;
  *blob_db = nullptr;

  FlushBegin_t fblistener = std::make_shared<BlobDBFlushBeginListener>();
  ReconcileWal_t rw_filter = std::make_shared<BlobReconcileWalFilter>();
  EvictAllVersions_t comp_filt_factory =
      std::make_shared<EvictAllVersionsFilterFactory>();

  {
    MutexLock l(&listener_mutex);
    all_flush_begins.push_back(fblistener);
    all_wal_filters.push_back(rw_filter);
  }

  changed_options->listeners.emplace_back(fblistener);
  changed_options->wal_filter = rw_filter.get();
  changed_options->compaction_filter_factory = comp_filt_factory;

  DBOptions db_options(*changed_options);

  // we need to open blob db first so that recovery can happen
  BlobDBImpl* bdb = new BlobDBImpl(dbname, bdb_options, db_options);

  fblistener->setImplPtr(bdb);
  rw_filter->setImplPtr(bdb);
  comp_filt_factory->setImplPtr(bdb);

  Status s = bdb->openPhase1();
  if (!s.ok()) return s;

  *blob_db = bdb;
  return s;
}

Status BlobDB::Open(const Options& options, const BlobDBOptions& bdb_options,
                    const std::string& dbname, BlobDB** blob_db) {
  *blob_db = nullptr;
  DBOptions db_options(options);
  ColumnFamilyOptions cf_options(options);
  std::vector<ColumnFamilyDescriptor> column_families;
  column_families.push_back(
      ColumnFamilyDescriptor(kDefaultColumnFamilyName, cf_options));
  std::vector<ColumnFamilyHandle*> handles;
  Status s = BlobDB::Open(db_options, bdb_options, dbname, column_families,
                          &handles, blob_db);
  if (s.ok()) {
    assert(handles.size() == 1);
    // i can delete the handle since DBImpl is always holding a reference to
    // default column family
    delete handles[0];
  }
  return s;
}

Status BlobDB::Open(const DBOptions& db_options,
                    const BlobDBOptions& bdb_options, const std::string& dbname,
                    const std::vector<ColumnFamilyDescriptor>& column_families,
                    std::vector<ColumnFamilyHandle*>* handles, BlobDB** blob_db,
                    bool no_base_db) {
  *blob_db = nullptr;

  DBOptions my_db_options(db_options);
  FlushBegin_t fblistener = std::make_shared<BlobDBFlushBeginListener>();
  ReconcileWal_t rw_filter = std::make_shared<BlobReconcileWalFilter>();

  my_db_options.listeners.emplace_back(fblistener);
  my_db_options.wal_filter = rw_filter.get();

  EvictAllVersions_t comp_filt_factory =
      std::make_shared<EvictAllVersionsFilterFactory>();
  std::vector<ColumnFamilyDescriptor> my_column_families = column_families;
  for (size_t i = 0; i < my_column_families.size(); ++i) {
    my_column_families[i].options.compaction_filter_factory = comp_filt_factory;
  }

  {
    MutexLock l(&listener_mutex);
    all_flush_begins.push_back(fblistener);
    all_wal_filters.push_back(rw_filter);
  }

  // we need to open blob db first so that recovery can happen
  BlobDBImpl* bdb = new BlobDBImpl(dbname, bdb_options, my_db_options);
  fblistener->setImplPtr(bdb);
  rw_filter->setImplPtr(bdb);
  comp_filt_factory->setImplPtr(bdb);

  Status s = bdb->openPhase1();
  if (!s.ok()) return s;

  if (no_base_db) return s;

  DB* db = nullptr;
  s = DB::Open(my_db_options, dbname, my_column_families, handles, &db);
  if (!s.ok()) return s;

  // set the implementation pointer
  s = bdb->LinkToBaseDB(db);
  if (!s.ok()) {
    delete bdb;
    bdb = nullptr;
  }
  *blob_db = bdb;
  return s;
}

BlobDB::BlobDB(DB* db) : StackableDB(db) {}

////////////////////////////////////////////////////////////////////////////////
//
//
// std::function<int(double)> fnCaller =
//     std::bind(&A::fn, &anInstance, std::placeholders::_1);
////////////////////////////////////////////////////////////////////////////////
BlobDBOptions::BlobDBOptions()
    : blob_dir("blob_dir"),
      path_relative(true),
      is_fifo(false),
      blob_dir_size(1000ULL * 1024ULL * 1024ULL * 1024ULL),
      ttl_range(3600),
      min_blob_size(512),
      bytes_per_sync(0),
      blob_file_size(256 * 1024 * 1024),
      num_concurrent_simple_blobs(4),
      deletion_check_period(2 * 1000),
      gc_file_pct(20),
      gc_check_period(60 * 1000),
      sanity_check_period(20 * 60 * 1000),
      open_files_trigger(100),
      wa_num_stats_periods(24),
      wa_stats_period(3600 * 1000),
      partial_expiration_gc_range(4 * 3600),
      partial_expiration_pct(75),
      fsync_files_period(10 * 1000),
      reclaim_of_period(1 * 1000),
      delete_obsf_period(10 * 1000),
      check_seqf_period(10 * 1000),
      default_ttl_extractor(false),
      compression(kNoCompression) {}

}  // namespace blobstorage
}  // namespace rocksdb
#endif
