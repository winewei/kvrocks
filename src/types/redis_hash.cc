/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */

#include "redis_hash.h"

#include <rocksdb/status.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <random>
#include <utility>

#include "config/config.h"
#include "db_util.h"
#include "parse_util.h"
#include "sample_helper.h"

namespace redis {

namespace {

// Bytewise ordering on the field name, the same order RocksDB keeps the sub keys in.
struct FieldLess {
  bool operator()(const FieldValue &lhs, const FieldValue &rhs) const { return lhs.field < rhs.field; }
  bool operator()(const FieldValue &lhs, std::string_view field) const { return std::string_view(lhs.field) < field; }
  bool operator()(std::string_view field, const FieldValue &rhs) const { return field < std::string_view(rhs.field); }
};

FieldValue *FindInlineField(std::vector<FieldValue> &fields, std::string_view field) {
  auto it = std::lower_bound(fields.begin(), fields.end(), field, FieldLess{});
  if (it == fields.end() || it->field != field) return nullptr;
  return &*it;
}

void UpsertInlineField(std::vector<FieldValue> &fields, std::string_view field, std::string_view value) {
  auto it = std::lower_bound(fields.begin(), fields.end(), field, FieldLess{});
  if (it != fields.end() && it->field == field) {
    it->value.assign(value.data(), value.size());
    return;
  }
  fields.emplace(it, std::string(field), std::string(value));
}

// Deduplicates the fields of one command (the last occurrence wins) and sorts them.
std::vector<FieldValue> SortedUniqueFields(const std::vector<FieldValue> &field_values) {
  std::vector<FieldValue> result;
  result.reserve(field_values.size());
  for (auto it = field_values.rbegin(); it != field_values.rend(); ++it) {
    if (FindInlineField(result, it->field) != nullptr) continue;
    UpsertInlineField(result, it->field, it->value);
  }
  return result;
}

}  // namespace

void EncodeInlineHashFields(const std::vector<FieldValue> &field_values, std::string *dst) {
  for (const auto &fv : field_values) {
    PutVarint32(dst, static_cast<uint32_t>(fv.field.size()));
    dst->append(fv.field);
    PutVarint32(dst, static_cast<uint32_t>(fv.value.size()));
    dst->append(fv.value);
  }
}

rocksdb::Status DecodeInlineHashFields(Slice payload, uint64_t size, std::vector<FieldValue> *field_values) {
  field_values->clear();
  field_values->reserve(size);
  for (uint64_t i = 0; i < size; i++) {
    uint32_t field_len = 0;
    if (!GetVarint32(&payload, &field_len) || payload.size() < field_len) {
      return rocksdb::Status::Corruption("inline hash field is truncated");
    }
    Slice field(payload.data(), field_len);
    payload.remove_prefix(field_len);

    uint32_t value_len = 0;
    if (!GetVarint32(&payload, &value_len) || payload.size() < value_len) {
      return rocksdb::Status::Corruption("inline hash value is truncated");
    }
    Slice value(payload.data(), value_len);
    payload.remove_prefix(value_len);

    field_values->emplace_back(field.ToString(), value.ToString());
  }
  return rocksdb::Status::OK();
}

rocksdb::Status Hash::GetMetadata(engine::Context &ctx, const Slice &ns_key, HashMetadata *metadata) {
  return Database::GetMetadata(ctx, {kRedisHash}, ns_key, metadata);
}

rocksdb::Status Hash::GetMetadata(engine::Context &ctx, const Slice &ns_key, HashMetadata *metadata,
                                  std::vector<FieldValue> *inline_fields) {
  inline_fields->clear();
  std::string raw_value;
  Slice rest;
  auto s = Database::GetMetadata(ctx, {kRedisHash}, ns_key, &raw_value, metadata, &rest);
  if (!s.ok() || !metadata->IsInline()) return s;
  return DecodeInlineHashFields(rest, metadata->size, inline_fields);
}

bool Hash::inlineLayoutEnabled() const { return storage_->GetConfig()->hash_inline_enabled; }

bool Hash::fitsInline(const std::vector<FieldValue> &field_values, std::string *encoded) const {
  const auto *config = storage_->GetConfig();
  if (!config->hash_inline_enabled) return false;
  if (field_values.size() > static_cast<size_t>(config->hash_inline_max_fields)) return false;
  encoded->clear();
  EncodeInlineHashFields(field_values, encoded);
  return encoded->size() <= static_cast<size_t>(config->hash_inline_max_bytes);
}

rocksdb::Status Hash::writeInlineOrPromote(rocksdb::WriteBatchBase *batch, const Slice &ns_key, HashMetadata *metadata,
                                           const std::vector<FieldValue> &field_values) {
  metadata->size = field_values.size();
  std::string bytes;
  std::string payload;
  if (fitsInline(field_values, &payload)) {
    metadata->SetInline(true);
    metadata->Encode(&bytes);
    bytes.append(payload);
    return batch->Put(metadata_cf_handle_, ns_key, bytes);
  }

  // The hash was inline (or new), so no sub key exists for this version yet and
  // writing every field is enough to switch to the sub-key layout.
  metadata->SetInline(false);
  for (const auto &fv : field_values) {
    std::string sub_key = InternalKey(ns_key, fv.field, metadata->version, storage_->IsSlotIdEncoded()).Encode();
    auto s = batch->Put(sub_key, fv.value);
    if (!s.ok()) return s;
  }
  metadata->Encode(&bytes);
  return batch->Put(metadata_cf_handle_, ns_key, bytes);
}

rocksdb::Status Hash::Size(engine::Context &ctx, const Slice &user_key, uint64_t *size) {
  *size = 0;

  std::string ns_key = AppendNamespacePrefix(user_key);
  HashMetadata metadata(false);
  rocksdb::Status s = GetMetadata(ctx, ns_key, &metadata);
  if (!s.ok()) return s;
  *size = metadata.size;
  return rocksdb::Status::OK();
}

rocksdb::Status Hash::Get(engine::Context &ctx, const Slice &user_key, const Slice &field, std::string *value) {
  std::string ns_key = AppendNamespacePrefix(user_key);
  HashMetadata metadata(false);
  std::vector<FieldValue> inline_fields;
  rocksdb::Status s = GetMetadata(ctx, ns_key, &metadata, &inline_fields);
  if (!s.ok()) return s;
  if (metadata.IsInline()) {
    auto *fv = FindInlineField(inline_fields, field.ToStringView());
    if (fv == nullptr) return rocksdb::Status::NotFound();
    *value = fv->value;
    return rocksdb::Status::OK();
  }
  std::string sub_key = InternalKey(ns_key, field, metadata.version, storage_->IsSlotIdEncoded()).Encode();
  return storage_->Get(ctx, ctx.GetReadOptions(), sub_key, value);
}

rocksdb::Status Hash::IncrBy(engine::Context &ctx, const Slice &user_key, const Slice &field, int64_t increment,
                             int64_t *new_value) {
  bool exists = false;
  int64_t old_value = 0;

  std::string ns_key = AppendNamespacePrefix(user_key);

  HashMetadata metadata;
  std::vector<FieldValue> inline_fields;
  rocksdb::Status s = GetMetadata(ctx, ns_key, &metadata, &inline_fields);
  if (!s.ok() && !s.IsNotFound()) return s;
  const bool key_exists = s.ok();
  const bool use_inline = key_exists ? metadata.IsInline() : inlineLayoutEnabled();

  std::string sub_key = InternalKey(ns_key, field, metadata.version, storage_->IsSlotIdEncoded()).Encode();
  std::string value_bytes;
  if (key_exists) {
    if (metadata.IsInline()) {
      auto *fv = FindInlineField(inline_fields, field.ToStringView());
      if (fv != nullptr) {
        value_bytes = fv->value;
        exists = true;
      }
    } else {
      s = storage_->Get(ctx, ctx.GetReadOptions(), sub_key, &value_bytes);
      if (!s.ok() && !s.IsNotFound()) return s;
      exists = s.ok();
    }
    if (exists) {
      auto parse_result = ParseInt<int64_t>(value_bytes, 10);
      if (!parse_result) {
        return rocksdb::Status::InvalidArgument(parse_result.Msg());
      }
      if (isspace(value_bytes[0])) {
        return rocksdb::Status::InvalidArgument("value is not an integer");
      }
      old_value = *parse_result;
    }
  }
  if ((increment < 0 && old_value < 0 && increment < (LLONG_MIN - old_value)) ||
      (increment > 0 && old_value > 0 && increment > (LLONG_MAX - old_value))) {
    return rocksdb::Status::InvalidArgument("increment or decrement would overflow");
  }

  *new_value = old_value + increment;
  auto batch = storage_->GetWriteBatchBase();
  WriteBatchLogData log_data(kRedisHash);
  s = batch->PutLogData(log_data.Encode());
  if (!s.ok()) return s;
  if (use_inline) {
    UpsertInlineField(inline_fields, field.ToStringView(), std::to_string(*new_value));
    s = writeInlineOrPromote(batch.Get(), ns_key, &metadata, inline_fields);
    if (!s.ok()) return s;
    return storage_->Write(ctx, storage_->DefaultWriteOptions(), batch->GetWriteBatch());
  }
  s = batch->Put(sub_key, std::to_string(*new_value));
  if (!s.ok()) return s;
  if (!exists) {
    metadata.size += 1;
    std::string bytes;
    metadata.Encode(&bytes);
    s = batch->Put(metadata_cf_handle_, ns_key, bytes);
    if (!s.ok()) return s;
  }
  return storage_->Write(ctx, storage_->DefaultWriteOptions(), batch->GetWriteBatch());
}

rocksdb::Status Hash::IncrByFloat(engine::Context &ctx, const Slice &user_key, const Slice &field, double increment,
                                  double *new_value) {
  bool exists = false;
  double old_value = 0;

  std::string ns_key = AppendNamespacePrefix(user_key);

  HashMetadata metadata;
  std::vector<FieldValue> inline_fields;
  rocksdb::Status s = GetMetadata(ctx, ns_key, &metadata, &inline_fields);
  if (!s.ok() && !s.IsNotFound()) return s;
  const bool key_exists = s.ok();
  const bool use_inline = key_exists ? metadata.IsInline() : inlineLayoutEnabled();

  std::string sub_key = InternalKey(ns_key, field, metadata.version, storage_->IsSlotIdEncoded()).Encode();
  std::string value_bytes;
  if (key_exists) {
    if (metadata.IsInline()) {
      auto *fv = FindInlineField(inline_fields, field.ToStringView());
      if (fv != nullptr) {
        value_bytes = fv->value;
        exists = true;
      }
    } else {
      s = storage_->Get(ctx, ctx.GetReadOptions(), sub_key, &value_bytes);
      if (!s.ok() && !s.IsNotFound()) return s;
      exists = s.ok();
    }
    if (exists) {
      auto value_stat = ParseFloat(value_bytes);
      if (!value_stat || isspace(value_bytes[0])) {
        return rocksdb::Status::InvalidArgument("value is not a number");
      }
      old_value = *value_stat;
    }
  }
  double n = old_value + increment;
  if (std::isinf(n) || std::isnan(n)) {
    return rocksdb::Status::InvalidArgument("increment would produce NaN or Infinity");
  }

  *new_value = n;
  auto batch = storage_->GetWriteBatchBase();
  WriteBatchLogData log_data(kRedisHash);
  s = batch->PutLogData(log_data.Encode());
  if (!s.ok()) return s;
  if (use_inline) {
    UpsertInlineField(inline_fields, field.ToStringView(), std::to_string(*new_value));
    s = writeInlineOrPromote(batch.Get(), ns_key, &metadata, inline_fields);
    if (!s.ok()) return s;
    return storage_->Write(ctx, storage_->DefaultWriteOptions(), batch->GetWriteBatch());
  }
  s = batch->Put(sub_key, std::to_string(*new_value));
  if (!s.ok()) return s;
  if (!exists) {
    metadata.size += 1;
    std::string bytes;
    metadata.Encode(&bytes);
    s = batch->Put(metadata_cf_handle_, ns_key, bytes);
    if (!s.ok()) return s;
  }
  return storage_->Write(ctx, storage_->DefaultWriteOptions(), batch->GetWriteBatch());
}

rocksdb::Status Hash::MGet(engine::Context &ctx, const Slice &user_key, const std::vector<Slice> &fields,
                           std::vector<std::string> *values, std::vector<rocksdb::Status> *statuses) {
  values->clear();
  statuses->clear();

  std::string ns_key = AppendNamespacePrefix(user_key);
  HashMetadata metadata(false);
  std::vector<FieldValue> inline_fields;
  rocksdb::Status s = GetMetadata(ctx, ns_key, &metadata, &inline_fields);
  if (!s.ok()) {
    return s;
  }

  if (metadata.IsInline()) {
    values->reserve(fields.size());
    statuses->reserve(fields.size());
    for (const auto &field : fields) {
      auto *fv = FindInlineField(inline_fields, field.ToStringView());
      if (fv == nullptr) {
        values->emplace_back();
        statuses->emplace_back(rocksdb::Status::NotFound());
      } else {
        values->emplace_back(fv->value);
        statuses->emplace_back(rocksdb::Status::OK());
      }
    }
    return rocksdb::Status::OK();
  }

  rocksdb::ReadOptions read_options = ctx.DefaultMultiGetOptions();
  std::vector<rocksdb::Slice> keys;

  keys.reserve(fields.size());
  std::vector<std::string> sub_keys;
  sub_keys.resize(fields.size());
  for (size_t i = 0; i < fields.size(); i++) {
    auto &field = fields[i];
    sub_keys[i] = InternalKey(ns_key, field, metadata.version, storage_->IsSlotIdEncoded()).Encode();
    keys.emplace_back(sub_keys[i]);
  }

  std::vector<rocksdb::PinnableSlice> values_vector;
  values_vector.resize(keys.size());
  std::vector<rocksdb::Status> statuses_vector;
  statuses_vector.resize(keys.size());
  storage_->MultiGet(ctx, read_options, storage_->GetDB()->DefaultColumnFamily(), keys.size(), keys.data(),
                     values_vector.data(), statuses_vector.data());
  for (size_t i = 0; i < keys.size(); i++) {
    if (!statuses_vector[i].ok() && !statuses_vector[i].IsNotFound()) return statuses_vector[i];
    values->emplace_back(values_vector[i].ToString());
    statuses->emplace_back(statuses_vector[i]);
  }
  return rocksdb::Status::OK();
}

rocksdb::Status Hash::Set(engine::Context &ctx, const Slice &user_key, const Slice &field, const Slice &value,
                          uint64_t *added_cnt) {
  return MSet(ctx, user_key, {{field.ToString(), value.ToString()}}, false, added_cnt);
}

rocksdb::Status Hash::Delete(engine::Context &ctx, const Slice &user_key, const std::vector<Slice> &fields,
                             uint64_t *deleted_cnt) {
  *deleted_cnt = 0;
  std::string ns_key = AppendNamespacePrefix(user_key);

  HashMetadata metadata(false);
  auto batch = storage_->GetWriteBatchBase();
  WriteBatchLogData log_data(kRedisHash);
  auto s = batch->PutLogData(log_data.Encode());
  if (!s.ok()) return s;

  std::vector<FieldValue> inline_fields;
  s = GetMetadata(ctx, ns_key, &metadata, &inline_fields);
  if (!s.ok()) return s.IsNotFound() ? rocksdb::Status::OK() : s;

  if (metadata.IsInline()) {
    for (const auto &field : fields) {
      auto it = std::lower_bound(inline_fields.begin(), inline_fields.end(), field.ToStringView(), FieldLess{});
      if (it == inline_fields.end() || it->field != field.ToStringView()) continue;
      inline_fields.erase(it);
      *deleted_cnt += 1;
    }
    if (*deleted_cnt == 0) {
      return rocksdb::Status::OK();
    }
    if (inline_fields.empty()) {
      s = batch->Delete(metadata_cf_handle_, ns_key);
    } else {
      metadata.size = inline_fields.size();
      std::string bytes;
      metadata.Encode(&bytes);
      EncodeInlineHashFields(inline_fields, &bytes);
      s = batch->Put(metadata_cf_handle_, ns_key, bytes);
    }
    if (!s.ok()) return s;
    return storage_->Write(ctx, storage_->DefaultWriteOptions(), batch->GetWriteBatch());
  }

  std::string value;
  std::unordered_set<std::string_view> field_set;
  for (const auto &field : fields) {
    if (!field_set.emplace(field.ToStringView()).second) {
      continue;
    }
    std::string sub_key = InternalKey(ns_key, field, metadata.version, storage_->IsSlotIdEncoded()).Encode();
    s = storage_->Get(ctx, ctx.GetReadOptions(), sub_key, &value);
    if (s.ok()) {
      *deleted_cnt += 1;
      s = batch->Delete(sub_key);
      if (!s.ok()) return s;
    }
  }
  if (*deleted_cnt == 0) {
    return rocksdb::Status::OK();
  }
  metadata.size -= *deleted_cnt;
  std::string bytes;
  metadata.Encode(&bytes);
  s = batch->Put(metadata_cf_handle_, ns_key, bytes);
  if (!s.ok()) return s;
  return storage_->Write(ctx, storage_->DefaultWriteOptions(), batch->GetWriteBatch());
}

rocksdb::Status Hash::MSet(engine::Context &ctx, const Slice &user_key, const std::vector<FieldValue> &field_values,
                           bool nx, uint64_t *added_cnt, uint64_t expire) {
  *added_cnt = 0;
  std::string ns_key = AppendNamespacePrefix(user_key);

  HashMetadata metadata;
  std::vector<FieldValue> inline_fields;
  rocksdb::Status s = GetMetadata(ctx, ns_key, &metadata, &inline_fields);
  if (!s.ok() && !s.IsNotFound()) return s;
  const bool key_exists = s.ok();
  bool ttl_updated = false;
  if (expire > 0 && metadata.expire != expire) {
    metadata.expire = expire;
    ttl_updated = true;
  }
  int added = 0;
  auto batch = storage_->GetWriteBatchBase();
  WriteBatchLogData log_data(kRedisHash);
  s = batch->PutLogData(log_data.Encode());
  if (!s.ok()) return s;

  if (!key_exists && inlineLayoutEnabled()) {
    // A new hash: write it inline in a single metadata value when it fits.
    std::vector<FieldValue> fields = SortedUniqueFields(field_values);
    std::string payload;
    if (fitsInline(fields, &payload)) {
      metadata.SetInline(true);
      metadata.size = fields.size();
      *added_cnt = fields.size();
      std::string bytes;
      metadata.Encode(&bytes);
      bytes.append(payload);
      s = batch->Put(metadata_cf_handle_, ns_key, bytes);
      if (!s.ok()) return s;
      return storage_->Write(ctx, storage_->DefaultWriteOptions(), batch->GetWriteBatch());
    }
  }

  if (key_exists && metadata.IsInline()) {
    bool changed = false;
    std::unordered_set<std::string_view> field_set;
    for (auto it = field_values.rbegin(); it != field_values.rend(); it++) {
      if (!field_set.insert(it->field).second) {
        continue;
      }
      auto *fv = FindInlineField(inline_fields, it->field);
      if (fv != nullptr) {
        if (nx || fv->value == it->value) continue;
        fv->value = it->value;
      } else {
        UpsertInlineField(inline_fields, it->field, it->value);
        added++;
      }
      changed = true;
    }
    if (!changed && !ttl_updated) {
      return rocksdb::Status::OK();
    }
    *added_cnt = added;
    s = writeInlineOrPromote(batch.Get(), ns_key, &metadata, inline_fields);
    if (!s.ok()) return s;
    return storage_->Write(ctx, storage_->DefaultWriteOptions(), batch->GetWriteBatch());
  }

  std::unordered_set<std::string_view> field_set;

  std::vector<std::string> keys_encoded;
  std::vector<std::string_view> values;
  keys_encoded.reserve(field_values.size());
  values.reserve(field_values.size());
  for (auto it = field_values.rbegin(); it != field_values.rend(); it++) {
    if (!field_set.insert(it->field).second) {
      continue;
    }

    keys_encoded.push_back(InternalKey(ns_key, it->field, metadata.version, storage_->IsSlotIdEncoded()).Encode());
    values.emplace_back(it->value);
  }
  // Slices are created only after keys_encoded stops growing, so they can never
  // dangle because of a vector reallocation moving short (SSO) strings.
  std::vector<rocksdb::Slice> keys(keys_encoded.begin(), keys_encoded.end());

  std::vector<rocksdb::PinnableSlice> values_vector;
  std::vector<rocksdb::Status> statuses_vector;
  if (metadata.size > 0) {
    values_vector.resize(keys.size());
    statuses_vector.resize(keys.size());
    rocksdb::ReadOptions read_options = ctx.DefaultMultiGetOptions();
    storage_->MultiGet(ctx, read_options, storage_->GetDB()->DefaultColumnFamily(), keys.size(), keys.data(),
                       values_vector.data(), statuses_vector.data());

    uint64_t existing_cnt = 0;
    for (const auto &field_status : statuses_vector) {
      if (!field_status.ok() && !field_status.IsNotFound()) return field_status;
      if (field_status.ok()) existing_cnt++;
    }
    // The command rewrites every field the hash currently has, so the written
    // fields are the whole hash: switch it to the inline layout under a new
    // version and let the compaction filter reclaim the orphaned sub keys.
    if (!nx && inlineLayoutEnabled() && existing_cnt == metadata.size) {
      std::vector<FieldValue> fields = SortedUniqueFields(field_values);
      std::string payload;
      if (fitsInline(fields, &payload)) {
        metadata.RegenerateVersion();
        metadata.SetInline(true);
        metadata.size = fields.size();
        *added_cnt = fields.size() - existing_cnt;
        std::string bytes;
        metadata.Encode(&bytes);
        bytes.append(payload);
        s = batch->Put(metadata_cf_handle_, ns_key, bytes);
        if (!s.ok()) return s;
        return storage_->Write(ctx, storage_->DefaultWriteOptions(), batch->GetWriteBatch());
      }
    }
  }

  for (size_t field_index = 0; field_index < keys.size(); field_index++) {
    const rocksdb::Slice field_key = keys[field_index];
    bool exists = false;

    if (metadata.size > 0) {
      rocksdb::Status &field_status = statuses_vector[field_index];
      if (field_status.ok()) {
        if (nx || values_vector[field_index] == values[field_index]) {
          continue;
        }
        exists = true;
      }
    }

    if (!exists) {
      added++;
    }

    s = batch->Put(field_key, values[field_index]);
    if (!s.ok()) return s;
  }

  if (added > 0 || ttl_updated) {
    *added_cnt = added;
    metadata.size += added;
    std::string bytes;
    metadata.Encode(&bytes);
    s = batch->Put(metadata_cf_handle_, ns_key, bytes);
    if (!s.ok()) return s;
  }

  return storage_->Write(ctx, storage_->DefaultWriteOptions(), batch->GetWriteBatch());
}

rocksdb::Status Hash::RangeByLex(engine::Context &ctx, const Slice &user_key, const RangeLexSpec &spec,
                                 std::vector<FieldValue> *field_values) {
  field_values->clear();
  if (spec.count == 0) {
    return rocksdb::Status::OK();
  }
  std::string ns_key = AppendNamespacePrefix(user_key);
  HashMetadata metadata(false);
  std::vector<FieldValue> inline_fields;
  rocksdb::Status s = GetMetadata(ctx, ns_key, &metadata, &inline_fields);
  if (!s.ok()) return s.IsNotFound() ? rocksdb::Status::OK() : s;

  if (metadata.IsInline()) {
    auto below_min = [&spec](const std::string &field) {
      return field < spec.min || (spec.minex && field == spec.min);
    };
    auto above_max = [&spec](const std::string &field) {
      return (spec.maxex && field == spec.max) || (!spec.max_infinite && field > spec.max);
    };
    int64_t pos = 0;
    auto emit = [&](const FieldValue &fv) {
      if (spec.offset >= 0 && pos++ < spec.offset) return false;
      field_values->emplace_back(fv.field, fv.value);
      return spec.count > 0 && field_values->size() >= static_cast<unsigned>(spec.count);
    };
    if (!spec.reversed) {
      for (const auto &fv : inline_fields) {
        if (below_min(fv.field)) continue;
        if (above_max(fv.field)) break;
        if (emit(fv)) break;
      }
    } else {
      for (auto it = inline_fields.rbegin(); it != inline_fields.rend(); ++it) {
        if (above_max(it->field)) continue;
        if (below_min(it->field)) break;
        if (emit(*it)) break;
      }
    }
    return rocksdb::Status::OK();
  }

  std::string start_member = spec.reversed ? spec.max : spec.min;
  std::string start_key = InternalKey(ns_key, start_member, metadata.version, storage_->IsSlotIdEncoded()).Encode();
  std::string prefix_key = InternalKey(ns_key, "", metadata.version, storage_->IsSlotIdEncoded()).Encode();
  std::string next_version_prefix_key =
      InternalKey(ns_key, "", metadata.version + 1, storage_->IsSlotIdEncoded()).Encode();
  rocksdb::ReadOptions read_options = ctx.DefaultScanOptions();
  rocksdb::Slice upper_bound(next_version_prefix_key);
  read_options.iterate_upper_bound = &upper_bound;
  rocksdb::Slice lower_bound(prefix_key);
  read_options.iterate_lower_bound = &lower_bound;

  auto iter = util::UniqueIterator(ctx, read_options);
  if (!spec.reversed) {
    iter->Seek(start_key);
  } else {
    if (spec.max_infinite) {
      iter->SeekToLast();
    } else {
      iter->SeekForPrev(start_key);
    }
  }
  int64_t pos = 0;
  for (; iter->Valid() && iter->key().starts_with(prefix_key); (!spec.reversed ? iter->Next() : iter->Prev())) {
    InternalKey ikey(iter->key(), storage_->IsSlotIdEncoded());
    if (spec.reversed) {
      if (ikey.GetSubKey().ToString() < spec.min || (spec.minex && ikey.GetSubKey().ToString() == spec.min)) {
        break;
      }
      if ((spec.maxex && ikey.GetSubKey().ToString() == spec.max) ||
          (!spec.max_infinite && ikey.GetSubKey().ToString() > spec.max)) {
        continue;
      }
    } else {
      if (spec.minex && ikey.GetSubKey().ToString() == spec.min) continue;  // the min member was exclusive
      if ((spec.maxex && ikey.GetSubKey().ToString() == spec.max) ||
          (!spec.max_infinite && ikey.GetSubKey().ToString() > spec.max))
        break;
    }
    if (spec.offset >= 0 && pos++ < spec.offset) continue;

    field_values->emplace_back(ikey.GetSubKey().ToString(), iter->value().ToString());
    if (spec.count > 0 && field_values->size() >= static_cast<unsigned>(spec.count)) break;
  }
  return rocksdb::Status::OK();
}

rocksdb::Status Hash::GetAll(engine::Context &ctx, const Slice &user_key, std::vector<FieldValue> *field_values,
                             HashFetchType type) {
  field_values->clear();

  std::string ns_key = AppendNamespacePrefix(user_key);
  HashMetadata metadata(false);
  std::vector<FieldValue> inline_fields;
  rocksdb::Status s = GetMetadata(ctx, ns_key, &metadata, &inline_fields);
  if (!s.ok()) return s.IsNotFound() ? rocksdb::Status::OK() : s;

  if (metadata.IsInline()) {
    if (type == HashFetchType::kAll) {
      *field_values = std::move(inline_fields);
      return rocksdb::Status::OK();
    }
    field_values->reserve(inline_fields.size());
    for (auto &fv : inline_fields) {
      if (type == HashFetchType::kOnlyKey) {
        field_values->emplace_back(std::move(fv.field), "");
      } else {
        field_values->emplace_back("", std::move(fv.value));
      }
    }
    return rocksdb::Status::OK();
  }

  std::string prefix_key = InternalKey(ns_key, "", metadata.version, storage_->IsSlotIdEncoded()).Encode();
  std::string next_version_prefix_key =
      InternalKey(ns_key, "", metadata.version + 1, storage_->IsSlotIdEncoded()).Encode();

  rocksdb::ReadOptions read_options = ctx.DefaultSingleKeyScanOptions(metadata.size);
  rocksdb::Slice upper_bound(next_version_prefix_key);
  read_options.iterate_upper_bound = &upper_bound;

  auto iter = util::UniqueIterator(ctx, read_options);
  for (iter->Seek(prefix_key); iter->Valid() && iter->key().starts_with(prefix_key); iter->Next()) {
    if (type == HashFetchType::kOnlyKey) {
      InternalKey ikey(iter->key(), storage_->IsSlotIdEncoded());
      field_values->emplace_back(ikey.GetSubKey().ToString(), "");
    } else if (type == HashFetchType::kOnlyValue) {
      field_values->emplace_back("", iter->value().ToString());
    } else {
      InternalKey ikey(iter->key(), storage_->IsSlotIdEncoded());
      field_values->emplace_back(ikey.GetSubKey().ToString(), iter->value().ToString());
    }
  }
  return rocksdb::Status::OK();
}

rocksdb::Status Hash::Scan(engine::Context &ctx, const Slice &user_key, const std::string &cursor, uint64_t limit,
                           const std::string &field_prefix, std::vector<std::string> *fields,
                           std::vector<std::string> *values) {
  std::string ns_key = AppendNamespacePrefix(user_key);
  HashMetadata metadata(false);
  std::vector<FieldValue> inline_fields;
  rocksdb::Status s = GetMetadata(ctx, ns_key, &metadata, &inline_fields);
  if (!s.ok()) return s;
  if (!metadata.IsInline()) {
    return SubKeyScanner::Scan(ctx, kRedisHash, user_key, cursor, limit, field_prefix, fields, values);
  }

  // Same contract as SubKeyScanner::Scan: resume right after the cursor field and stop at the
  // first field that no longer matches the prefix.
  const std::string &start = cursor.empty() ? field_prefix : cursor;
  auto it = std::lower_bound(inline_fields.begin(), inline_fields.end(), std::string_view(start), FieldLess{});
  uint64_t cnt = 0;
  for (; it != inline_fields.end(); ++it) {
    if (!cursor.empty() && it->field == cursor) continue;
    if (it->field.compare(0, field_prefix.size(), field_prefix) != 0) break;
    fields->emplace_back(it->field);
    if (values != nullptr) values->emplace_back(it->value);
    cnt++;
    if (limit > 0 && cnt >= limit) break;
  }
  return rocksdb::Status::OK();
}

rocksdb::Status Hash::RandField(engine::Context &ctx, const Slice &user_key, int64_t command_count,
                                std::vector<FieldValue> *field_values, HashFetchType type) {
  uint64_t count = (command_count >= 0) ? static_cast<uint64_t>(command_count) : static_cast<uint64_t>(-command_count);
  bool unique = (command_count >= 0);

  std::string ns_key = AppendNamespacePrefix(user_key);
  HashMetadata metadata(/*generate_version=*/false);
  rocksdb::Status s = GetMetadata(ctx, ns_key, &metadata);
  if (!s.ok()) return s;

  std::vector<FieldValue> samples;
  // TODO: Getting all values in Hash might be heavy, consider lazy-loading these values later
  if (count == 0) return rocksdb::Status::OK();
  s = ExtractRandMemberFromSet<FieldValue>(
      unique, count,
      [this, user_key, type, &ctx](std::vector<FieldValue> *elements) {
        return this->GetAll(ctx, user_key, elements, type);
      },
      field_values);
  if (!s.ok()) {
    return s;
  }
  switch (type) {
    case HashFetchType::kAll:
      break;
    case HashFetchType::kOnlyKey: {
      // GetAll should only fetching the key, checking all the values is empty
      for (const FieldValue &value : *field_values) {
        CHECK(value.value.empty());
      }
      break;
    }
    case HashFetchType::kOnlyValue:
      unreachable();
  }
  return rocksdb::Status::OK();
}

}  // namespace redis
