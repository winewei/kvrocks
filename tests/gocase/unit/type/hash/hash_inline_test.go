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

package hash

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"

	"github.com/apache/kvrocks/tests/gocase/util"
)

// hashEncoding returns the "encoding" entry of OBJECT DUMP: "inline" or "subkey".
func hashEncoding(t *testing.T, rdb *redis.Client, key string) string {
	infos, err := rdb.Do(context.Background(), "OBJECT", "DUMP", key).Slice()
	require.NoError(t, err)
	for i := 0; i+1 < len(infos); i += 2 {
		if infos[i] == "encoding" {
			return infos[i+1].(string)
		}
	}
	require.Failf(t, "missing encoding", "OBJECT DUMP %s has no encoding entry: %v", key, infos)
	return ""
}

func sortedFields(fields map[string]string) []string {
	keys := make([]string, 0, len(fields))
	for k := range fields {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func TestHashInline(t *testing.T) {
	srv := util.StartServer(t, map[string]string{
		"hash-inline-enabled":    "yes",
		"hash-inline-max-fields": "8",
		"hash-inline-max-bytes":  "512",
	})
	defer srv.Close()

	rdb := srv.NewClient()
	defer func() { require.NoError(t, rdb.Close()) }()

	ctx := context.Background()

	t.Run("small hash is stored inline and every read command sees it", func(t *testing.T) {
		require.NoError(t, rdb.Del(ctx, "h").Err())
		require.EqualValues(t, 3, rdb.HSet(ctx, "h", "b", "2", "a", "1", "c", "3").Val())
		require.Equal(t, "inline", hashEncoding(t, rdb, "h"))
		require.Equal(t, "hash", rdb.Type(ctx, "h").Val())
		require.EqualValues(t, 3, rdb.HLen(ctx, "h").Val())
		require.Equal(t, "1", rdb.HGet(ctx, "h", "a").Val())
		require.Equal(t, redis.Nil, rdb.HGet(ctx, "h", "zz").Err())
		require.True(t, rdb.HExists(ctx, "h", "c").Val())
		require.False(t, rdb.HExists(ctx, "h", "d").Val())
		require.EqualValues(t, 1, rdb.HStrLen(ctx, "h", "b").Val())
		require.Equal(t, []interface{}{"1", nil, "3"}, rdb.HMGet(ctx, "h", "a", "nope", "c").Val())
		require.Equal(t, map[string]string{"a": "1", "b": "2", "c": "3"}, rdb.HGetAll(ctx, "h").Val())
		// The inline layout keeps the fields sorted, exactly like the sub-key layout iterates them.
		require.Equal(t, []string{"a", "b", "c"}, rdb.HKeys(ctx, "h").Val())
		require.Equal(t, []string{"1", "2", "3"}, rdb.HVals(ctx, "h").Val())
		require.Len(t, rdb.HRandField(ctx, "h", 2).Val(), 2)
		require.Len(t, rdb.HRandFieldWithValues(ctx, "h", -5).Val(), 5)
	})

	t.Run("updates of an inline hash", func(t *testing.T) {
		require.NoError(t, rdb.Del(ctx, "h").Err())
		require.EqualValues(t, 2, rdb.HSet(ctx, "h", "a", "1", "b", "2").Val())
		// Overwrite one field, add one field, duplicate field in one command (last wins).
		require.EqualValues(t, 1, rdb.HSet(ctx, "h", "a", "10", "c", "x", "c", "3").Val())
		require.Equal(t, map[string]string{"a": "10", "b": "2", "c": "3"}, rdb.HGetAll(ctx, "h").Val())
		require.Equal(t, "inline", hashEncoding(t, rdb, "h"))

		require.False(t, rdb.HSetNX(ctx, "h", "a", "no").Val())
		require.True(t, rdb.HSetNX(ctx, "h", "d", "4").Val())
		require.Equal(t, "10", rdb.HGet(ctx, "h", "a").Val())
		require.Equal(t, "4", rdb.HGet(ctx, "h", "d").Val())

		require.EqualValues(t, 15, rdb.HIncrBy(ctx, "h", "a", 5).Val())
		require.EqualValues(t, 7, rdb.HIncrBy(ctx, "h", "n", 7).Val())
		require.InDelta(t, 1.5, rdb.HIncrByFloat(ctx, "h", "f", 1.5).Val(), 1e-9)
		require.InDelta(t, 4.0, rdb.HIncrByFloat(ctx, "h", "f", 2.5).Val(), 1e-9)
		require.Error(t, rdb.HIncrBy(ctx, "h", "f", 1).Err())
		require.EqualValues(t, 6, rdb.HLen(ctx, "h").Val())
		require.Equal(t, "inline", hashEncoding(t, rdb, "h"))

		require.EqualValues(t, 2, rdb.HDel(ctx, "h", "a", "missing", "n", "n").Val())
		require.Equal(t, map[string]string{"b": "2", "c": "3", "d": "4", "f": "4.000000"}, rdb.HGetAll(ctx, "h").Val())
		require.EqualValues(t, 0, rdb.HDel(ctx, "h", "missing").Val())
		require.EqualValues(t, 4, rdb.HDel(ctx, "h", "b", "c", "d", "f").Val())
		require.EqualValues(t, 0, rdb.Exists(ctx, "h").Val())
		require.Equal(t, "none", rdb.Type(ctx, "h").Val())
	})

	t.Run("hash exceeding the field limit is promoted to the sub-key layout", func(t *testing.T) {
		require.NoError(t, rdb.Del(ctx, "h").Err())
		fields := map[string]string{}
		for i := 0; i < 8; i++ {
			fields[fmt.Sprintf("f%02d", i)] = fmt.Sprintf("v%d", i)
		}
		require.NoError(t, rdb.HSet(ctx, "h", fields).Err())
		require.Equal(t, "inline", hashEncoding(t, rdb, "h"))

		fields["f08"] = "v8"
		require.EqualValues(t, 1, rdb.HSet(ctx, "h", "f08", "v8").Val())
		require.Equal(t, "subkey", hashEncoding(t, rdb, "h"))
		require.Equal(t, fields, rdb.HGetAll(ctx, "h").Val())
		require.Equal(t, sortedFields(fields), rdb.HKeys(ctx, "h").Val())
		require.EqualValues(t, 9, rdb.HLen(ctx, "h").Val())

		// Shrinking does not demote: the layout only changes on a full overwrite.
		require.EqualValues(t, 2, rdb.HDel(ctx, "h", "f08", "f07").Val())
		require.Equal(t, "subkey", hashEncoding(t, rdb, "h"))
		require.EqualValues(t, 7, rdb.HLen(ctx, "h").Val())

		// A hash created above the limit is never inline.
		require.NoError(t, rdb.Del(ctx, "h").Err())
		fields["f08"] = "v8"
		require.NoError(t, rdb.HSet(ctx, "h", fields).Err())
		require.Equal(t, "subkey", hashEncoding(t, rdb, "h"))
		require.Equal(t, fields, rdb.HGetAll(ctx, "h").Val())
	})

	t.Run("hash exceeding the byte limit is promoted by HINCRBY too", func(t *testing.T) {
		require.NoError(t, rdb.Del(ctx, "h").Err())
		big := strings.Repeat("x", 400)
		require.NoError(t, rdb.HSet(ctx, "h", "a", big).Err())
		require.Equal(t, "inline", hashEncoding(t, rdb, "h"))
		require.EqualValues(t, 1, rdb.HIncrBy(ctx, "h", "cnt", 1).Val())
		require.Equal(t, "inline", hashEncoding(t, rdb, "h"))
		require.NoError(t, rdb.HSet(ctx, "h", "b", strings.Repeat("y", 200)).Err())
		require.Equal(t, "subkey", hashEncoding(t, rdb, "h"))
		require.Equal(t, map[string]string{"a": big, "cnt": "1", "b": strings.Repeat("y", 200)},
			rdb.HGetAll(ctx, "h").Val())
		require.EqualValues(t, 2, rdb.HIncrBy(ctx, "h", "cnt", 1).Val())
	})

	t.Run("HSCAN and HRANGEBYLEX on inline hashes match the sub-key layout", func(t *testing.T) {
		fields := map[string]string{}
		for _, f := range []string{"apple", "apricot", "banana", "blueberry", "cherry", "date", "fig"} {
			fields[f] = strings.ToUpper(f)
		}
		require.NoError(t, rdb.Del(ctx, "small", "large").Err())
		require.NoError(t, rdb.HSet(ctx, "small", fields).Err())
		require.Equal(t, "inline", hashEncoding(t, rdb, "small"))
		// The same content in the sub-key layout, forced by lowering the byte limit.
		require.NoError(t, rdb.ConfigSet(ctx, "hash-inline-max-bytes", "1").Err())
		require.NoError(t, rdb.HSet(ctx, "large", fields).Err())
		require.NoError(t, rdb.ConfigSet(ctx, "hash-inline-max-bytes", "512").Err())
		require.Equal(t, "subkey", hashEncoding(t, rdb, "large"))

		for _, args := range [][]interface{}{
			{"-", "+"}, {"[b", "+"}, {"-", "(cherry"}, {"[apricot", "[date"}, {"(apricot", "(date"},
			{"[b", "[c"}, {"-", "+", "LIMIT", 1, 3}, {"-", "+", "LIMIT", 2, -1}, {"[zzz", "+"},
			{"+", "[b", "REV"}, {"+", "-", "REV"}, {"(date", "(apple", "REV"}, {"[fig", "[banana", "REV", "LIMIT", 1, 2},
		} {
			small, err := rdb.Do(ctx, append([]interface{}{"HRANGEBYLEX", "small"}, args...)...).Slice()
			require.NoError(t, err, "HRANGEBYLEX %v", args)
			large, err := rdb.Do(ctx, append([]interface{}{"HRANGEBYLEX", "large"}, args...)...).Slice()
			require.NoError(t, err, "HRANGEBYLEX %v", args)
			require.Equal(t, large, small, "HRANGEBYLEX %v", args)
		}
		nonEmpty, err := rdb.Do(ctx, "HRANGEBYLEX", "small", "[b", "(cherry").Slice()
		require.NoError(t, err)
		require.Equal(t, []interface{}{"banana", "BANANA", "blueberry", "BLUEBERRY"}, nonEmpty)

		for _, match := range []string{"", "a*", "b*", "c*", "z*"} {
			smallFields, largeFields := []string{}, []string{}
			smallCursor, largeCursor := uint64(0), uint64(0)
			for {
				keys, next, err := rdb.HScan(ctx, "small", smallCursor, match, 2).Result()
				require.NoError(t, err)
				smallFields = append(smallFields, keys...)
				smallCursor = next
				if next == 0 {
					break
				}
			}
			for {
				keys, next, err := rdb.HScan(ctx, "large", largeCursor, match, 2).Result()
				require.NoError(t, err)
				largeFields = append(largeFields, keys...)
				largeCursor = next
				if next == 0 {
					break
				}
			}
			require.Equal(t, largeFields, smallFields, "HSCAN MATCH %q", match)
		}
		all, _, err := rdb.HScan(ctx, "small", 0, "", 100).Result()
		require.NoError(t, err)
		require.Len(t, all, 2*len(fields))
	})

	t.Run("sub-key hash is converted on a full overwrite only", func(t *testing.T) {
		require.NoError(t, rdb.Del(ctx, "h").Err())
		require.NoError(t, rdb.ConfigSet(ctx, "hash-inline-enabled", "no").Err())
		require.NoError(t, rdb.HSet(ctx, "h", "a", "1", "b", "2", "c", "3").Err())
		require.Equal(t, "subkey", hashEncoding(t, rdb, "h"))
		require.NoError(t, rdb.ConfigSet(ctx, "hash-inline-enabled", "yes").Err())

		// Partial writes keep the sub-key layout.
		require.EqualValues(t, 0, rdb.HSet(ctx, "h", "a", "10").Val())
		require.EqualValues(t, 1, rdb.HSet(ctx, "h", "d", "4").Val())
		require.Equal(t, "subkey", hashEncoding(t, rdb, "h"))
		require.Equal(t, map[string]string{"a": "10", "b": "2", "c": "3", "d": "4"}, rdb.HGetAll(ctx, "h").Val())

		// HSETNX never converts.
		require.False(t, rdb.HSetNX(ctx, "h", "a", "x").Val())
		require.Equal(t, "subkey", hashEncoding(t, rdb, "h"))

		// Rewriting every existing field (plus a new one) converts the hash.
		require.EqualValues(t, 1, rdb.HSet(ctx, "h", "a", "100", "b", "2", "c", "3", "d", "4", "e", "5").Val())
		require.Equal(t, "inline", hashEncoding(t, rdb, "h"))
		require.Equal(t, map[string]string{"a": "100", "b": "2", "c": "3", "d": "4", "e": "5"},
			rdb.HGetAll(ctx, "h").Val())

		// The orphaned sub keys of the old version are invisible and get compacted away.
		require.NoError(t, rdb.Do(ctx, "COMPACT").Err())
		time.Sleep(time.Second)
		require.Equal(t, map[string]string{"a": "100", "b": "2", "c": "3", "d": "4", "e": "5"},
			rdb.HGetAll(ctx, "h").Val())
		require.EqualValues(t, 5, rdb.HLen(ctx, "h").Val())

		// Growing past the limit afterwards must not resurrect the old sub keys.
		fields := map[string]string{}
		for i := 0; i < 9; i++ {
			fields[fmt.Sprintf("n%d", i)] = "v"
		}
		require.EqualValues(t, 9, rdb.HSet(ctx, "h", fields).Val())
		require.Equal(t, "subkey", hashEncoding(t, rdb, "h"))
		require.EqualValues(t, 14, rdb.HLen(ctx, "h").Val())
		got := rdb.HGetAll(ctx, "h").Val()
		require.Len(t, got, 14)
		require.Equal(t, "100", got["a"])
	})

	t.Run("disabling the option promotes inline hashes on their next write", func(t *testing.T) {
		require.NoError(t, rdb.Del(ctx, "h", "untouched").Err())
		require.NoError(t, rdb.HSet(ctx, "h", "a", "1", "b", "2").Err())
		require.NoError(t, rdb.HSet(ctx, "untouched", "a", "1").Err())
		require.Equal(t, "inline", hashEncoding(t, rdb, "h"))
		require.NoError(t, rdb.ConfigSet(ctx, "hash-inline-enabled", "no").Err())
		defer func() { require.NoError(t, rdb.ConfigSet(ctx, "hash-inline-enabled", "yes").Err()) }()

		// Reads keep working without the option.
		require.Equal(t, map[string]string{"a": "1"}, rdb.HGetAll(ctx, "untouched").Val())
		require.Equal(t, "inline", hashEncoding(t, rdb, "untouched"))

		require.EqualValues(t, 1, rdb.HSet(ctx, "h", "c", "3").Val())
		require.Equal(t, "subkey", hashEncoding(t, rdb, "h"))
		require.Equal(t, map[string]string{"a": "1", "b": "2", "c": "3"}, rdb.HGetAll(ctx, "h").Val())
		require.EqualValues(t, 1, rdb.HDel(ctx, "h", "a").Val())
		require.Equal(t, map[string]string{"b": "2", "c": "3"}, rdb.HGetAll(ctx, "h").Val())

		require.NoError(t, rdb.HSet(ctx, "h2", "a", "1").Err())
		require.Equal(t, "subkey", hashEncoding(t, rdb, "h2"))
	})

	t.Run("generic key commands preserve inline hashes", func(t *testing.T) {
		require.NoError(t, rdb.Del(ctx, "h", "h-copy", "h-renamed").Err())
		require.NoError(t, rdb.HSet(ctx, "h", "a", "1", "b", "2").Err())

		require.NoError(t, rdb.Expire(ctx, "h", 100*time.Second).Err())
		require.Greater(t, rdb.TTL(ctx, "h").Val(), 90*time.Second)
		require.Equal(t, map[string]string{"a": "1", "b": "2"}, rdb.HGetAll(ctx, "h").Val())
		require.True(t, rdb.Persist(ctx, "h").Val())
		require.Equal(t, time.Duration(-1), rdb.TTL(ctx, "h").Val())

		require.EqualValues(t, 1, rdb.Copy(ctx, "h", "h-copy", 0, false).Val())
		require.Equal(t, "inline", hashEncoding(t, rdb, "h-copy"))
		require.Equal(t, map[string]string{"a": "1", "b": "2"}, rdb.HGetAll(ctx, "h-copy").Val())
		require.NoError(t, rdb.HSet(ctx, "h-copy", "c", "3").Err())
		require.Equal(t, map[string]string{"a": "1", "b": "2"}, rdb.HGetAll(ctx, "h").Val())

		require.NoError(t, rdb.Rename(ctx, "h", "h-renamed").Err())
		require.EqualValues(t, 0, rdb.Exists(ctx, "h").Val())
		require.Equal(t, "inline", hashEncoding(t, rdb, "h-renamed"))
		require.Equal(t, map[string]string{"a": "1", "b": "2"}, rdb.HGetAll(ctx, "h-renamed").Val())

		// A hash with a TTL written through HSETEXPIRE stays inline and expires as a whole.
		require.NoError(t, rdb.Do(ctx, "HSETEXPIRE", "h-ttl", 1, "a", "1").Err())
		require.Equal(t, "inline", hashEncoding(t, rdb, "h-ttl"))
		require.Eventually(t, func() bool { return rdb.Exists(ctx, "h-ttl").Val() == 0 }, 5*time.Second, 100*time.Millisecond)

		require.NoError(t, rdb.Set(ctx, "h-renamed", "string-now", 0).Err())
		require.Equal(t, "string", rdb.Type(ctx, "h-renamed").Val())
		require.NoError(t, rdb.Del(ctx, "h-renamed").Err())
		require.NoError(t, rdb.HSet(ctx, "h-renamed", "z", "26").Err())
		require.Equal(t, map[string]string{"z": "26"}, rdb.HGetAll(ctx, "h-renamed").Val())
	})

	t.Run("binary-safe fields and values", func(t *testing.T) {
		require.NoError(t, rdb.Del(ctx, "h").Err())
		fields := map[string]string{
			"":                       "empty-field",
			"\x00\x01":               "",
			"\xff\xfe":               "\x00binary\xff",
			"a b":                    "with space",
			strings.Repeat("k", 100): strings.Repeat("v", 300),
		}
		require.NoError(t, rdb.HSet(ctx, "h", fields).Err())
		require.Equal(t, "inline", hashEncoding(t, rdb, "h"))
		require.Equal(t, fields, rdb.HGetAll(ctx, "h").Val())
		// Bytewise order: "" < "\x00\x01" < "a b" < "kkk..." < "\xff\xfe".
		require.Equal(t, []string{"", "\x00\x01", "a b", strings.Repeat("k", 100), "\xff\xfe"}, rdb.HKeys(ctx, "h").Val())
		require.Equal(t, "", rdb.HGet(ctx, "h", "\x00\x01").Val())
		require.Equal(t, "empty-field", rdb.HGet(ctx, "h", "").Val())
	})
}
