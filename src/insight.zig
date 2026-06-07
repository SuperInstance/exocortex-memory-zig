//! Insight type — the fundamental unit of memory in the exocortex store.
//!
//! Each Insight represents a discrete piece of knowledge with associated
//! metadata: tags for categorization, confidence for uncertainty, a timestamp,
//! and an optional ternary embedding for semantic similarity search.

const std = @import("std");
const embedding = @import("embedding.zig");

pub const TagLen = 40;
pub const MaxTags = 16;
pub const Tag = [TagLen]u8;

/// Core insight struct, parameterized by embedding dimension.
pub fn Insight(comptime dim: usize) type {
    return struct {
        id: u64,
        text: []const u8,
        tags: [MaxTags]Tag,
        tags_count: usize,
        confidence: f64,
        timestamp: i64,
        embedding: ?embedding.TernaryVector(dim),

        const Self = @This();

        /// Create a builder for constructing insights incrementally.
        pub fn builder(id: u64) InsightBuilder(dim) {
            return InsightBuilder(dim).init(id);
        }

        /// Check if this insight has a specific tag.
        pub fn hasTag(self: Self, tag: []const u8) bool {
            for (self.tags[0..self.tags_count]) |t| {
                var end: usize = 0;
                while (end < TagLen and t[end] != 0) end += 1;
                if (std.mem.eql(u8, t[0..end], tag)) return true;
            }
            return false;
        }

        /// Check overlap between this insight's tags and another set.
        pub fn tagsOverlap(self: Self, other: Self) usize {
            var count: usize = 0;
            for (self.tags[0..self.tags_count]) |t1| {
                var end1: usize = 0;
                while (end1 < TagLen and t1[end1] != 0) end1 += 1;
                const s1 = t1[0..end1];
                for (other.tags[0..other.tags_count]) |t2| {
                    var end2: usize = 0;
                    while (end2 < TagLen and t2[end2] != 0) end2 += 1;
                    const s2 = t2[0..end2];
                    if (std.mem.eql(u8, s1, s2)) {
                        count += 1;
                        break;
                    }
                }
            }
            return count;
        }

        /// Check if similarity exceeds a threshold (requires embeddings).
        pub fn similarityThreshold(self: Self, other: Self, threshold: f64) bool {
            const a = self.embedding orelse return false;
            const b = other.embedding orelse return false;
            return a.cosineSimilarity(b) >= threshold;
        }
    };
}

/// Builder for Insight structs.
pub fn InsightBuilder(comptime dim: usize) type {
    return struct {
        id: u64,
        text: []const u8,
        tags: [MaxTags]Tag,
        tags_count: usize,
        confidence: f64,
        timestamp: i64,
        embedding: ?embedding.TernaryVector(dim),

        const Self = @This();

        pub fn init(id: u64) Self {
            return .{
                .id = id,
                .text = "",
                .tags = [_]Tag{[_]u8{0} ** TagLen} ** MaxTags,
                .tags_count = 0,
                .confidence = 0.0,
                .timestamp = 0,
                .embedding = null,
            };
        }

        pub fn setText(self: Self, t: []const u8) Self {
            var s = self;
            s.text = t;
            return s;
        }

        pub fn addTag(self: Self, t: []const u8) Self {
            var s = self;
            if (s.tags_count < MaxTags and t.len <= TagLen) {
                s.tags[s.tags_count] = [_]u8{0} ** TagLen;
                @memcpy(s.tags[s.tags_count][0..t.len], t[0..t.len]);
                s.tags_count += 1;
            }
            return s;
        }

        pub fn setConfidence(self: Self, c: f64) Self {
            var s = self;
            s.confidence = c;
            return s;
        }

        pub fn setTimestamp(self: Self, ts: i64) Self {
            var s = self;
            s.timestamp = ts;
            return s;
        }

        pub fn setEmbedding(self: Self, emb: embedding.TernaryVector(dim)) Self {
            var s = self;
            s.embedding = emb;
            return s;
        }

        pub fn build(self: Self) Insight(dim) {
            return .{
                .id = self.id,
                .text = self.text,
                .tags = self.tags,
                .tags_count = self.tags_count,
                .confidence = self.confidence,
                .timestamp = self.timestamp,
                .embedding = self.embedding,
            };
        }
    };
}

/// Helper: create a Tag from a string literal.
pub fn makeTag(s: []const u8) Tag {
    var buf: Tag = [_]u8{0} ** TagLen;
    const len = @min(s.len, TagLen);
    @memcpy(buf[0..len], s[0..len]);
    return buf;
}

test "Insight builder produces valid insight" {
    const Dim = 8;
    const ins = Insight(Dim).builder(1)
        .setText("hello world")
        .addTag("greeting")
        .setConfidence(0.9)
        .setTimestamp(1000)
        .build();
    try std.testing.expectEqual(@as(u64, 1), ins.id);
    try std.testing.expectEqualStrings("hello world", ins.text);
    try std.testing.expectEqual(@as(f64, 0.9), ins.confidence);
    try std.testing.expectEqual(@as(usize, 1), ins.tags_count);
}

test "Insight hasTag" {
    const Dim = 8;
    const ins = Insight(Dim).builder(1)
        .addTag("zig")
        .addTag("comptime")
        .build();
    try std.testing.expect(ins.hasTag("zig"));
    try std.testing.expect(ins.hasTag("comptime"));
    try std.testing.expect(!ins.hasTag("rust"));
}

test "Insight tagsOverlap" {
    const Dim = 8;
    const a = Insight(Dim).builder(1).addTag("a").addTag("b").build();
    const b = Insight(Dim).builder(2).addTag("b").addTag("c").build();
    try std.testing.expectEqual(@as(usize, 1), a.tagsOverlap(b));
}

test "Insight similarityThreshold" {
    const Dim = 4;
    var emb1 = embedding.TernaryVector(Dim).initZero();
    emb1.data = [_]i8{ 1, 1, 1, 1 };
    var emb2 = embedding.TernaryVector(Dim).initZero();
    emb2.data = [_]i8{ 1, 1, 1, 1 };

    const a = Insight(Dim).builder(1).setEmbedding(emb1).build();
    const b = Insight(Dim).builder(2).setEmbedding(emb2).build();
    try std.testing.expect(a.similarityThreshold(b, 0.99));
}

test "Insight similarityThreshold no embedding" {
    const Dim = 4;
    const a = Insight(Dim).builder(1).build();
    const b = Insight(Dim).builder(2).build();
    try std.testing.expect(!a.similarityThreshold(b, 0.5));
}

test "Insight builder addTag limit" {
    const Dim = 4;
    var b = Insight(Dim).builder(1);
    for (0..MaxTags + 4) |i| {
        b = b.addTag(&[_]u8{@intCast(i % 26 + 97)}); // single-char tags
    }
    const ins = b.build();
    try std.testing.expectEqual(@as(usize, MaxTags), ins.tags_count);
}
