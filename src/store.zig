//! MemoryStore — the central in-memory store for insights.
//!
//! Maintains an ArrayList of Insights and provides insert, query (top-K
//! similarity search), tag-based lookup, time-range filtering, and removal.
//! Uses a brute-force similarity index over ternary embeddings.

const std = @import("std");
const embedding = @import("embedding.zig");
const insight_mod = @import("insight.zig");
const index_mod = @import("index.zig");

/// A match result from a similarity query.
pub fn MatchResult(comptime dim: usize) type {
    return struct {
        insight_id: u64,
        similarity: f64,
        insight: insight_mod.Insight(dim),
    };
}

/// In-memory store parameterized by embedding dimension.
pub fn MemoryStore(comptime dim: usize) type {
    return struct {
        allocator: std.mem.Allocator,
        insights: std.ArrayList(insight_mod.Insight(dim)),
        next_id: u64,

        const Self = @This();
        const InsightType = insight_mod.Insight(dim);
        const Match = MatchResult(dim);

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .insights = std.ArrayList(InsightType).init(allocator),
                .next_id = 1,
            };
        }

        pub fn deinit(self: *Self) void {
            self.insights.deinit();
        }

        /// Insert an insight into the store.
        pub fn insert(self: *Self, ins: InsightType) !void {
            try self.insights.append(ins);
        }

        /// Get the current count of stored insights.
        pub fn count(self: Self) usize {
            return self.insights.items.len;
        }

        /// Query by embedding vector, returning top-K results sorted by similarity (descending).
        pub fn query(self: Self, query_emb: embedding.TernaryVector(dim), top_k: usize) ![]Match {
            var results = std.ArrayList(Match).init(self.allocator);
            defer results.deinit();

            for (self.insights.items) |ins| {
                const emb = ins.embedding orelse continue;
                const sim = query_emb.cosineSimilarity(emb);
                try results.append(.{
                    .insight_id = ins.id,
                    .similarity = sim,
                    .insight = ins,
                });
            }

            // Sort descending by similarity
            const SortCtx = struct {
                fn lessThan(_: void, a: Match, b: Match) bool {
                    return a.similarity > b.similarity;
                }
            };
            std.sort.insertion(Match, results.items, {}, SortCtx.lessThan);

            const k = @min(top_k, results.items.len);
            return self.allocator.dupe(Match, results.items[0..k]);
        }

        /// Find insights that have a specific tag.
        pub fn findByTag(self: Self, tag: []const u8) ![]InsightType {
            var results = std.ArrayList(InsightType).init(self.allocator);
            defer results.deinit();
            for (self.insights.items) |ins| {
                if (ins.hasTag(tag)) {
                    try results.append(ins);
                }
            }
            return self.allocator.dupe(InsightType, results.items);
        }

        /// Find insights within a time range [start, end].
        pub fn findByTimeRange(self: Self, start: i64, end: i64) ![]InsightType {
            var results = std.ArrayList(InsightType).init(self.allocator);
            defer results.deinit();
            for (self.insights.items) |ins| {
                if (ins.timestamp >= start and ins.timestamp <= end) {
                    try results.append(ins);
                }
            }
            return self.allocator.dupe(InsightType, results.items);
        }

        /// Remove an insight by ID. Returns true if found and removed.
        pub fn remove(self: *Self, id: u64) bool {
            for (self.insights.items, 0..) |ins, i| {
                if (ins.id == id) {
                    _ = self.insights.orderedRemove(i);
                    return true;
                }
            }
            return false;
        }

        /// Get insight by ID.
        pub fn getById(self: Self, id: u64) ?InsightType {
            for (self.insights.items) |ins| {
                if (ins.id == id) return ins;
            }
            return null;
        }
    };
}

test "MemoryStore insert and count" {
    const Dim = 4;
    var store = MemoryStore(Dim).init(std.testing.allocator);
    defer store.deinit();
    const ins = insight_mod.Insight(Dim).builder(1).setText("test").build();
    try store.insert(ins);
    try std.testing.expectEqual(@as(usize, 1), store.count());
}

test "MemoryStore query top-K" {
    const Dim = 4;
    var store = MemoryStore(Dim).init(std.testing.allocator);
    defer store.deinit();

    var emb1 = embedding.TernaryVector(Dim).initZero();
    emb1.data = [_]i8{ 1, 1, 1, 1 };
    var emb2 = embedding.TernaryVector(Dim).initZero();
    emb2.data = [_]i8{ 1, 0, 0, 0 };
    var emb3 = embedding.TernaryVector(Dim).initZero();
    emb3.data = [_]i8{ -1, -1, -1, -1 };

    try store.insert(insight_mod.Insight(Dim).builder(1).setEmbedding(emb1).build());
    try store.insert(insight_mod.Insight(Dim).builder(2).setEmbedding(emb2).build());
    try store.insert(insight_mod.Insight(Dim).builder(3).setEmbedding(emb3).build());

    const results = try store.query(emb1, 2);
    defer std.testing.allocator.free(results);
    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqual(@as(u64, 1), results[0].insight_id); // identical → highest similarity
}

test "MemoryStore findByTag" {
    const Dim = 4;
    var store = MemoryStore(Dim).init(std.testing.allocator);
    defer store.deinit();

    try store.insert(insight_mod.Insight(Dim).builder(1).addTag("zig").build());
    try store.insert(insight_mod.Insight(Dim).builder(2).addTag("rust").build());
    try store.insert(insight_mod.Insight(Dim).builder(3).addTag("zig").addTag("lang").build());

    const found = try store.findByTag("zig");
    defer std.testing.allocator.free(found);
    try std.testing.expectEqual(@as(usize, 2), found.len);
}

test "MemoryStore findByTimeRange" {
    const Dim = 4;
    var store = MemoryStore(Dim).init(std.testing.allocator);
    defer store.deinit();

    try store.insert(insight_mod.Insight(Dim).builder(1).setTimestamp(100).build());
    try store.insert(insight_mod.Insight(Dim).builder(2).setTimestamp(200).build());
    try store.insert(insight_mod.Insight(Dim).builder(3).setTimestamp(300).build());

    const found = try store.findByTimeRange(150, 250);
    defer std.testing.allocator.free(found);
    try std.testing.expectEqual(@as(usize, 1), found.len);
    try std.testing.expectEqual(@as(u64, 2), found[0].id);
}

test "MemoryStore remove" {
    const Dim = 4;
    var store = MemoryStore(Dim).init(std.testing.allocator);
    defer store.deinit();

    try store.insert(insight_mod.Insight(Dim).builder(1).build());
    try store.insert(insight_mod.Insight(Dim).builder(2).build());

    try std.testing.expect(store.remove(1));
    try std.testing.expectEqual(@as(usize, 1), store.count());
    try std.testing.expect(!store.remove(999));
}

test "MemoryStore getById" {
    const Dim = 4;
    var store = MemoryStore(Dim).init(std.testing.allocator);
    defer store.deinit();

    try store.insert(insight_mod.Insight(Dim).builder(42).setText("the answer").build());
    const found = store.getById(42);
    try std.testing.expect(found != null);
    try std.testing.expectEqualStrings("the answer", found.?.text);
    try std.testing.expect(store.getById(99) == null);
}
