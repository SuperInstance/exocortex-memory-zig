//! Similarity index — brute-force cosine similarity search over ternary embeddings.
//!
//! Provides a precomputed index that stores references to insights and their
//! embeddings. Query returns top-K matches sorted by similarity.
//!
//! Current implementation is brute-force O(n*d) per query, which is efficient
//! enough for thousands of insights with ternary vectors (integer arithmetic).

const std = @import("std");
const embedding = @import("embedding.zig");
const insight_mod = @import("insight.zig");

/// A match result from the similarity index.
pub const MatchResult = struct {
    insight_id: u64,
    similarity: f64,
};

/// Similarity index parameterized by embedding dimension.
pub fn SimilarityIndex(comptime dim: usize) type {
    return struct {
        allocator: std.mem.Allocator,
        entries: std.ArrayList(Entry),

        const Self = @This();

        const Entry = struct {
            insight_id: u64,
            embedding: embedding.TernaryVector(dim),
        };

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .entries = std.ArrayList(Entry).init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            self.entries.deinit();
        }

        /// Build index from a slice of insights (those with embeddings).
        pub fn buildIndex(self: *Self, insights: []const insight_mod.Insight(dim)) !void {
            self.entries.clearRetainingCapacity();
            for (insights) |ins| {
                if (ins.embedding) |emb| {
                    try self.entries.append(.{
                        .insight_id = ins.id,
                        .embedding = emb,
                    });
                }
            }
        }

        /// Query for top-K most similar entries to the given embedding.
        pub fn query(self: Self, query_emb: embedding.TernaryVector(dim), top_k: usize) ![]MatchResult {
            var results = std.ArrayList(MatchResult).init(self.allocator);
            defer results.deinit();

            for (self.entries.items) |entry| {
                const sim = query_emb.cosineSimilarity(entry.embedding);
                try results.append(.{
                    .insight_id = entry.insight_id,
                    .similarity = sim,
                });
            }

            // Sort descending by similarity
            const SortCtx = struct {
                fn lessThan(_: void, a: MatchResult, b: MatchResult) bool {
                    return a.similarity > b.similarity;
                }
            };
            std.sort.insertion(MatchResult, results.items, {}, SortCtx.lessThan);

            const k = @min(top_k, results.items.len);
            return self.allocator.dupe(MatchResult, results.items[0..k]);
        }

        /// Get the number of indexed entries.
        pub fn size(self: Self) usize {
            return self.entries.items.len;
        }
    };
}

test "SimilarityIndex build and size" {
    const Dim = 4;
    var idx = SimilarityIndex(Dim).init(std.testing.allocator);
    defer idx.deinit();

    var emb = embedding.TernaryVector(Dim).initZero();
    emb.data = [_]i8{ 1, 1, 1, 1 };

    const insights = [_]insight_mod.Insight(Dim){
        insight_mod.Insight(Dim).builder(1).setEmbedding(emb).build(),
        insight_mod.Insight(Dim).builder(2).build(), // no embedding
    };

    try idx.buildIndex(&insights);
    try std.testing.expectEqual(@as(usize, 1), idx.size());
}

test "SimilarityIndex query top-K" {
    const Dim = 4;
    var idx = SimilarityIndex(Dim).init(std.testing.allocator);
    defer idx.deinit();

    var emb1 = embedding.TernaryVector(Dim).initZero();
    emb1.data = [_]i8{ 1, 1, 1, 1 };
    var emb2 = embedding.TernaryVector(Dim).initZero();
    emb2.data = [_]i8{ 1, 0, 0, 0 };
    var emb3 = embedding.TernaryVector(Dim).initZero();
    emb3.data = [_]i8{ -1, -1, -1, -1 };

    const insights = [_]insight_mod.Insight(Dim){
        insight_mod.Insight(Dim).builder(1).setEmbedding(emb1).build(),
        insight_mod.Insight(Dim).builder(2).setEmbedding(emb2).build(),
        insight_mod.Insight(Dim).builder(3).setEmbedding(emb3).build(),
    };

    try idx.buildIndex(&insights);
    const results = try idx.query(emb1, 2);
    defer std.testing.allocator.free(results);

    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqual(@as(u64, 1), results[0].insight_id);
    try std.testing.expect(results[0].similarity > results[1].similarity);
}

test "SimilarityIndex query single match" {
    const Dim = 4;
    var idx = SimilarityIndex(Dim).init(std.testing.allocator);
    defer idx.deinit();

    var emb = embedding.TernaryVector(Dim).initZero();
    emb.data = [_]i8{ 1, -1, 1, 0 };

    const insights = [_]insight_mod.Insight(Dim){
        insight_mod.Insight(Dim).builder(1).setEmbedding(emb).build(),
    };

    try idx.buildIndex(&insights);
    const results = try idx.query(emb, 5);
    defer std.testing.allocator.free(results);
    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), results[0].similarity, 1e-10);
}

test "SimilarityIndex rebuild clears old data" {
    const Dim = 4;
    var idx = SimilarityIndex(Dim).init(std.testing.allocator);
    defer idx.deinit();

    var emb = embedding.TernaryVector(Dim).initZero();
    emb.data = [_]i8{ 1, 0, 0, 0 };

    const insights1 = [_]insight_mod.Insight(Dim){
        insight_mod.Insight(Dim).builder(1).setEmbedding(emb).build(),
    };
    try idx.buildIndex(&insights1);
    try std.testing.expectEqual(@as(usize, 1), idx.size());

    const insights2 = [_]insight_mod.Insight(Dim){
        insight_mod.Insight(Dim).builder(10).setEmbedding(emb).build(),
        insight_mod.Insight(Dim).builder(20).setEmbedding(emb).build(),
    };
    try idx.buildIndex(&insights2);
    try std.testing.expectEqual(@as(usize, 2), idx.size());
}

test "SimilarityIndex query empty index" {
    const Dim = 4;
    var idx = SimilarityIndex(Dim).init(std.testing.allocator);
    defer idx.deinit();

    const insights = [_]insight_mod.Insight(Dim){};
    try idx.buildIndex(&insights);

    const emb = embedding.TernaryVector(Dim).initZero();
    const results = try idx.query(emb, 5);
    defer std.testing.allocator.free(results);
    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "SimilarityIndex skips insights without embeddings" {
    const Dim = 4;
    var idx = SimilarityIndex(Dim).init(std.testing.allocator);
    defer idx.deinit();

    const insights = [_]insight_mod.Insight(Dim){
        insight_mod.Insight(Dim).builder(1).build(),
        insight_mod.Insight(Dim).builder(2).build(),
    };
    try idx.buildIndex(&insights);
    try std.testing.expectEqual(@as(usize, 0), idx.size());
}
