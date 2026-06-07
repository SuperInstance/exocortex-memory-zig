const std = @import("std");
const index = @import("exocortex_memory").index;
const insight = @import("exocortex_memory").insight;
const embedding = @import("exocortex_memory").embedding;

test "index_test: buildIndex and size" {
    const Dim = 4;
    var idx = index.SimilarityIndex(Dim).init(std.testing.allocator);
    defer idx.deinit();

    var emb = embedding.TernaryVector(Dim).initZero();
    emb.data = [_]i8{ 1, 1, 1, 1 };

    const insights = [_]insight.Insight(Dim){
        insight.Insight(Dim).builder(1).setText("a").addTag("x").setConfidence(0.9).setTimestamp(1).setEmbedding(emb).build(),
        insight.Insight(Dim).builder(2).setText("b").addTag("y").setConfidence(0.8).setTimestamp(2).build(), // no embedding
    };

    try idx.buildIndex(&insights);
    try std.testing.expectEqual(@as(usize, 1), idx.size());
}

test "index_test: query returns sorted results" {
    const Dim = 4;
    var idx = index.SimilarityIndex(Dim).init(std.testing.allocator);
    defer idx.deinit();

    var emb1 = embedding.TernaryVector(Dim).initZero();
    emb1.data = [_]i8{ 1, 1, 1, 1 };
    var emb2 = embedding.TernaryVector(Dim).initZero();
    emb2.data = [_]i8{ 1, 0, 0, 0 };
    var emb3 = embedding.TernaryVector(Dim).initZero();
    emb3.data = [_]i8{ -1, -1, -1, -1 };

    const insights = [_]insight.Insight(Dim){
        insight.Insight(Dim).builder(1).setText("high").addTag("a").setConfidence(0.9).setTimestamp(1).setEmbedding(emb1).build(),
        insight.Insight(Dim).builder(2).setText("mid").addTag("b").setConfidence(0.8).setTimestamp(2).setEmbedding(emb2).build(),
        insight.Insight(Dim).builder(3).setText("low").addTag("c").setConfidence(0.7).setTimestamp(3).setEmbedding(emb3).build(),
    };

    try idx.buildIndex(&insights);

    const results = try idx.query(emb1, 10);
    defer std.testing.allocator.free(results);
    try std.testing.expectEqual(@as(usize, 3), results.len);
    try std.testing.expectEqual(@as(u64, 1), results[0].insight_id);
    try std.testing.expect(results[0].similarity > results[1].similarity);
    try std.testing.expect(results[1].similarity > results[2].similarity);
}

test "index_test: query top-K limits results" {
    const Dim = 4;
    var idx = index.SimilarityIndex(Dim).init(std.testing.allocator);
    defer idx.deinit();

    var emb = embedding.TernaryVector(Dim).initZero();
    emb.data = [_]i8{ 1, 0, 0, 0 };

    var insights: [5]insight.Insight(Dim) = undefined;
    for (&insights, 0..) |*ins, i| {
        ins.* = insight.Insight(Dim).builder(@intCast(i)).setText("item").addTag("t").setConfidence(0.5).setTimestamp(@intCast(i)).setEmbedding(emb).build();
    }

    try idx.buildIndex(&insights);
    const results = try idx.query(emb, 2);
    defer std.testing.allocator.free(results);
    try std.testing.expectEqual(@as(usize, 2), results.len);
}

test "index_test: query identical embedding returns 1.0" {
    const Dim = 4;
    var idx = index.SimilarityIndex(Dim).init(std.testing.allocator);
    defer idx.deinit();

    var emb = embedding.TernaryVector(Dim).initZero();
    emb.data = [_]i8{ 1, -1, 0, 1 };

    const insights = [_]insight.Insight(Dim){
        insight.Insight(Dim).builder(99).setText("exact").addTag("t").setConfidence(1.0).setTimestamp(0).setEmbedding(emb).build(),
    };

    try idx.buildIndex(&insights);
    const results = try idx.query(emb, 1);
    defer std.testing.allocator.free(results);
    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), results[0].similarity, 1e-10);
}

test "index_test: rebuildIndex clears old data" {
    const Dim = 4;
    var idx = index.SimilarityIndex(Dim).init(std.testing.allocator);
    defer idx.deinit();

    var emb = embedding.TernaryVector(Dim).initZero();
    emb.data = [_]i8{ 1, 0, 0, 0 };

    const batch1 = [_]insight.Insight(Dim){
        insight.Insight(Dim).builder(1).setText("a").addTag("t").setConfidence(0.5).setTimestamp(1).setEmbedding(emb).build(),
    };
    try idx.buildIndex(&batch1);
    try std.testing.expectEqual(@as(usize, 1), idx.size());

    const batch2 = [_]insight.Insight(Dim){
        insight.Insight(Dim).builder(10).setText("x").addTag("t").setConfidence(0.5).setTimestamp(10).setEmbedding(emb).build(),
        insight.Insight(Dim).builder(20).setText("y").addTag("t").setConfidence(0.5).setTimestamp(20).setEmbedding(emb).build(),
        insight.Insight(Dim).builder(30).setText("z").addTag("t").setConfidence(0.5).setTimestamp(30).setEmbedding(emb).build(),
    };
    try idx.buildIndex(&batch2);
    try std.testing.expectEqual(@as(usize, 3), idx.size());
}

test "index_test: empty index returns empty results" {
    const Dim = 4;
    var idx = index.SimilarityIndex(Dim).init(std.testing.allocator);
    defer idx.deinit();

    const insights = [_]insight.Insight(Dim){};
    try idx.buildIndex(&insights);

    const emb = embedding.TernaryVector(Dim).initZero();
    const results = try idx.query(emb, 5);
    defer std.testing.allocator.free(results);
    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "index_test: insights without embeddings are skipped" {
    const Dim = 4;
    var idx = index.SimilarityIndex(Dim).init(std.testing.allocator);
    defer idx.deinit();

    const insights = [_]insight.Insight(Dim){
        insight.Insight(Dim).builder(1).setText("noemb").addTag("t").setConfidence(0.5).setTimestamp(1).build(),
        insight.Insight(Dim).builder(2).setText("noemb2").addTag("t").setConfidence(0.5).setTimestamp(2).build(),
    };

    try idx.buildIndex(&insights);
    try std.testing.expectEqual(@as(usize, 0), idx.size());
}

test "index_test: large dimension query" {
    const Dim = 256;
    var idx = index.SimilarityIndex(Dim).init(std.testing.allocator);
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(77);
    const emb1 = embedding.TernaryVector(Dim).initRandom(rng.random());
    const emb2 = embedding.TernaryVector(Dim).initRandom(rng.random());

    const insights = [_]insight.Insight(Dim){
        insight.Insight(Dim).builder(1).setText("close").addTag("t").setConfidence(0.5).setTimestamp(1).setEmbedding(emb1).build(),
        insight.Insight(Dim).builder(2).setText("far").addTag("t").setConfidence(0.5).setTimestamp(2).setEmbedding(emb2).build(),
    };

    try idx.buildIndex(&insights);
    const results = try idx.query(emb1, 2);
    defer std.testing.allocator.free(results);
    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqual(@as(u64, 1), results[0].insight_id);
    try std.testing.expect(results[0].similarity > results[1].similarity);
}
