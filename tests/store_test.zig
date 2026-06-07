const std = @import("std");
const store = @import("exocortex_memory").store;
const insight = @import("exocortex_memory").insight;
const embedding = @import("exocortex_memory").embedding;

test "store_test: insert and count" {
    const Dim = 8;
    var s = store.MemoryStore(Dim).init(std.testing.allocator);
    defer s.deinit();

    try std.testing.expectEqual(@as(usize, 0), s.count());
    try s.insert(insight.Insight(Dim).builder(1).setText("a").addTag("x").setConfidence(0.9).setTimestamp(100).build());
    try std.testing.expectEqual(@as(usize, 1), s.count());
    try s.insert(insight.Insight(Dim).builder(2).setText("b").addTag("y").setConfidence(0.8).setTimestamp(200).build());
    try std.testing.expectEqual(@as(usize, 2), s.count());
}

test "store_test: query returns sorted by similarity" {
    const Dim = 4;
    var s = store.MemoryStore(Dim).init(std.testing.allocator);
    defer s.deinit();

    var emb1 = embedding.TernaryVector(Dim).initZero();
    emb1.data = [_]i8{ 1, 1, 1, 1 };
    var emb2 = embedding.TernaryVector(Dim).initZero();
    emb2.data = [_]i8{ 1, 0, 0, 0 };
    var emb3 = embedding.TernaryVector(Dim).initZero();
    emb3.data = [_]i8{ 0, 0, 1, 0 };

    try s.insert(insight.Insight(Dim).builder(1).setText("perfect").addTag("a").setConfidence(0.9).setTimestamp(100).setEmbedding(emb1).build());
    try s.insert(insight.Insight(Dim).builder(2).setText("partial").addTag("b").setConfidence(0.8).setTimestamp(200).setEmbedding(emb2).build());
    try s.insert(insight.Insight(Dim).builder(3).setText("some").addTag("c").setConfidence(0.7).setTimestamp(300).setEmbedding(emb3).build());

    const results = try s.query(emb1, 10);
    defer std.testing.allocator.free(results);
    try std.testing.expectEqual(@as(usize, 3), results.len);
    try std.testing.expectEqual(@as(u64, 1), results[0].insight_id);
    try std.testing.expect(results[0].similarity >= results[1].similarity);
    try std.testing.expect(results[1].similarity >= results[2].similarity);
}

test "store_test: query top-K limits results" {
    const Dim = 4;
    var s = store.MemoryStore(Dim).init(std.testing.allocator);
    defer s.deinit();

    var emb = embedding.TernaryVector(Dim).initZero();
    emb.data = [_]i8{ 1, 0, 0, 0 };

    for (0..10) |i| {
        try s.insert(insight.Insight(Dim).builder(@intCast(i)).setText("item").addTag("t").setConfidence(0.5).setTimestamp(@intCast(i)).setEmbedding(emb).build());
    }

    const results = try s.query(emb, 3);
    defer std.testing.allocator.free(results);
    try std.testing.expectEqual(@as(usize, 3), results.len);
}

test "store_test: findByTag" {
    const Dim = 4;
    var s = store.MemoryStore(Dim).init(std.testing.allocator);
    defer s.deinit();

    try s.insert(insight.Insight(Dim).builder(1).setText("a").addTag("zig").setConfidence(0.5).setTimestamp(1).build());
    try s.insert(insight.Insight(Dim).builder(2).setText("b").addTag("rust").setConfidence(0.5).setTimestamp(2).build());
    try s.insert(insight.Insight(Dim).builder(3).setText("c").addTag("zig").setConfidence(0.5).setTimestamp(3).build());

    const zig = try s.findByTag("zig");
    defer std.testing.allocator.free(zig);
    try std.testing.expectEqual(@as(usize, 2), zig.len);

    const rust = try s.findByTag("rust");
    defer std.testing.allocator.free(rust);
    try std.testing.expectEqual(@as(usize, 1), rust.len);

    const none = try s.findByTag("python");
    defer std.testing.allocator.free(none);
    try std.testing.expectEqual(@as(usize, 0), none.len);
}

test "store_test: findByTimeRange" {
    const Dim = 4;
    var s = store.MemoryStore(Dim).init(std.testing.allocator);
    defer s.deinit();

    try s.insert(insight.Insight(Dim).builder(1).setText("a").addTag("x").setConfidence(0.5).setTimestamp(100).build());
    try s.insert(insight.Insight(Dim).builder(2).setText("b").addTag("x").setConfidence(0.5).setTimestamp(200).build());
    try s.insert(insight.Insight(Dim).builder(3).setText("c").addTag("x").setConfidence(0.5).setTimestamp(300).build());
    try s.insert(insight.Insight(Dim).builder(4).setText("d").addTag("x").setConfidence(0.5).setTimestamp(400).build());

    const r1 = try s.findByTimeRange(150, 350);
    defer std.testing.allocator.free(r1);
    try std.testing.expectEqual(@as(usize, 2), r1.len);

    const r2 = try s.findByTimeRange(100, 100);
    defer std.testing.allocator.free(r2);
    try std.testing.expectEqual(@as(usize, 1), r2.len);

    const r3 = try s.findByTimeRange(500, 600);
    defer std.testing.allocator.free(r3);
    try std.testing.expectEqual(@as(usize, 0), r3.len);
}

test "store_test: remove existing" {
    const Dim = 4;
    var s = store.MemoryStore(Dim).init(std.testing.allocator);
    defer s.deinit();

    try s.insert(insight.Insight(Dim).builder(1).setText("a").addTag("x").setConfidence(0.5).setTimestamp(1).build());
    try s.insert(insight.Insight(Dim).builder(2).setText("b").addTag("x").setConfidence(0.5).setTimestamp(2).build());

    try std.testing.expect(s.remove(1));
    try std.testing.expectEqual(@as(usize, 1), s.count());
    try std.testing.expect(s.getById(1) == null);
    try std.testing.expect(s.getById(2) != null);
}

test "store_test: remove non-existent" {
    const Dim = 4;
    var s = store.MemoryStore(Dim).init(std.testing.allocator);
    defer s.deinit();

    try s.insert(insight.Insight(Dim).builder(1).setText("a").addTag("x").setConfidence(0.5).setTimestamp(1).build());
    try std.testing.expect(!s.remove(999));
    try std.testing.expectEqual(@as(usize, 1), s.count());
}

test "store_test: getById" {
    const Dim = 4;
    var s = store.MemoryStore(Dim).init(std.testing.allocator);
    defer s.deinit();

    try s.insert(insight.Insight(Dim).builder(42).setText("answer").addTag("deep").setConfidence(1.0).setTimestamp(0).build());
    const found = s.getById(42);
    try std.testing.expect(found != null);
    try std.testing.expectEqualStrings("answer", found.?.text);
    try std.testing.expect(s.getById(99) == null);
}

test "store_test: query with no embeddings returns empty" {
    const Dim = 4;
    var s = store.MemoryStore(Dim).init(std.testing.allocator);
    defer s.deinit();

    try s.insert(insight.Insight(Dim).builder(1).setText("noemb").addTag("x").setConfidence(0.5).setTimestamp(1).build());

    const emb = embedding.TernaryVector(Dim).initZero();
    const results = try s.query(emb, 5);
    defer std.testing.allocator.free(results);
    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "store_test: multiple inserts and removals" {
    const Dim = 4;
    var s = store.MemoryStore(Dim).init(std.testing.allocator);
    defer s.deinit();

    for (0..20) |i| {
        try s.insert(insight.Insight(Dim).builder(@intCast(i)).setText("item").addTag("t").setConfidence(0.5).setTimestamp(@intCast(i)).build());
    }
    try std.testing.expectEqual(@as(usize, 20), s.count());

    for (0..10) |i| {
        _ = s.remove(@intCast(i * 2));
    }
    try std.testing.expectEqual(@as(usize, 10), s.count());
}
