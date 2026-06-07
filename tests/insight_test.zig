const std = @import("std");
const insight = @import("exocortex_memory").insight;
const embedding = @import("exocortex_memory").embedding;

test "insight_test: builder creates valid insight" {
    const Dim = 8;
    const ins = insight.Insight(Dim).builder(42)
        .setText("Zig comptime is powerful")
        .addTag("zig")
        .addTag("comptime")
        .setConfidence(0.95)
        .setTimestamp(1700000000)
        .build();

    try std.testing.expectEqual(@as(u64, 42), ins.id);
    try std.testing.expectEqualStrings("Zig comptime is powerful", ins.text);
    try std.testing.expectEqual(@as(f64, 0.95), ins.confidence);
    try std.testing.expectEqual(@as(i64, 1700000000), ins.timestamp);
    try std.testing.expectEqual(@as(usize, 2), ins.tags_count);
    try std.testing.expect(ins.embedding == null);
}

test "insight_test: builder with embedding" {
    const Dim = 4;
    var emb = embedding.TernaryVector(Dim).initZero();
    emb.data = [_]i8{ 1, -1, 0, 1 };

    const ins = insight.Insight(Dim).builder(1)
        .setText("embedded thought")
        .addTag("test")
        .setConfidence(0.8)
        .setEmbedding(emb)
        .build();

    try std.testing.expect(ins.embedding != null);
    try std.testing.expectEqual(@as(i8, 1), ins.embedding.?.data[0]);
}

test "insight_test: hasTag positive" {
    const Dim = 4;
    const ins = insight.Insight(Dim).builder(1)
        .addTag("alpha")
        .addTag("beta")
        .build();
    try std.testing.expect(ins.hasTag("alpha"));
    try std.testing.expect(ins.hasTag("beta"));
}

test "insight_test: hasTag negative" {
    const Dim = 4;
    const ins = insight.Insight(Dim).builder(1).addTag("zig").build();
    try std.testing.expect(!ins.hasTag("rust"));
}

test "insight_test: tagsOverlap partial" {
    const Dim = 4;
    const a = insight.Insight(Dim).builder(1).addTag("x").addTag("y").build();
    const b = insight.Insight(Dim).builder(2).addTag("y").addTag("z").build();
    try std.testing.expectEqual(@as(usize, 1), a.tagsOverlap(b));
}

test "insight_test: tagsOverlap full" {
    const Dim = 4;
    const a = insight.Insight(Dim).builder(1).addTag("a").addTag("b").build();
    const b = insight.Insight(Dim).builder(2).addTag("a").addTag("b").build();
    try std.testing.expectEqual(@as(usize, 2), a.tagsOverlap(b));
}

test "insight_test: tagsOverlap none" {
    const Dim = 4;
    const a = insight.Insight(Dim).builder(1).addTag("a").build();
    const b = insight.Insight(Dim).builder(2).addTag("b").build();
    try std.testing.expectEqual(@as(usize, 0), a.tagsOverlap(b));
}

test "insight_test: similarityThreshold true" {
    const Dim = 4;
    var emb = embedding.TernaryVector(Dim).initZero();
    emb.data = [_]i8{ 1, 1, 1, 1 };

    const a = insight.Insight(Dim).builder(1).setEmbedding(emb).build();
    const b = insight.Insight(Dim).builder(2).setEmbedding(emb).build();
    try std.testing.expect(a.similarityThreshold(b, 0.99));
}

test "insight_test: similarityThreshold false" {
    const Dim = 4;
    var emb1 = embedding.TernaryVector(Dim).initZero();
    emb1.data = [_]i8{ 1, 1, 1, 1 };
    var emb2 = embedding.TernaryVector(Dim).initZero();
    emb2.data = [_]i8{ -1, -1, -1, -1 };

    const a = insight.Insight(Dim).builder(1).setEmbedding(emb1).build();
    const b = insight.Insight(Dim).builder(2).setEmbedding(emb2).build();
    try std.testing.expect(!a.similarityThreshold(b, 0.5));
}

test "insight_test: makeTag helper" {
    const tag = insight.makeTag("hello");
    try std.testing.expectEqual(@as(u8, 'h'), tag[0]);
    try std.testing.expectEqual(@as(u8, 'o'), tag[4]);
    try std.testing.expectEqual(@as(u8, 0), tag[5]);
}

test "insight_test: builder tag limit" {
    const Dim = 4;
    var b = insight.Insight(Dim).builder(1);
    for (0..20) |i| {
        const name = &[_]u8{@intCast(65 + (i % 26))};
        b = b.addTag(name);
    }
    const ins = b.build();
    try std.testing.expectEqual(@as(usize, insight.MaxTags), ins.tags_count);
}

test "insight_test: empty insight defaults" {
    const Dim = 4;
    const ins = insight.Insight(Dim).builder(0).build();
    try std.testing.expectEqualStrings("", ins.text);
    try std.testing.expectEqual(@as(f64, 0.0), ins.confidence);
    try std.testing.expectEqual(@as(usize, 0), ins.tags_count);
}
