const std = @import("std");
const schema = @import("exocortex_memory").schema;
const insight = @import("exocortex_memory").insight;
const embedding = @import("exocortex_memory").embedding;

test "schema_test: validateInsightSchema accepts Insight" {
    try std.testing.expect(schema.validateInsightSchema(insight.Insight(64), 64));
    try std.testing.expect(schema.validateInsightSchema(insight.Insight(128), 128));
    try std.testing.expect(schema.validateInsightSchema(insight.Insight(256), 256));
}

test "schema_test: getFieldNames returns correct count" {
    const names = schema.getFieldNames(insight.Insight(4));
    try std.testing.expect(names.len == 7);
}

test "schema_test: getFieldNames contains expected fields" {
    const names = schema.getFieldNames(insight.Insight(4));
    var found_id = false;
    var found_text = false;
    var found_confidence = false;
    for (names) |name| {
        if (std.mem.eql(u8, name, "id")) found_id = true;
        if (std.mem.eql(u8, name, "text")) found_text = true;
        if (std.mem.eql(u8, name, "confidence")) found_confidence = true;
    }
    try std.testing.expect(found_id);
    try std.testing.expect(found_text);
    try std.testing.expect(found_confidence);
}

test "schema_test: countFields" {
    try std.testing.expectEqual(@as(comptime_int, 7), schema.countFields(insight.Insight(4)));
}

test "schema_test: hasField positive cases" {
    try std.testing.expect(schema.hasField(insight.Insight(4), "id"));
    try std.testing.expect(schema.hasField(insight.Insight(4), "text"));
    try std.testing.expect(schema.hasField(insight.Insight(4), "tags"));
    try std.testing.expect(schema.hasField(insight.Insight(4), "tags_count"));
    try std.testing.expect(schema.hasField(insight.Insight(4), "confidence"));
    try std.testing.expect(schema.hasField(insight.Insight(4), "timestamp"));
    try std.testing.expect(schema.hasField(insight.Insight(4), "embedding"));
}

test "schema_test: hasField negative cases" {
    try std.testing.expect(!schema.hasField(insight.Insight(4), "name"));
    try std.testing.expect(!schema.hasField(insight.Insight(4), "description"));
    try std.testing.expect(!schema.hasField(insight.Insight(4), "data"));
}

test "schema_test: getFieldType correct types" {
    try std.testing.expect(schema.getFieldType(insight.Insight(4), "id") == u64);
    try std.testing.expect(schema.getFieldType(insight.Insight(4), "confidence") == f64);
    try std.testing.expect(schema.getFieldType(insight.Insight(4), "timestamp") == i64);
    try std.testing.expect(schema.getFieldType(insight.Insight(4), "tags_count") == usize);
}

test "schema_test: validateConfidence accepts boundary values" {
    comptime schema.validateConfidence(0.0);
    comptime schema.validateConfidence(1.0);
    comptime schema.validateConfidence(0.5);
}

test "schema_test: validateTagsNonEmpty accepts positive" {
    comptime schema.validateTagsNonEmpty(1);
    comptime schema.validateTagsNonEmpty(10);
    comptime schema.validateTagsNonEmpty(16);
}

test "schema_test: validateEmbeddingDim accepts matching dimension" {
    comptime schema.validateEmbeddingDim(embedding.TernaryVector(32), 32);
    comptime schema.validateEmbeddingDim(embedding.TernaryVector(128), 128);
}

test "schema_test: validateInsight accepts valid comptime insight" {
    const Dim = 8;
    const ins = comptime blk: {
        var emb = embedding.TernaryVector(Dim).initZero();
        emb.data = [_]i8{ 1, -1, 0, 1, 0, -1, 1, 0 };
        break :blk insight.Insight(Dim).builder(1)
            .setText("comptime validated")
            .addTag("test")
            .setConfidence(0.9)
            .setTimestamp(1000)
            .setEmbedding(emb)
            .build();
    };
    comptime schema.validateInsight(ins, Dim);
    try std.testing.expectEqual(@as(u64, 1), ins.id);
}

test "schema_test: hasField works on arbitrary struct" {
    const MyStruct = struct { x: i32, y: f64, label: []const u8, active: bool };
    try std.testing.expect(schema.hasField(MyStruct, "x"));
    try std.testing.expect(schema.hasField(MyStruct, "label"));
    try std.testing.expect(schema.hasField(MyStruct, "active"));
    try std.testing.expect(!schema.hasField(MyStruct, "z"));
}

test "schema_test: countFields on custom struct" {
    const Point = struct { x: f32, y: f32, z: f32 };
    try std.testing.expectEqual(@as(comptime_int, 3), schema.countFields(Point));
}

test "schema_test: getFieldType on custom struct" {
    const Entry = struct { key: []const u8, value: u64 };
    try std.testing.expect(schema.getFieldType(Entry, "key") == []const u8);
    try std.testing.expect(schema.getFieldType(Entry, "value") == u64);
}
