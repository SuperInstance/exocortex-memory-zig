//! JSON serialization for Insight and MemoryStore.
//!
//! Uses std.json for encoding/decoding. TernaryVector is serialized as an
//! array of integers. Tags are serialized as an array of strings (null-trimmed).

const std = @import("std");
const embedding = @import("embedding.zig");
const insight_mod = @import("insight.zig");
const store_mod = @import("store.zig");

/// Serialize an Insight to a JSON string.
pub fn insightToJson(allocator: std.mem.Allocator, ins: anytype) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();
    var writer = buf.writer();

    try writer.writeAll("{\n");
    try writer.print("  \"id\": {},\n", .{ins.id});
    try writer.print("  \"text\": \"{}\",\n", .{std.zig.fmtEscapes(ins.text)});

    // Tags
    try writer.writeAll("  \"tags\": [");
    for (ins.tags[0..ins.tags_count], 0..) |tag_buf, i| {
        var end: usize = 0;
        while (end < insight_mod.TagLen and tag_buf[end] != 0) end += 1;
        if (i > 0) try writer.writeAll(", ");
        try writer.print("\"{}\"", .{std.zig.fmtEscapes(tag_buf[0..end])});
    }
    try writer.writeAll("],\n");

    try writer.print("  \"confidence\": {d},\n", .{ins.confidence});
    try writer.print("  \"timestamp\": {},\n", .{ins.timestamp});

    // Embedding
    if (ins.embedding) |emb| {
        try writer.writeAll("  \"embedding\": [");
        for (emb.data, 0..) |val, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("{}", .{val});
        }
        try writer.writeAll("]\n");
    } else {
        try writer.writeAll("  \"embedding\": null\n");
    }

    try writer.writeAll("}");
    return buf.toOwnedSlice();
}

/// Serialize all insights in a store to a JSON string.
pub fn serializeStore(allocator: std.mem.Allocator, store: anytype) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();
    var writer = buf.writer();

    try writer.writeByte('[');
    for (store.insights.items, 0..) |ins, i| {
        if (i > 0) try writer.writeAll(",");
        const json_str = try insightToJson(allocator, ins);
        defer allocator.free(json_str);
        try writer.writeAll(json_str);
    }
    try writer.writeByte(']');
    return buf.toOwnedSlice();
}

/// Parse a JSON string into a parsed insight struct.
pub fn parseInsightJson(allocator: std.mem.Allocator, data: []const u8) !ParsedInsight {
    var diag = std.json.Diagnostics{};
    var stream = std.json.TokenStream.init(data);
    stream.diag = &diag;
    const parsed = try std.json.parse(struct {
        id: u64,
        text: []const u8,
        confidence: f64,
        timestamp: i64,
    }, &stream, .{
        .allocator = allocator,
        .ignore_unknown_fields = true,
    });

    return .{
        .id = parsed.id,
        .text = parsed.text,
        .confidence = parsed.confidence,
        .timestamp = parsed.timestamp,
    };
}

pub const ParsedInsight = struct {
    id: u64,
    text: []const u8,
    confidence: f64,
    timestamp: i64,
};

test "serialize insightToJson" {
    const Dim = 4;
    var emb = embedding.TernaryVector(Dim).initZero();
    emb.data = [_]i8{ 1, -1, 0, 1 };

    const ins = insight_mod.Insight(Dim).builder(1)
        .setText("hello zig")
        .addTag("test")
        .setConfidence(0.85)
        .setTimestamp(1234567890)
        .setEmbedding(emb)
        .build();

    const json = try insightToJson(std.testing.allocator, ins);
    defer std.testing.allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "hello zig") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "test") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "0.85") != null);
}

test "serialize insightToJson without embedding" {
    const Dim = 4;
    const ins = insight_mod.Insight(Dim).builder(2)
        .setText("no embedding")
        .setConfidence(0.5)
        .setTimestamp(100)
        .build();

    const json = try insightToJson(std.testing.allocator, ins);
    defer std.testing.allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "null") != null);
}

test "serialize store" {
    const Dim = 4;
    var s = store_mod.MemoryStore(Dim).init(std.testing.allocator);
    defer s.deinit();

    try s.insert(insight_mod.Insight(Dim).builder(1).setText("first").addTag("a").setConfidence(0.9).setTimestamp(100).build());
    try s.insert(insight_mod.Insight(Dim).builder(2).setText("second").addTag("b").setConfidence(0.8).setTimestamp(200).build());

    const json = try serializeStore(std.testing.allocator, s);
    defer std.testing.allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "first") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "second") != null);
    try std.testing.expect(json[0] == '[');
}

test "serialize empty store" {
    const Dim = 4;
    var s = store_mod.MemoryStore(Dim).init(std.testing.allocator);
    defer s.deinit();

    const json = try serializeStore(std.testing.allocator, s);
    defer std.testing.allocator.free(json);
    try std.testing.expectEqualStrings("[]", json);
}

test "serialize insight with multiple tags" {
    const Dim = 4;
    const ins = insight_mod.Insight(Dim).builder(1)
        .setText("multi")
        .addTag("alpha")
        .addTag("beta")
        .addTag("gamma")
        .setConfidence(1.0)
        .setTimestamp(0)
        .build();

    const json = try insightToJson(std.testing.allocator, ins);
    defer std.testing.allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "alpha") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "beta") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "gamma") != null);
}
