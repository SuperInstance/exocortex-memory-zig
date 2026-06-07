const std = @import("std");
const embedding = @import("embedding.zig");
const insight = @import("insight.zig");
const store = @import("store.zig");

pub fn main() !void {
    const Dim = 128;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== exocortex-memory-zig demo ===\n\n", .{});

    // Generate a random embedding
    var rng = std.Random.DefaultPrng.init(42);
    var emb = embedding.TernaryVector(Dim).initRandom(rng.random());
    std.debug.print("Random embedding (dim={}): {any}...\n", .{ Dim, emb.data[0..8] });

    // Build an insight
    const ins = insight.Insight(Dim){
        .id = 1,
        .text = "Zig comptime eliminates the need for macros",
        .confidence = 0.95,
        .timestamp = std.time.timestamp(),
        .embedding = emb,
        .tags_count = 2,
        .tags = [_][40]u8{blankTag("zig"), blankTag("comptime")},
    };
    std.debug.print("Insight: id={} text=\"{s}\" confidence={d:.2}\n", .{ ins.id, ins.text, ins.confidence });

    // Store
    var ms = store.MemoryStore(Dim).init(allocator);
    defer ms.deinit();
    try ms.insert(ins);
    std.debug.print("Store count: {}\n", .{ms.count()});

    // Query
    const results = try ms.query(emb, 5);
    defer allocator.free(results);
    std.debug.print("Query returned {} results\n", .{results.len});
    for (results, 0..) |r, i| {
        std.debug.print("  [{d}] id={} similarity={d:.4}\n", .{ i, r.insight_id, r.similarity });
    }

    std.debug.print("\nDemo complete.\n", .{});
}

fn blankTag(comptime s: []const u8) [40]u8 {
    var buf: [40]u8 = [_]u8{0} ** 40;
    @memcpy(buf[0..s.len], s);
    return buf;
}
