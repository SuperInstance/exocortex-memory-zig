//! Comptime schema validation — the heart of exocortex-memory-zig.
//!
//! This module demonstrates that Zig's comptime can replace C macros,
//! C++ templates, and Rust derive macros for schema validation, producing
//! compile-time errors with zero runtime cost.
//!
//! "The type system is a theorem prover. Comptime is the proof."

const std = @import("std");
const insight_mod = @import("insight.zig");
const embedding_mod = @import("embedding.zig");

/// Error context for schema validation (used at comptime).
pub const SchemaError = error{
    MissingRequiredField,
    EmptyTags,
    ConfidenceOutOfRange,
    EmbeddingDimensionMismatch,
    InvalidInsightStruct,
};

/// Validate at comptime that a struct type conforms to the Insight schema.
pub fn validateInsightSchema(comptime T: type, comptime expected_dim: usize) bool {
    const info = @typeInfo(T);
    if (info != .Struct) {
        @compileError("Schema validation failed: expected struct, got " ++ @typeName(T));
    }

    const fields = info.Struct.fields;

    // Check required fields
    comptime {
        const required = [_][]const u8{ "id", "text", "tags", "tags_count", "confidence", "timestamp", "embedding" };
        for (required) |req| {
            var found = false;
            for (fields) |f| {
                if (std.mem.eql(u8, f.name, req)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                @compileError("Schema validation failed: missing required field '" ++ req ++ "' in " ++ @typeName(T));
            }
        }
    }

    // Check field types
    comptime {
        for (fields) |f| {
            if (std.mem.eql(u8, f.name, "id")) {
                if (f.type != u64) @compileError("Schema validation failed: 'id' must be u64");
            }
            if (std.mem.eql(u8, f.name, "confidence")) {
                if (f.type != f64) @compileError("Schema validation failed: 'confidence' must be f64");
            }
            if (std.mem.eql(u8, f.name, "timestamp")) {
                if (f.type != i64) @compileError("Schema validation failed: 'timestamp' must be i64");
            }
        }
    }

    _ = expected_dim;
    return true;
}

/// Comptime check: confidence value is in [0, 1].
pub fn validateConfidence(comptime conf: f64) void {
    if (conf < 0.0 or conf > 1.0) {
        @compileError("Schema validation failed: confidence out of range [0, 1]");
    }
}

/// Comptime check: tags are not empty.
pub fn validateTagsNonEmpty(comptime tags_count: usize) void {
    if (tags_count == 0) {
        @compileError("Schema validation failed: tags must not be empty");
    }
}

/// Comptime check: embedding dimension matches expected.
pub fn validateEmbeddingDim(comptime EmbeddingType: type, comptime expected_dim: usize) void {
    if (!@hasDecl(EmbeddingType, "dimension")) {
        @compileError("Schema validation failed: embedding type has no 'dimension' decl");
    }
    if (EmbeddingType.dimension != expected_dim) {
        @compileError("Schema validation failed: embedding dimension mismatch");
    }
}

/// Validate an Insight value at comptime.
pub fn validateInsight(comptime ins: anytype, comptime expected_dim: usize) void {
    _ = validateInsightSchema(@TypeOf(ins), expected_dim);

    if (ins.confidence < 0.0 or ins.confidence > 1.0) {
        @compileError("Schema validation failed: confidence out of range [0, 1]");
    }

    if (ins.tags_count == 0) {
        @compileError("Schema validation failed: insight has no tags");
    }
}

/// Count fields in a struct at comptime.
pub fn countFields(comptime T: type) comptime_int {
    return @typeInfo(T).Struct.fields.len;
}

/// Check if a struct has a field with the given name at comptime.
pub fn hasField(comptime T: type, comptime name: []const u8) bool {
    return @hasField(T, name);
}

/// Get the type of a field by name at comptime.
pub fn getFieldType(comptime T: type, comptime name: []const u8) type {
    const fields = @typeInfo(T).Struct.fields;
    inline for (fields) |f| {
        if (std.mem.eql(u8, f.name, name)) return f.type;
    }
    @compileError("No field '" ++ name ++ "' in " ++ @typeName(T));
}

/// Get field names of a struct at comptime.
pub fn getFieldNames(comptime T: type) []const []const u8 {
    const info = @typeInfo(T);
    if (info != .Struct) @compileError("Expected struct type");
    return comptime blk: {
        const fields = info.Struct.fields;
        var names: [fields.len][]const u8 = undefined;
        for (fields, 0..) |f, i| {
            names[i] = f.name;
        }
        const final_names = names;
        break :blk final_names[0..];
    };
}

test "schema validateInsightSchema accepts valid Insight" {
    const valid = validateInsightSchema(insight_mod.Insight(128), 128);
    try std.testing.expect(valid);
}

test "schema validateInsightSchema accepts different dims" {
    const valid = validateInsightSchema(insight_mod.Insight(64), 64);
    try std.testing.expect(valid);
}

test "schema getFieldNames" {
    const names = getFieldNames(insight_mod.Insight(4));
    try std.testing.expect(names.len >= 7);
}

test "schema countFields" {
    const cnt = countFields(insight_mod.Insight(4));
    try std.testing.expect(cnt == 7);
}

test "schema hasField positive" {
    try std.testing.expect(hasField(insight_mod.Insight(4), "id"));
    try std.testing.expect(hasField(insight_mod.Insight(4), "text"));
    try std.testing.expect(hasField(insight_mod.Insight(4), "confidence"));
}

test "schema hasField negative" {
    try std.testing.expect(!hasField(insight_mod.Insight(4), "nonexistent"));
}

test "schema getFieldType" {
    try std.testing.expect(getFieldType(insight_mod.Insight(4), "id") == u64);
    try std.testing.expect(getFieldType(insight_mod.Insight(4), "confidence") == f64);
    try std.testing.expect(getFieldType(insight_mod.Insight(4), "timestamp") == i64);
}

test "schema validateConfidence accepts valid" {
    comptime validateConfidence(0.5);
    comptime validateConfidence(0.0);
    comptime validateConfidence(1.0);
}

test "schema validateTagsNonEmpty accepts non-zero" {
    comptime validateTagsNonEmpty(1);
    comptime validateTagsNonEmpty(5);
}

test "schema validateEmbeddingDim accepts matching" {
    const TV = embedding_mod.TernaryVector(64);
    comptime validateEmbeddingDim(TV, 64);
}

test "schema full validateInsight accepts valid insight" {
    const Dim = 8;
    const ins = comptime insight_mod.Insight(Dim).builder(1)
        .setText("test")
        .addTag("zig")
        .setConfidence(0.9)
        .setTimestamp(1000)
        .build();
    comptime validateInsight(ins, Dim);
}

test "schema hasField works on generic struct" {
    const S = struct { x: u32, y: f64, name: []const u8 };
    try std.testing.expect(hasField(S, "x"));
    try std.testing.expect(hasField(S, "name"));
    try std.testing.expect(!hasField(S, "z"));
}

test "schema countFields on simple struct" {
    const S = struct { a: u8, b: u16 };
    try std.testing.expectEqual(@as(comptime_int, 2), countFields(S));
}
