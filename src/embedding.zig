//! Ternary embedding vectors for semantic memory representation.
//!
//! Each dimension holds a value in {-1, 0, +1}, providing a compact and
//! discretized representation of dense float embeddings. Cosine similarity
//! on ternary vectors approximates the similarity of the original continuous
//! vectors while reducing storage by ~8x (1 bit per dimension vs 32 bits).
//!
//! Comptime dimension parameter: `TernaryVector(comptime dim: usize)`.

const std = @import("std");

/// A ternary vector with `dim` dimensions, each valued in {-1, 0, +1}.
pub fn TernaryVector(comptime dim: usize) type {
    return struct {
        data: [dim]i8,

        const Self = @This();

        pub const dimension = dim;

        /// Initialize with all zeros.
        pub fn initZero() Self {
            return .{ .data = [_]i8{0} ** dim };
        }

        /// Initialize with random ternary values {-1, 0, +1} using the provided RNG.
        pub fn initRandom(rng: std.Random) Self {
            var v = Self.initZero();
            for (&v.data) |*val| {
                const r = rng.intRangeAtMost(i8, -1, 1);
                val.* = r;
            }
            return v;
        }

        /// Discretize a float vector to ternary: < -thresh → -1, > +thresh → +1, else 0.
        pub fn fromFloats(floats: [dim]f64, threshold: f64) Self {
            var v = Self.initZero();
            for (floats, 0..) |f, i| {
                if (f > threshold) {
                    v.data[i] = 1;
                } else if (f < -threshold) {
                    v.data[i] = -1;
                } else {
                    v.data[i] = 0;
                }
            }
            return v;
        }

        /// Dot product: sum of element-wise products.
        pub fn dotProduct(self: Self, other: Self) i64 {
            var sum: i64 = 0;
            for (self.data, other.data) |a, b| {
                sum += @as(i64, @intCast(a)) * @as(i64, @intCast(b));
            }
            return sum;
        }

        /// Cosine similarity between two ternary vectors.
        /// Returns a value in [-1, 1].
        pub fn cosineSimilarity(self: Self, other: Self) f64 {
            const dp = @as(f64, @floatFromInt(self.dotProduct(other)));
            const mag_a = self.magnitude();
            const mag_b = other.magnitude();
            if (mag_a == 0 or mag_b == 0) return 0.0;
            return dp / (mag_a * mag_b);
        }

        /// Euclidean magnitude (L2 norm) as float.
        pub fn magnitude(self: Self) f64 {
            var sum: f64 = 0;
            for (self.data) |v| {
                const fv: f64 = @floatFromInt(v);
                sum += fv * fv;
            }
            return @sqrt(sum);
        }

        /// Hamming distance: number of positions where values differ.
        pub fn hammingDistance(self: Self, other: Self) usize {
            var dist: usize = 0;
            for (self.data, other.data) |a, b| {
                if (a != b) dist += 1;
            }
            return dist;
        }

        /// Count of non-zero (active) dimensions.
        pub fn activeCount(self: Self) usize {
            var c: usize = 0;
            for (self.data) |v| {
                if (v != 0) c += 1;
            }
            return c;
        }

        /// Check all values are valid ternary {-1, 0, 1}.
        pub fn isValid(self: Self) bool {
            for (self.data) |v| {
                if (v < -1 or v > 1) return false;
            }
            return true;
        }
    };
}

test "TernaryVector initZero" {
    const v = TernaryVector(8).initZero();
    for (v.data) |val| {
        try std.testing.expectEqual(@as(i8, 0), val);
    }
}

test "TernaryVector initRandom produces valid ternary values" {
    var rng = std.Random.DefaultPrng.init(123);
    const v = TernaryVector(256).initRandom(rng.random());
    try std.testing.expect(v.isValid());
}

test "TernaryVector fromFloats discretization" {
    const floats = [_]f64{ 0.5, -0.5, 0.1, -0.1, 0.9, -0.9, 0.0, 0.3 };
    const v = TernaryVector(8).fromFloats(floats, 0.2);
    try std.testing.expectEqual(@as(i8, 1), v.data[0]); // 0.5 > 0.2
    try std.testing.expectEqual(@as(i8, -1), v.data[1]); // -0.5 < -0.2
    try std.testing.expectEqual(@as(i8, 0), v.data[2]); // 0.1 in [-0.2, 0.2]
    try std.testing.expectEqual(@as(i8, 0), v.data[3]); // -0.1 in [-0.2, 0.2]
    try std.testing.expectEqual(@as(i8, 1), v.data[4]); // 0.9 > 0.2
    try std.testing.expectEqual(@as(i8, -1), v.data[5]); // -0.9 < -0.2
}

test "TernaryVector dotProduct" {
    var v1 = TernaryVector(4).initZero();
    v1.data = [_]i8{ 1, -1, 0, 1 };
    var v2 = TernaryVector(4).initZero();
    v2.data = [_]i8{ 1, 1, 0, -1 };
    try std.testing.expectEqual(@as(i64, -1), v1.dotProduct(v2)); // 1 - 1 + 0 - 1 = -1
}

test "TernaryVector cosineSimilarity orthogonal" {
    var v1 = TernaryVector(4).initZero();
    v1.data = [_]i8{ 1, 0, 0, 0 };
    var v2 = TernaryVector(4).initZero();
    v2.data = [_]i8{ 0, 1, 0, 0 };
    const sim = v1.cosineSimilarity(v2);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), sim, 1e-10);
}

test "TernaryVector cosineSimilarity identical" {
    var v = TernaryVector(4).initZero();
    v.data = [_]i8{ 1, -1, 1, 0 };
    const sim = v.cosineSimilarity(v);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sim, 1e-10);
}

test "TernaryVector cosineSimilarity opposite" {
    var v1 = TernaryVector(4).initZero();
    v1.data = [_]i8{ 1, 1, 1, 1 };
    var v2 = TernaryVector(4).initZero();
    v2.data = [_]i8{ -1, -1, -1, -1 };
    const sim = v1.cosineSimilarity(v2);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), sim, 1e-10);
}

test "TernaryVector hammingDistance" {
    var v1 = TernaryVector(4).initZero();
    v1.data = [_]i8{ 1, -1, 0, 1 };
    var v2 = TernaryVector(4).initZero();
    v2.data = [_]i8{ 1, 1, 0, -1 };
    try std.testing.expectEqual(@as(usize, 2), v1.hammingDistance(v2));
}

test "TernaryVector magnitude" {
    var v = TernaryVector(4).initZero();
    v.data = [_]i8{ 1, -1, 0, 1 };
    const mag = v.magnitude();
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(3.0)), mag, 1e-10);
}

test "TernaryVector activeCount" {
    var v = TernaryVector(4).initZero();
    v.data = [_]i8{ 1, 0, -1, 0 };
    try std.testing.expectEqual(@as(usize, 2), v.activeCount());
}

test "TernaryVector isValid" {
    var v = TernaryVector(4).initZero();
    v.data = [_]i8{ 1, -1, 0, 1 };
    try std.testing.expect(v.isValid());
}

test "TernaryVector zero vector cosine similarity" {
    const v1 = TernaryVector(4).initZero();
    var v2 = TernaryVector(4).initZero();
    v2.data = [_]i8{ 1, 0, 0, 0 };
    try std.testing.expectEqual(@as(f64, 0.0), v1.cosineSimilarity(v2));
}
