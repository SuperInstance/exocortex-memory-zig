const std = @import("std");
const embedding = @import("exocortex_memory").embedding;

test "embedding_test: initZero all zeros" {
    const v = embedding.TernaryVector(16).initZero();
    for (v.data) |val| {
        try std.testing.expectEqual(@as(i8, 0), val);
    }
}

test "embedding_test: initRandom valid ternary" {
    var rng = std.Random.DefaultPrng.init(42);
    const v = embedding.TernaryVector(512).initRandom(rng.random());
    for (v.data) |val| {
        try std.testing.expect(val >= -1 and val <= 1);
    }
}

test "embedding_test: fromFloats threshold boundaries" {
    const floats = [_]f64{ 0.3, -0.3, 0.2, -0.2, 0.0, 0.5, -0.5, 1.0 };
    const v = embedding.TernaryVector(8).fromFloats(floats, 0.25);
    try std.testing.expectEqual(@as(i8, 1), v.data[0]); // 0.3 > 0.25
    try std.testing.expectEqual(@as(i8, -1), v.data[1]); // -0.3 < -0.25
    try std.testing.expectEqual(@as(i8, 0), v.data[2]); // 0.2 in [-0.25, 0.25]
    try std.testing.expectEqual(@as(i8, 0), v.data[3]); // -0.2 in [-0.25, 0.25]
}

test "embedding_test: dotProduct known vectors" {
    var a = embedding.TernaryVector(6).initZero();
    a.data = [_]i8{ 1, 1, 1, -1, -1, 0 };
    var b = embedding.TernaryVector(6).initZero();
    b.data = [_]i8{ 1, -1, 0, 1, -1, 1 };
    // 1*1 + 1*(-1) + 1*0 + (-1)*1 + (-1)*(-1) + 0*1 = 1-1+0-1+1+0 = 0
    try std.testing.expectEqual(@as(i64, 0), a.dotProduct(b));
}

test "embedding_test: cosineSimilarity known angle" {
    // (1,1,0,0) and (0,0,1,1): orthogonal → similarity = 0
    var a = embedding.TernaryVector(4).initZero();
    a.data = [_]i8{ 1, 1, 0, 0 };
    var b = embedding.TernaryVector(4).initZero();
    b.data = [_]i8{ 0, 0, 1, 1 };
    const sim = a.cosineSimilarity(b);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), sim, 1e-10);
}

test "embedding_test: cosineSimilarity 45-degree angle" {
    // (1,1,0) and (1,0,0): cos(45°) ≈ 0.707
    var a = embedding.TernaryVector(3).initZero();
    a.data = [_]i8{ 1, 1, 0 };
    var b = embedding.TernaryVector(3).initZero();
    b.data = [_]i8{ 1, 0, 0 };
    const sim = a.cosineSimilarity(b);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / @sqrt(2.0)), sim, 1e-10);
}

test "embedding_test: hammingDistance self is zero" {
    var v = embedding.TernaryVector(8).initZero();
    v.data = [_]i8{ 1, -1, 0, 1, -1, 0, 1, 1 };
    try std.testing.expectEqual(@as(usize, 0), v.hammingDistance(v));
}

test "embedding_test: hammingDistance all differ" {
    var a = embedding.TernaryVector(4).initZero();
    a.data = [_]i8{ 1, 1, 1, 1 };
    var b = embedding.TernaryVector(4).initZero();
    b.data = [_]i8{ -1, -1, -1, -1 };
    try std.testing.expectEqual(@as(usize, 4), a.hammingDistance(b));
}

test "embedding_test: activeCount" {
    var v = embedding.TernaryVector(6).initZero();
    v.data = [_]i8{ 1, 0, -1, 0, 1, 0 };
    try std.testing.expectEqual(@as(usize, 3), v.activeCount());
}

test "embedding_test: isValid rejects invalid" {
    const T = embedding.TernaryVector(2);
    var v = T.initZero();
    v.data = [_]i8{ 1, 2 }; // 2 is not ternary
    try std.testing.expect(!v.isValid());
}

test "embedding_test: dimension constant" {
    try std.testing.expectEqual(@as(usize, 128), embedding.TernaryVector(128).dimension);
    try std.testing.expectEqual(@as(usize, 256), embedding.TernaryVector(256).dimension);
}

test "embedding_test: magnitude zero vector" {
    const v = embedding.TernaryVector(4).initZero();
    try std.testing.expectEqual(@as(f64, 0.0), v.magnitude());
}

test "embedding_test: fromFloats with zero threshold" {
    const floats = [_]f64{ 0.01, -0.01, 0.0, 0.5, -0.5, 1.0, -1.0, 0.001 };
    const v = embedding.TernaryVector(8).fromFloats(floats, 0.0);
    try std.testing.expectEqual(@as(i8, 1), v.data[0]); // > 0
    try std.testing.expectEqual(@as(i8, -1), v.data[1]); // < 0
    try std.testing.expectEqual(@as(i8, 0), v.data[2]); // == 0
}
