const std = @import("std");

extern "c" fn matmul(A: *const f32, B: *const f32, C: *const f32) void;
extern "c" fn cuda_device_check() u8;

fn createRandomMatrix(N: usize, allocator: *std.mem.Allocator) ![]f32 {
    // Create a vector to hold the matrix elements
    const matrix = try allocator.alloc(f32, N * N);

    // Seed the random number generator
    var rng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.milliTimestamp())));

    // Fill the matrix with random f32 values
    for (matrix) |*value| {
        value.* = @mod(@as(f32, @floatFromInt(rng.next())), 100.0); // Random value between 0 and 99
    }

    return matrix;
}

pub fn main() !void {
    _ = cuda_device_check();

    var allocator = std.heap.page_allocator;

    const N = 2048;
    var A: []f32 = try createRandomMatrix(N, &allocator);
    var B: []f32 = try createRandomMatrix(N, &allocator);
    var C: []f32 = try createRandomMatrix(N, &allocator);

    matmul(&A[0], &B[0], &C[0]);

    for (0.., C) |index, value| {
        std.debug.print("C[{}] = {}\n", .{ index, value });
    }
}
