const std = @import("std");
const time = std.time.milliTimestamp;

extern "c" fn matmul(A: *const f32, B: *const f32, C: *const f32, N: u8) void;
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

fn zig_matmul(A: []const f32, B: []const f32, C: []f32, N: usize) void {
    for (0..N) |i| {
        for (0..N) |j| {
            var sum: f32 = 0.0;
            for (0..N) |k| {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

pub fn main() !void {
    _ = cuda_device_check();

    var allocator = std.heap.page_allocator;

    const N = 5; // can change this now
    var A: []f32 = try createRandomMatrix(N, &allocator);
    var B: []f32 = try createRandomMatrix(N, &allocator);
    var C: []f32 = try createRandomMatrix(N, &allocator);

    const start_cuda = time();
    matmul(&A[0], &B[0], &C[0], N);
    const end_cuda = time();
    const cuda_time = end_cuda - start_cuda;

    const start_zig = time();
    //zig matmul here
    zig_matmul(A, B, C, N);
    const end_zig = time();
    const zig_time = end_zig - start_zig;

    for (0.., C) |index, value| {
        std.debug.print("C[{}] = {}\n", .{ index, value });
    }

    std.debug.print("\n\nMatmul {}x{}\n---------------------\n", .{ N, N });
    std.debug.print("Zig: {}ms\n", .{zig_time});
    std.debug.print("CUDA: {}ms\n", .{cuda_time});
}
