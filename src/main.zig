const std = @import("std");
const time = std.time.milliTimestamp;

extern "c" fn matmul(A: *const f32, B: *const f32, C: *const f32, N: u16) void;
extern "c" fn cuda_device_check() u8;

fn createRandomMatrix(N: usize, allocator: *std.mem.Allocator) ![]f32 {
    // Create a vector to hold the matrix elements
    const matrix = try allocator.alloc(f32, N * N);

    // Seed the random number generator
    var rng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.milliTimestamp())));

    // Fill the matrix with random f32 values
    for (matrix) |*value| {
        value.* = (@mod(@as(f32, @floatFromInt(rng.next())), 100.0) / 100.0) * 2.0 - 1.0; // Random value between -1.0 and 1.0
    }

    return matrix;
}

fn createConstMatrix(N: usize, allocator: *std.mem.Allocator, c: f32) ![]f32 {
    // Create a vector to hold the matrix elements
    const matrix = try allocator.alloc(f32, N * N);

    // Fill the matrix with random f32 values
    for (matrix) |*value| {
        value.* = c;
    }

    return matrix;
}

fn zig_matmul(A: *[]f32, B: *[]f32, C: *[]f32, N: usize) void {
    for (0..N) |i| {
        for (0..N) |j| {
            var sum: f32 = 0.0;
            for (0..N) |k| {
                sum += A.*[i * N + k] * B.*[k * N + j]; // pointers do not support indexing :(
            }
            C.*[i * N + j] = sum;
        }
    }
}

pub fn main() !void {
    _ = cuda_device_check();

    var allocator = std.heap.page_allocator;

    const N = 2048; // can change this now

    // set up for CUDA
    var A: []f32 = try createRandomMatrix(N, &allocator);
    var B: []f32 = try createRandomMatrix(N, &allocator);
    var C: []f32 = try createRandomMatrix(N, &allocator);

    const start_cuda = time();
    matmul(&A[0], &B[0], &C[0], N);
    const end_cuda = time();
    const cuda_time = end_cuda - start_cuda;

    // reset for zig
    var X: []f32 = try createRandomMatrix(N, &allocator);
    var Y: []f32 = try createRandomMatrix(N, &allocator);
    var Z: []f32 = try createRandomMatrix(N, &allocator);

    const start_zig = time();
    //zig matmul here
    zig_matmul(&X, &Y, &Z, N);
    const end_zig = time();
    const zig_time = end_zig - start_zig;

    // really annoying for any N over 5
    std.debug.print("\n\nZig result\n", .{});
    for (0..10) |index| {
        std.debug.print("Z[{}] = {}\n", .{ index, Z[index] });
    }
    std.debug.print("...\n", .{});

    std.debug.print("\n\nCuda result\n", .{});
    for (0..10) |index| {
        std.debug.print("C[{}] = {}\n", .{ index, C[index] });
    }

    std.debug.print("...\n", .{});
    std.debug.print("\n\nMatmul {}x{}\n---------------------\n", .{ N, N });
    std.debug.print("Pure zig: {}ms\n", .{zig_time});
    std.debug.print("zig/CUDA: {}ms\n", .{cuda_time});
}
