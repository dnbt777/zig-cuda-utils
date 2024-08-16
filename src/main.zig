const std = @import("std");

extern "c" fn matmul(A: *const f32, B: *const f32, C: *const f32) void;

pub fn main() !void {
    // const allocator = std.heap.page_allocator;

    const N = 2;
    var A: [N * N]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    var B: [N * N]f32 = .{ 5.0, 6.0, 7.0, 8.0 };
    var C: [N * N]f32 = .{ 0.0, 0.0, 0.0, 0.0 };

    matmul(&A[0], &B[0], &C[0]);

    for (0.., C) |index, value| {
        std.debug.print("C[{}] = {}\n", .{ index, value });
    }
}
