`timescale 1ns / 1ps
module add_with_saturate(a, b, out);

parameter WL = 16;
parameter FL = 14;
parameter LARGEST_NUM = {1'b0, {(WL - 1){1'b1}}};
parameter SMALLEST_NUM = (1 << (WL - 1));

input [(WL - 1):0] a, b;
output [(WL - 1):0] out;

wire [(WL - 1):0] add_val;
wire overflow;
wire overflow_dir;

assign add_val = a + b;
assign overflow = ~(a[WL - 1] ^ b[WL - 1]) & (b[WL - 1] ^ add_val[WL - 1]);
assign overflow_dir = a[WL - 1];
assign out = overflow ? (overflow_dir ? SMALLEST_NUM : LARGEST_NUM) : add_val;

endmodule