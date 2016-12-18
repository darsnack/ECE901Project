`timescale 1ns / 1ps
module max2(a, b, out);

parameter WL = 16;
parameter FL = 14;

input signed [(WL - 1):0] a;
input signed [(WL - 1):0] b;
output signed [(WL - 1):0] out;

wire signed [(WL - 1):0] sub_val;
wire overflow;

assign sub_val = a - b;
assign overflow = (a[WL - 1] ^ b[WL - 1]) & ~(b[WL - 1] ^ sub_val[WL - 1]);
assign out = overflow ? (b[WL - 1] ? a : b) : (sub_val[WL - 1] ? b : a);

endmodule
