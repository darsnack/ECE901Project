`timescale 1ns / 1ps
module stochastic_quantizer(CLK, RESET, in, out);

parameter WL = 16;
parameter FL = 15;

input CLK;
input RESET;
input [(2*WL - 1):0] in;
output [(WL - 1):0] out;

reg [(FL - 1):0] rand_num;
wire [(2*WL - 1):0] truncated_val;

lfsr #(.WL(FL)) rng(
    .CLK(CLK),
    .RESET(RESET),
    .out(rand_num)
);

assign truncated_val = (in + {{(2*WL - 1 - FL){1'b0}}, rand_num}) >> FL;
assign out = truncated_val[(WL - 1):0];

endmodule
