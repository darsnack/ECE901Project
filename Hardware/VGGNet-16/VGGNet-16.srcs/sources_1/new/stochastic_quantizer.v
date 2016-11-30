`timescale 1ns / 1ps
module stochastic_quantizer(CLK, RESET, in, out, overflow);

parameter WL = 16;
parameter FL = 15;

input CLK;
input RESET;
input signed [(2*WL - 1):0] in;
output signed [(WL - 1):0] out;
output overflow;

wire [(FL - 1):0] rand_num;
wire signed [(2*WL - 1):0] truncated_val;
wire signed [(2*WL - 1):0] abs_in;

lfsr #(.WL(FL)) rng(
    .CLK(CLK),
    .RESET(RESET),
    .out(rand_num)
);

assign abs_in = (in[2*WL - 1] == 1'b1) ? (~in + 1'b1) : in;
// Use this line if IL = 1
assign overflow = abs_in[(2*WL - 3)];
// Use this line if IL > 1
//assign overflow = (abs_in[(2*WL - 3):(FL + WL - 1)] != 0);
assign truncated_val = (in + {{(2*WL - FL){1'b0}}, rand_num}) >> FL;
assign out = overflow ? ((in[2*WL - 1] == 1'b1) ? (1 << (WL - 1)) : {1'b0, {(WL - 1){1'b1}}}) : truncated_val[(WL - 1):0];

endmodule
