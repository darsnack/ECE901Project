`timescale 1ns / 1ps
module sigma_prime(in, out);

parameter WL = 16;
parameter FL = 14;

input [(WL - 1):0] in;
output [(WL - 1):0] out;

assign out = (in[WL - 1] == 1'b1) ? 0 : (1 << FL);

endmodule