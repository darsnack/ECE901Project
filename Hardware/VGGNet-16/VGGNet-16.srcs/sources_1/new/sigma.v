`timescale 1ns / 1ps
module sigma(in, out);

parameter WL = 16;

input [(WL - 1):0] in;
output [(WL - 1):0] out;

assign out = (in <= 0) ? 0 : in;

endmodule