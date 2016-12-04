`timescale 1ns / 1ps
module pool2x2(in1, in2, in3, in4, out);

parameter WL = 16;
parameter FL = 14;

input signed [(WL - 1):0] in1, in2, in3, in4;
output signed [(WL - 1):0] out;

wire signed [(WL - 1):0] compare1, compare2;

max2 #(.WL(WL), .FL(FL)) max1(
    .a(in1),
    .b(in2),
    .out(compare1)
);

max2 #(.WL(WL), .FL(FL)) max2(
    .a(in3),
    .b(in4),
    .out(compare2)
);

max2 #(.WL(WL), .FL(FL)) max3(
    .a(compare1),
    .b(compare2),
    .out(out)
);

endmodule
