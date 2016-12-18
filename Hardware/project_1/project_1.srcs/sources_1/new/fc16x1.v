`timescale 1ns / 1ps
module fc16x1(CLK, RESET, in11, in12, in13, in14, in15, in16, in17, in18, in19, in110, in111, in112, in113, in114, in115, in116,
    in21, in22, in23, in24, in25, in26, in27, in28, in29, in210, in211, in212, in213, in214, in215, in216,
    out);

parameter WL = 16;
parameter FL = 14;
parameter PATCH_LENGTH = 16;

input CLK, RESET;
input signed [(WL - 1):0] in11, in12, in13, in14, in15, in16, in17, in18, in19, in110, in111, in112, in113, in114, in115, in116;
input signed [(WL - 1):0] in21, in22, in23, in24, in25, in26, in27, in28, in29, in210, in211, in212, in213, in214, in215, in216;
output signed [(WL - 1):0] out;

wire signed [(2*WL - 1):0] prod [0:(PATCH_LENGTH - 1)];
wire signed [(2*WL - 1):0] prod_sum [0:(PATCH_LENGTH - 2)];

assign prod[0] = in11 * in21;
assign prod[1] = in12 * in22;
assign prod[2] = in13 * in23;
assign prod[3] = in14 * in24;
assign prod[4] = in15 * in25;
assign prod[5] = in16 * in26;
assign prod[6] = in17 * in27;
assign prod[7] = in18 * in28;
assign prod[8] = in19 * in29;
assign prod[9] = in110 * in210;
assign prod[10] = in111 * in211;
assign prod[11] = in112 * in212;
assign prod[12] = in113 * in213;
assign prod[13] = in114 * in214;
assign prod[14] = in115 * in215;
assign prod[15] = in116 * in216;

add_with_saturate #(.WL(2*WL - 1), .FL(2*FL)) add1(
    .a(prod[0]),
    .b(prod[1]),
    .out(prod_sum[0])
);
add_with_saturate #(.WL(2*WL - 1), .FL(2*FL)) add2(
    .a(prod[2]),
    .b(prod[3]),
    .out(prod_sum[1])
);
add_with_saturate #(.WL(2*WL - 1), .FL(2*FL)) add3(
    .a(prod[4]),
    .b(prod[5]),
    .out(prod_sum[2])
);
add_with_saturate #(.WL(2*WL - 1), .FL(2*FL)) add4(
    .a(prod[6]),
    .b(prod[7]),
    .out(prod_sum[3])
);
add_with_saturate #(.WL(2*WL - 1), .FL(2*FL)) add5(
    .a(prod[8]),
    .b(prod[9]),
    .out(prod_sum[4])
);
add_with_saturate #(.WL(2*WL - 1), .FL(2*FL)) add6(
    .a(prod[10]),
    .b(prod[11]),
    .out(prod_sum[5])
);
add_with_saturate #(.WL(2*WL - 1), .FL(2*FL)) add7(
    .a(prod[12]),
    .b(prod[13]),
    .out(prod_sum[6])
);
add_with_saturate #(.WL(2*WL - 1), .FL(2*FL)) add8(
    .a(prod[14]),
    .b(prod[15]),
    .out(prod_sum[7])
);
add_with_saturate #(.WL(2*WL - 1), .FL(2*FL)) add9(
    .a(prod_sum[0]),
    .b(prod_sum[1]),
    .out(prod_sum[8])
);
add_with_saturate #(.WL(2*WL - 1), .FL(2*FL)) add10(
    .a(prod_sum[2]),
    .b(prod_sum[3]),
    .out(prod_sum[9])
);
add_with_saturate #(.WL(2*WL - 1), .FL(2*FL)) add11(
    .a(prod_sum[4]),
    .b(prod_sum[5]),
    .out(prod_sum[10])
);
add_with_saturate #(.WL(2*WL - 1), .FL(2*FL)) add12(
    .a(prod_sum[6]),
    .b(prod_sum[7]),
    .out(prod_sum[11])
);
add_with_saturate #(.WL(2*WL - 1), .FL(2*FL)) add13(
    .a(prod_sum[8]),
    .b(prod_sum[9]),
    .out(prod_sum[12])
);
add_with_saturate #(.WL(2*WL - 1), .FL(2*FL)) add14(
    .a(prod_sum[10]),
    .b(prod_sum[11]),
    .out(prod_sum[13])
);
add_with_saturate #(.WL(2*WL - 1), .FL(2*FL)) add15(
    .a(prod_sum[12]),
    .b(prod_sum[13]),
    .out(prod_sum[14])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) quantizer(
    .CLK(CLK),
    .RESET(RESET),
    .in({prod_sum[14][2*WL - 2], prod_sum[14][(2*WL - 2):0]}),
    .out(out)
);

endmodule
