`timescale 1ns / 1ps
module fc_gr_update16x1 (CLK, RESET,
    err_factor, 
    in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15, in16,
    wght1, wght2, wght3, wght4, wght5, wght6, wght7, wght8, wght9, wght10, wght11, wght12, wght13, wght14, wght15, wght16,
    out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16);

parameter PATCH_LENGTH = 16;
parameter WL = 16;
parameter FL = 14;
parameter LEARNING_RATE = 16'd15552; // 0.95 using WL = 16 and FL = 14

input CLK, RESET;
input signed [(WL - 1):0] err_factor;
input signed [(WL - 1):0] in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15, in16;
input signed [(WL - 1):0] wght1, wght2, wght3, wght4, wght5, wght6, wght7, wght8, wght9, wght10, wght11, wght12, wght13, wght14, wght15, wght16;
output signed [(WL - 1):0] out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16;

wire signed [(WL - 1):0] intermediate_val[0:(PATCH_LENGTH - 1)];
wire signed [(WL - 1):0] gr_update [0:(PATCH_LENGTH - 1)];

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer1(
    .CLK(CLK),
    .RESET(RESET),
    .in((err_factor * in1)),
    .out(intermediate_val[0])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer2(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * in2),
    .out(intermediate_val[1])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer3(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * in3),
    .out(intermediate_val[2])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer4(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * in4),
    .out(intermediate_val[3])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer5(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * in5),
    .out(intermediate_val[4])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer6(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * in6),
    .out(intermediate_val[5])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer7(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * in7),
    .out(intermediate_val[6])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer8(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * in8),
    .out(intermediate_val[7])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer9(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * in9),
    .out(intermediate_val[8])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer10(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * in10),
    .out(intermediate_val[9])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer11(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * in11),
    .out(intermediate_val[10])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer12(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * in12),
    .out(intermediate_val[11])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer13(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * in13),
    .out(intermediate_val[12])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer14(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * in14),
    .out(intermediate_val[13])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer15(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * in15),
    .out(intermediate_val[14])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer16(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * in16),
    .out(intermediate_val[15])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer1(
    .CLK(CLK),
    .RESET(RESET),
    .in($signed(LEARNING_RATE) * intermediate_val[0]),
    .out(gr_update[0])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer2(
    .CLK(CLK),
    .RESET(RESET),
    .in($signed(LEARNING_RATE) * intermediate_val[1]),
    .out(gr_update[1])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer3(
    .CLK(CLK),
    .RESET(RESET),
    .in($signed(LEARNING_RATE) * intermediate_val[2]),
    .out(gr_update[2])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer4(
    .CLK(CLK),
    .RESET(RESET),
    .in($signed(LEARNING_RATE) * intermediate_val[3]),
    .out(gr_update[3])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer5(
    .CLK(CLK),
    .RESET(RESET),
    .in($signed(LEARNING_RATE) * intermediate_val[4]),
    .out(gr_update[4])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer6(
    .CLK(CLK),
    .RESET(RESET),
    .in($signed(LEARNING_RATE) * intermediate_val[5]),
    .out(gr_update[5])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer7(
    .CLK(CLK),
    .RESET(RESET),
    .in($signed(LEARNING_RATE) * intermediate_val[6]),
    .out(gr_update[6])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer8(
    .CLK(CLK),
    .RESET(RESET),
    .in($signed(LEARNING_RATE) * intermediate_val[7]),
    .out(gr_update[7])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer9(
    .CLK(CLK),
    .RESET(RESET),
    .in($signed(LEARNING_RATE) * intermediate_val[8]),
    .out(gr_update[8])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer10(
    .CLK(CLK),
    .RESET(RESET),
    .in($signed(LEARNING_RATE) * intermediate_val[9]),
    .out(gr_update[9])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer11(
    .CLK(CLK),
    .RESET(RESET),
    .in($signed(LEARNING_RATE) * intermediate_val[10]),
    .out(gr_update[10])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer12(
    .CLK(CLK),
    .RESET(RESET),
    .in(LEARNING_RATE * intermediate_val[11]),
    .out(gr_update[11])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer13(
    .CLK(CLK),
    .RESET(RESET),
    .in(LEARNING_RATE * intermediate_val[12]),
    .out(gr_update[12])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer14(
    .CLK(CLK),
    .RESET(RESET),
    .in(LEARNING_RATE * intermediate_val[13]),
    .out(gr_update[13])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer15(
    .CLK(CLK),
    .RESET(RESET),
    .in(LEARNING_RATE * intermediate_val[14]),
    .out(gr_update[14])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer16(
    .CLK(CLK),
    .RESET(RESET),
    .in(LEARNING_RATE * intermediate_val[15]),
    .out(gr_update[15])
);

sub_with_saturate #(.WL(WL), .FL(FL)) output_sub1(
    .a(wght1),
    .b(gr_update[0]),
    .out(out1)
);

sub_with_saturate #(.WL(WL), .FL(FL)) output_sub2(
    .a(wght2),
    .b(gr_update[1]),
    .out(out2)
);

sub_with_saturate #(.WL(WL), .FL(FL)) output_sub3(
    .a(wght3),
    .b(gr_update[2]),
    .out(out3)
);

sub_with_saturate #(.WL(WL), .FL(FL)) output_sub4(
    .a(wght4),
    .b(gr_update[3]),
    .out(out4)
);

sub_with_saturate #(.WL(WL), .FL(FL)) output_sub5(
    .a(wght5),
    .b(gr_update[4]),
    .out(out5)
);

sub_with_saturate #(.WL(WL), .FL(FL)) output_sub6(
    .a(wght6),
    .b(gr_update[5]),
    .out(out6)
);

sub_with_saturate #(.WL(WL), .FL(FL)) output_sub7(
    .a(wght7),
    .b(gr_update[6]),
    .out(out7)
);

sub_with_saturate #(.WL(WL), .FL(FL)) output_sub8(
    .a(wght8),
    .b(gr_update[7]),
    .out(out8)
);

sub_with_saturate #(.WL(WL), .FL(FL)) output_sub9(
    .a(wght9),
    .b(gr_update[8]),
    .out(out9)
);

sub_with_saturate #(.WL(WL), .FL(FL)) output_sub10(
    .a(wght10),
    .b(gr_update[9]),
    .out(out10)
);

sub_with_saturate #(.WL(WL), .FL(FL)) output_sub11(
    .a(wght11),
    .b(gr_update[10]),
    .out(out11)
);

sub_with_saturate #(.WL(WL), .FL(FL)) output_sub12(
    .a(wght12),
    .b(gr_update[11]),
    .out(out12)
);

sub_with_saturate #(.WL(WL), .FL(FL)) output_sub13(
    .a(wght13),
    .b(gr_update[12]),
    .out(out13)
);

sub_with_saturate #(.WL(WL), .FL(FL)) output_sub14(
    .a(wght14),
    .b(gr_update[13]),
    .out(out14)
);

sub_with_saturate #(.WL(WL), .FL(FL)) output_sub15(
    .a(wght15),
    .b(gr_update[14]),
    .out(out15)
);

sub_with_saturate #(.WL(WL), .FL(FL)) output_sub16(
    .a(wght16),
    .b(gr_update[15]),
    .out(out16)
);

endmodule