`timescale 1ns / 1ps
module fc_bp16x1 (CLK, RESET,
    err_factor, 
    der1, der2, der3, der4, der5, der6, der7, der8, der9, der10, der11, der12, der13, der14, der15, der16,
    wght1, wght2, wght3, wght4, wght5, wght6, wght7, wght8, wght9, wght10, wght11, wght12, wght13, wght14, wght15, wght16,
    out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16);

parameter PATCH_LENGTH = 16;
parameter WL = 16;
parameter FL = 14;

input CLK, RESET;
input signed [(WL - 1):0] err_factor;
input signed [(WL - 1):0] der1, der2, der3, der4, der5, der6, der7, der8, der9, der10, der11, der12, der13, der14, der15, der16;
input signed [(WL - 1):0] wght1, wght2, wght3, wght4, wght5, wght6, wght7, wght8, wght9, wght10, wght11, wght12, wght13, wght14, wght15, wght16;
output signed [(WL - 1):0] out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16;

wire signed [(WL - 1):0] intermediate_val[0:(PATCH_LENGTH - 1)];

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer1(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * wght1),
    .out(intermediate_val[0])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer2(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * wght2),
    .out(intermediate_val[1])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer3(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * wght3),
    .out(intermediate_val[2])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer4(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * wght4),
    .out(intermediate_val[3])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer5(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * wght5),
    .out(intermediate_val[4])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer6(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * wght6),
    .out(intermediate_val[5])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer7(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * wght7),
    .out(intermediate_val[6])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer8(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * wght8),
    .out(intermediate_val[7])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer9(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor * wght9),
    .out(intermediate_val[8])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer10(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor *  wght10),
    .out(intermediate_val[9])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer11(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor *  wght11),
    .out(intermediate_val[10])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer12(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor *  wght12),
    .out(intermediate_val[11])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer13(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor *  wght13),
    .out(intermediate_val[12])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer14(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor *  wght14),
    .out(intermediate_val[13])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer15(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor *  wght15),
    .out(intermediate_val[14])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) intermediate_quantizer16(
    .CLK(CLK),
    .RESET(RESET),
    .in(err_factor *  wght16),
    .out(intermediate_val[15])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer1(
    .CLK(CLK),
    .RESET(RESET),
    .in(der1 * intermediate_val[0]),
    .out(out1)
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer2(
    .CLK(CLK),
    .RESET(RESET),
    .in(der2 * intermediate_val[1]),
    .out(out2)
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer3(
    .CLK(CLK),
    .RESET(RESET),
    .in(der3 * intermediate_val[2]),
    .out(out3)
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer4(
    .CLK(CLK),
    .RESET(RESET),
    .in(der4 * intermediate_val[3]),
    .out(out4)
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer5(
    .CLK(CLK),
    .RESET(RESET),
    .in(der5 * intermediate_val[4]),
    .out(out5)
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer6(
    .CLK(CLK),
    .RESET(RESET),
    .in(der6 * intermediate_val[5]),
    .out(out6)
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer7(
    .CLK(CLK),
    .RESET(RESET),
    .in(der7 * intermediate_val[6]),
    .out(out7)
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer8(
    .CLK(CLK),
    .RESET(RESET),
    .in(der8 * intermediate_val[7]),
    .out(out8)
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer9(
    .CLK(CLK),
    .RESET(RESET),
    .in(der9 * intermediate_val[8]),
    .out(out9)
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer10(
    .CLK(CLK),
    .RESET(RESET),
    .in(der10 * intermediate_val[9]),
    .out(out10)
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer11(
    .CLK(CLK),
    .RESET(RESET),
    .in(der11 * intermediate_val[10]),
    .out(out11)
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer12(
    .CLK(CLK),
    .RESET(RESET),
    .in(der12 * intermediate_val[11]),
    .out(out12)
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer13(
    .CLK(CLK),
    .RESET(RESET),
    .in(der13 * intermediate_val[12]),
    .out(out13)
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer14(
    .CLK(CLK),
    .RESET(RESET),
    .in(der14 * intermediate_val[13]),
    .out(out14)
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer15(
    .CLK(CLK),
    .RESET(RESET),
    .in(der15 * intermediate_val[14]),
    .out(out15)
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_quantizer16(
    .CLK(CLK),
    .RESET(RESET),
    .in(der16 * intermediate_val[15]),
    .out(out16)
);

endmodule