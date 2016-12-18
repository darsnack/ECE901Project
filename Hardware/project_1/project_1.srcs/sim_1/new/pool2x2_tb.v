`timescale 1ns / 1ps
module pool2x2_tb;

parameter WL = 16;
parameter FL = 14;

reg signed [(WL - 1):0] in1, in2, in3, in4;
wire signed [(WL - 1):0] out;

pool2x2 #(.WL(WL), .FL(FL)) dut(
    .in1(in1),
    .in2(in2),
    .in3(in3),
    .in4(in4),
    .out(out)
);

reg CLK;
always #5 CLK = ~CLK;

initial begin
    CLK = 0;
    in1 = (1 << (FL - 1));
    in2 = (1 << (FL - 2));
    in3 = (1 << (FL - 3));
    in4 = (1 << (FL - 4));
    
    #20
    
    $stop;
end

endmodule
