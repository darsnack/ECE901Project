`timescale 1ns / 1ps
module top_cnn_tb;

parameter KERNEL_SIZE = 3;
parameter LEARNING_RATE = 16'd15552; // 0.95 using WL = 16 and FL = 14
parameter WL = 16;
parameter FL = 14;

parameter FF_MODE = 0, FB_MODE = 1, GR_MODE = 2;

integer i;

reg CLK, RESET, Start;
reg [(WL - 1):0] in1, in2, in3, in4, in5, in6, in7, in8, in9;
wire [(WL - 1):0] out1, out2, out3, out4, out5, out6, out7, out8, out9, out10;
wire Done;

top_cnn #(.KERNEL_SIZE(KERNEL_SIZE), .LEARNING_RATE(LEARNING_RATE), .WL(WL), .FL(FL)) dut(
    CLK, RESET, Start, Done, 
    in1, in2, in3, in4, in5, in6, in7, in8, in9, 
    out1, out2, out3, out4, out5, out6, out7, out8, out9, out10
);

always #5 CLK = ~CLK;

initial begin
    RESET = 0;
    CLK = 0;
    in1 = (1 << (FL - 2));
    in2 = (1 << (FL - 2));
    in3 = (1 << (FL - 2));
    in4 = (1 << (FL - 2));
    in5 = (1 << (FL - 2));
    in6 = (1 << (FL - 2));
    in7 = (1 << (FL - 2));
    in8 = (1 << (FL - 2));
    in9 = (1 << (FL - 2));
    
    #15
    
    RESET = 1;
    
    #10
    
    Start = 1;
    
    #((2*3 + 1 + 1)*32*32 * 10 + 20)

    #(2*16*16*10 + 10)
    
    #(3*16*4*10 + 10)
        
    #10
    
    $stop;
end

endmodule
