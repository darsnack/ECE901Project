`timescale 1ns / 1ps
module filter3x3_tb;

parameter KERNEL_SIZE = 3;
parameter PATCH_LENGTH = KERNEL_SIZE * KERNEL_SIZE;
parameter LEARNING_RATE = 16'd15552; // 0.95 using WL = 16 and FL = 14
parameter WL = 16;
parameter FL = 14;

parameter FF_MODE = 0, FB_MODE = 1, GR_MODE = 2;

integer i;

reg CLK, RESET;
reg [1:0] mode;
reg signed [(WL - 1):0] activation_derivative;
reg signed [(WL - 1):0] in11, in12, in13, in14, in15, in16, in17, in18, in19;
reg signed [(WL - 1):0] in21, in22, in23, in24, in25, in26, in27, in28, in29;
wire signed [(WL - 1):0] out1;
wire signed [(WL - 1):0] out2;

filter3x3 dut(
    CLK, RESET, mode, activation_derivative,
    in11, in12, in13, in14, in15, in16, in17, in18, in19,
    in21, in22, in23, in24, in25, in26, in27, in28, in29,
    out1, out2
);

always #5 CLK = ~CLK;

initial begin
    RESET = 0;
    CLK = 0;
    in11 = (1 << (FL - 1));
    in12 = (1 << (FL - 1));
    in13 = (1 << (FL - 1));
    in14 = (1 << (FL - 1));
    in15 = (1 << (FL - 1));
    in16 = (1 << (FL - 1));
    in17 = (1 << (FL - 1));
    in18 = (1 << (FL - 1));
    in19 = (1 << (FL - 1));
    in21 = (1 << (FL - 1));
    in22 = (1 << (FL - 1));
    in23 = (1 << (FL - 1));
    in24 = (1 << (FL - 1));
    in25 = (1 << (FL - 1));
    in26 = (1 << (FL - 1));
    in27 = (1 << (FL - 1));
    in28 = (1 << (FL - 1));
    in29 = (1 << (FL - 1));
    mode = FF_MODE;
    activation_derivative = 0;
    
    #10
    
    RESET = 1;
    
    #20
    
    in11 = ~in11 + 1'b1;
    in12 = ~in12 + 1'b1;
    in13 = ~in13 + 1'b1;
    in14 = ~in14 + 1'b1;
    in15 = ~in15 + 1'b1;
    in16 = ~in16 + 1'b1;
    in17 = ~in17 + 1'b1;
    in18 = ~in18 + 1'b1;
    in19 = ~in19 + 1'b1;
    in21 = ~in21 + 1'b1;
    in22 = ~in22 + 1'b1;
    in23 = ~in23 + 1'b1;
    in24 = ~in24 + 1'b1;
    in25 = ~in25 + 1'b1;
    in26 = ~in26 + 1'b1;
    in27 = ~in27 + 1'b1;
    in28 = ~in28 + 1'b1;
    in29 = ~in29 + 1'b1;
    
    #20
    
    in11 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    in12 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    in13 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    in14 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    in15 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    in16 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    in17 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    in18 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    in19 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    in21 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    in22 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    in23 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    in24 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    in25 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    in26 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    in27 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    in28 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    in29 = (1 << FL) | (1 << (FL - 1)) | (1 << (FL - 2)) | (1 << (FL - 3));
    
    #20
    
    in11 = ~in11 + 1'b1;
    in12 = ~in12 + 1'b1;
    in13 = ~in13 + 1'b1;
    in14 = ~in14 + 1'b1;
    in15 = ~in15 + 1'b1;
    in16 = ~in16 + 1'b1;
    in17 = ~in17 + 1'b1;
    in18 = ~in18 + 1'b1;
    in19 = ~in19 + 1'b1;
    in21 = ~in21 + 1'b1;
    in22 = ~in22 + 1'b1;
    in23 = ~in23 + 1'b1;
    in24 = ~in24 + 1'b1;
    in25 = ~in25 + 1'b1;
    in26 = ~in26 + 1'b1;
    in27 = ~in27 + 1'b1;
    in28 = ~in28 + 1'b1;
    in29 = ~in29 + 1'b1;
    
    #20
    
    in11 = (1 << (FL - 1));
    in12 = (1 << (FL - 1));
    in13 = (1 << (FL - 1));
    in14 = (1 << (FL - 1));
    in15 = (1 << (FL - 1)) | (1 << (FL - 2));
    in16 = (1 << (FL - 1)) | (1 << (FL - 2));
    in17 = (1 << (FL - 1)) | (1 << (FL - 2));
    in18 = (1 << (FL - 1)) | (1 << (FL - 2));
    in19 = (1 << (FL - 1)) | (1 << (FL - 2));
    in21 = (1 << (FL - 1));
    in22 = (1 << (FL - 1));
    in23 = (1 << (FL - 1));
    in24 = (1 << (FL - 1));
    in25 = (1 << (FL - 1)) | (1 << (FL - 2));
    in26 = (1 << (FL - 1)) | (1 << (FL - 2));
    in27 = (1 << (FL - 1)) | (1 << (FL - 2));
    in28 = (1 << (FL - 1)) | (1 << (FL - 2));
    in29 = (1 << (FL - 1)) | (1 << (FL - 2));
    activation_derivative = (1 << (FL - 1));
    mode = FB_MODE;
    
    #20
    
    mode = GR_MODE;
    
    #20
    
    in11 = (1 << (FL - 1));
    in12 = (1 << (FL - 1));
    in13 = (1 << (FL - 1));
    in14 = (1 << (FL - 1));
    in15 = (1 << (FL - 1));
    in16 = (1 << (FL - 1));
    in17 = (1 << (FL - 1));
    in18 = (1 << (FL - 1));
    in19 = (1 << (FL - 1));
    in21 = (1 << (FL - 3));
    in22 = (1 << (FL - 3));
    in23 = (1 << (FL - 3));
    in24 = (1 << (FL - 3));
    in25 = (1 << (FL - 3));
    in26 = (1 << (FL - 3));
    in27 = (1 << (FL - 3));
    in28 = (1 << (FL - 3));
    in29 = (1 << (FL - 3));
    
    #20
    
    in11 = ~in11 + 1'b1;
    in12 = ~in12 + 1'b1;
    in13 = ~in13 + 1'b1;
    in14 = ~in14 + 1'b1;
    in15 = ~in15 + 1'b1;
    in16 = ~in16 + 1'b1;
    in17 = ~in17 + 1'b1;
    in18 = ~in18 + 1'b1;
    in19 = ~in19 + 1'b1;
    
    #20
    
    $stop;
end

endmodule
