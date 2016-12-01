`timescale 1ns / 1ps
module stochastic_quantizer_tb;

parameter WL = 16;
parameter FL = 14;

reg CLK;
reg RESET;
reg signed [(2*WL - 1):0] in;
wire signed [(WL - 1):0] out;

stochastic_quantizer #(.WL(WL), .FL(FL)) dut(
    .CLK(CLK),
    .RESET(RESET),
    .in(in),
    .out(out)
);

always #5 CLK = ~CLK;

initial begin
    RESET = 0;
    CLK = 0;
    in = (1'b1 << (2*FL)) | (1'b1 << (FL - 1));
    
    #12
    
    RESET = 1;
    
    #(WL*10)
        
    $display("Initialization period over");
    
    #(30*10)
    
    in = in | (1'b1 << (2*FL + 1));
    
    #(30*10)
    
    in = ~in + 1'b1;
    
    #(30*10)
    
    $stop;
end

endmodule
