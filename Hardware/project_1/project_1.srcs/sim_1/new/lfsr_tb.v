`timescale 1ns / 1ps
module lfsr_tb;

parameter WL = 16;

reg CLK;
reg RESET;
wire [(WL - 1):0] out;

lfsr dut(
    .CLK(CLK),
    .RESET(RESET),
    .out(out)
);

always #5 CLK = ~CLK;

initial begin
    RESET = 0;
    CLK = 0;
    
    #12
    
    RESET = 1;
    
    #(WL*10)
    
    $display("Initialization period over");
    
    #(10*10)
    
    $stop;
end

endmodule