`timescale 1ns / 1ps
module activation_tb;

parameter WL = 16;

reg [(WL - 1):0] in;
wire [(WL - 1):0] out;
wire [(WL - 1):0] out_prime;

sigma dut(.in(in), .out(out));
sigma_prime dut_prime(.in(in), .out(out_prime));

reg CLK;
always #5 CLK = ~CLK;

integer i;
initial begin
    CLK = 0;
    in = 0;
    
    #10
    
    in = 16'h0001;
    
    #10
    
    in = 16'h0010;
    
    #10
    
    in = 16'h0100;
    
    #10
    
    in = 16'h1000;
    
    #10
    
    in = 16'h8100;
    
    #10
    
    $stop;
end

endmodule
