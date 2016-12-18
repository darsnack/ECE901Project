`timescale 1ns / 1ps
module max2_tb;

parameter WL = 16;
parameter FL = 14;

reg signed [(WL - 1):0] a, b;
wire signed [(WL - 1):0] out;

max2 #(.WL(WL), .FL(FL)) dut(
    .a(a),
    .b(b),
    .out(out)
);

reg CLK;
always #5 CLK = ~CLK;

initial begin
    CLK = 0;
    a = 0;
    b = 0;
    
    #10
    
    a = (1 << FL);
    
    #10
    
    a = ~a + 1;
    
    #10
    
    a = {1'b0, {(WL - 1){1'b1}}};
    b = (1 << (FL - 1));
    
    #10
    
    a = (1 << (FL - 1));
    b = {1'b0, {(WL - 1){1'b1}}};
    
    #10
    
    a = (1 << (WL - 1));
    b = (1 << (FL - 1));
    
    #10
    
    a = (1 << (FL - 1));
    b = (1 << (WL - 1));
    
    #10
    
    a = {1'b0, {(WL - 1){1'b1}}};
    b = {1'b0, {(WL - 1){1'b1}}};
    
    #10
    
    a = (1 << (WL - 1));
    b = (1 << (WL - 1));
    
    #10
    
    a = (1 << (WL - 1));
    b = {1'b0, {(WL - 1){1'b1}}};
    
    #10
    
    a = {1'b0, {(WL - 1){1'b1}}};
    b = (1 << (WL - 1));
    
    $stop;
end

endmodule
