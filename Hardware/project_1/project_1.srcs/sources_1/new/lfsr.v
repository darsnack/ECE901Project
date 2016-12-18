`timescale 1ns / 1ps
module lfsr(CLK, RESET, out);

parameter WL = 16;

input CLK;
input RESET;
output reg [(WL - 1):0] out;

reg [(WL - 1):0] register;
wire tap;

always @(posedge CLK) begin
    if (RESET == 0) begin
        register <= 16'b1010110011100001;
        out <= 0;
    end else begin
        register <= {register[(WL - 2):0], tap};
        out <= {out[(WL - 2):0], register[WL - 1]};
    end
end

assign tap = register[WL - 1] ^ register[WL - 3] ^ register[WL - 4] ^ register[WL - 6];

endmodule
