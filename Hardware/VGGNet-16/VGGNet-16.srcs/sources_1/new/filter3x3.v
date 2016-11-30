`timescale 1ns / 1ps
// The following code is custom designed for a KERNEL_SIZE=3
module filter3x3 (CLK, RESET, mode, activation_derivative, 
    in11, in12, in13, in14, in15, in16, in17, in18, in19, 
    in21, in22, in23, in24, in25, in26, in27, in28, in29,
    out1, out2);

parameter KERNEL_SIZE = 3;
parameter PATCH_LENGTH = KERNEL_SIZE * KERNEL_SIZE;
parameter LEARNING_RATE = 16'd15552; // 0.95 using WL = 16 and FL = 14
parameter WL = 16;
parameter FL = 14;

parameter FF_MODE = 0, FB_MODE = 1, GR_MODE = 2;

integer i;

input CLK, RESET;
input [1:0] mode;
input signed [(WL - 1):0] activation_derivative;
input signed [(WL - 1):0] in11, in12, in13, in14, in15, in16, in17, in18, in19;
input signed [(WL - 1):0] in21, in22, in23, in24, in25, in26, in27, in28, in29;
output reg signed [(WL - 1):0] out1;
output reg signed [(WL - 1):0] out2;

reg signed [(WL - 1):0] weights [0:(PATCH_LENGTH - 1)];
reg signed [(WL - 1):0] bias;

wire signed [(WL - 1):0] in1 [0:(PATCH_LENGTH - 1)];
wire signed [(WL - 1):0] in2 [0:(PATCH_LENGTH - 1)];
reg signed [(2*WL - 1):0] prod [0:(PATCH_LENGTH - 1)];
wire signed [(2*WL - 1):0] prod_sum [0:(PATCH_LENGTH - 2)];
wire signed [(WL - 1):0] prod_sum_q;
wire signed [(WL - 1):0] activation_input;
wire signed [(WL - 1):0] sigma_out, sigma_prime_out;
wire signed [(2*WL - 1):0] grad_update;
wire signed [(WL - 1):0] grad_update_q;
wire overflow;

assign in1[0] = in11;
assign in1[1] = in12;
assign in1[2] = in13;
assign in1[3] = in14;
assign in1[4] = in15;
assign in1[5] = in16;
assign in1[6] = in17;
assign in1[7] = in18;
assign in1[8] = in19;
assign in2[0] = in21;
assign in2[1] = in22;
assign in2[2] = in23;
assign in2[3] = in24;
assign in2[4] = in25;
assign in2[5] = in26;
assign in2[6] = in27;
assign in2[7] = in28;
assign in2[8] = in29;

always @(negedge RESET) begin
	for (i = 0; i < PATCH_LENGTH; i = i + 1) begin
		weights[i] <= (1 << (FL - 3));
	end
	bias <= (1 << (FL - 3));
end

always @(mode, in1, in2, RESET) begin
	case (mode)
	FF_MODE: begin
		for (i = 0; i < PATCH_LENGTH; i = i + 1) begin
			prod[i] <= in1[i] * weights[i];
		end
	end
	FB_MODE: begin
		for (i = 0; i < PATCH_LENGTH; i = i + 1) begin
			prod[i] <= in1[i] * weights[PATCH_LENGTH - 1 - i];
		end
	end
	GR_MODE: begin
		for (i = 0; i < PATCH_LENGTH; i = i + 1) begin
			prod[i] <= in1[i] * in2[i];
		end
	end
	endcase
end

assign prod_sum[0] = prod[0] + prod[1];
assign prod_sum[1] = prod[2] + prod[3];
assign prod_sum[2] = prod[4] + prod[5];
assign prod_sum[3] = prod[6] + prod[7];
assign prod_sum[4] = prod_sum[0] + prod_sum[1];
assign prod_sum[5] = prod_sum[2] + prod_sum[3];
assign prod_sum[6] = prod_sum[4] + prod_sum[5];
assign prod_sum[7] = prod_sum[6] + prod[8];

stochastic_quantizer #(.WL(WL), .FL(FL)) quantizer(
    .CLK(CLK),
    .RESET(RESET),
	.in(prod_sum[PATCH_LENGTH - 2]),
	.out(prod_sum_q),
	.overflow(overflow)
);

assign activation_input = (overflow && ~(prod_sum_q[WL - 1] ^ bias[WL - 1])) ? prod_sum_q : (prod_sum_q + bias);

sigma #(.WL(WL)) activator(
	.in(activation_input),
	.out(sigma_out)
);

sigma_prime #(.WL(WL), .FL(FL)) derivative(
	.in(activation_input),
	.out(sigma_prime_out)
);

always @(mode, prod_sum_q, sigma_out, sigma_prime_out) begin
    case (mode)
    FF_MODE: begin
        out1 <= sigma_out;
        out2 <= sigma_prime_out;
    end
    FB_MODE: begin
        out1 <= prod_sum_q;
        out2 <= 0;
    end
    GR_MODE: begin
        out1 <= 0;
        out2 <= 0;
    end
    default: begin
        out1 <= 0;
        out2 <= 0;
    end
    endcase
end

assign grad_update = LEARNING_RATE * prod_sum_q;

stochastic_quantizer grad_quantizer(
    .CLK(CLK),
    .RESET(RESET),
    .in(grad_update),
    .out(grad_update_q),
    .overflow()
);

always @(mode, prod_sum_q, in1) begin
    case(mode)
    FF_MODE: begin
        for (i = 0; i < PATCH_LENGTH; i = i + 1) begin
            weights[i] <= weights[i];
        end
        bias <= bias;
    end
    FB_MODE: begin
        for (i = 0; i < PATCH_LENGTH; i = i + 1) begin
            weights[i] <= weights[i];
        end
        bias <= bias;
    end
    GR_MODE: begin
        for (i = 0; i < PATCH_LENGTH; i = i + 1) begin
            weights[i] <= weights[i] - grad_update_q;
        end
        bias <= bias - in1[4]; // Center of 3x3 filter
    end
    endcase
end

endmodule