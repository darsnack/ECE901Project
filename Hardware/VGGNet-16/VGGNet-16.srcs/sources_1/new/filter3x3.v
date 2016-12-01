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
parameter LARGEST_NUM = {1'b0, {(WL - 1){1'b1}}};
parameter SMALLEST_NUM = (1 << (WL - 1));

parameter FF_MODE = 0, FB_MODE = 1, GR_MODE = 2;
parameter STATE_RESET = 0, STATE_NORMAL = 1, STATE_UPDATE = 2;

integer i;

input CLK, RESET;
input [1:0] mode;
input signed [(WL - 1):0] activation_derivative;
input signed [(WL - 1):0] in11, in12, in13, in14, in15, in16, in17, in18, in19;
input signed [(WL - 1):0] in21, in22, in23, in24, in25, in26, in27, in28, in29;
output reg signed [(WL - 1):0] out1;
output reg signed [(WL - 1):0] out2;

reg [1:0] CurrentState;
reg [1:0] NextState;

reg signed [(WL - 1):0] weights [0:(PATCH_LENGTH - 1)];
reg signed [(WL - 1):0] bias;

wire signed [(WL - 1):0] in1 [0:(PATCH_LENGTH - 1)];
wire signed [(WL - 1):0] in2 [0:(PATCH_LENGTH - 1)];
reg signed [(2*WL - 1):0] prod [0:(PATCH_LENGTH - 1)];
wire signed [(2*WL - 1):0] prod_sum [0:(PATCH_LENGTH - 2)];
wire signed [(WL - 1):0] prod_sum_q;
wire signed [(2*WL - 1):0] activation_input;
wire signed [(WL - 1):0] activation_input_q;
wire signed [(WL - 1):0] sigma_out, sigma_prime_out;
wire signed [(2*WL - 1):0] grad_update;
wire signed [(2*WL - 1):0] prop_error;
wire signed [(WL - 1):0] prop_error_q;
wire overflow;
wire grad_update_overflow;
reg signed [(2*WL - 1):0] weight_updates [0:(PATCH_LENGTH - 1)];
wire signed [(WL - 1):0] weight_updates_q [0:(PATCH_LENGTH - 1)];

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

always @(posedge CLK) begin
	if (RESET == 1'b0) begin
	   CurrentState <= STATE_RESET;
	end
	else begin
	   CurrentState <= NextState;
	end
end

always @(CurrentState, mode) begin
    case (CurrentState)
    STATE_RESET: NextState <= STATE_NORMAL;
    STATE_NORMAL: begin
        if (mode == GR_MODE) NextState <= STATE_UPDATE;
        else NextState <= STATE_NORMAL;
    end
    STATE_UPDATE: NextState <= STATE_NORMAL;
    endcase
end

always @(CurrentState, weight_updates_q, in1) begin
    if (CurrentState == STATE_RESET) begin
        for (i = 0; i < PATCH_LENGTH; i = i + 1) begin
            weights[i] <= (1 << (FL - 3));
        end
        bias <= (1 << (FL - 3));
    end
    else if (CurrentState == STATE_UPDATE) begin
        for (i = 0; i < PATCH_LENGTH; i = i + 1) begin
            weights[i] <= weight_updates_q[i];
        end
        bias <= in1[4]; // Center of 3x3 filter
    end
    else begin
        for (i = 0; i < PATCH_LENGTH; i = i + 1) begin
            weights[i] <= weights[i];
        end
        bias <= bias;
    end
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

stochastic_quantizer #(.WL(WL), .FL(FL)) prod_sum_quantizer(
    .CLK(CLK),
    .RESET(RESET),
	.in(prod_sum[PATCH_LENGTH - 2]),
	.out(prod_sum_q),
	.overflow(overflow)
);

assign activation_input = prod_sum[PATCH_LENGTH - 2] + (bias << FL);

stochastic_quantizer #(.WL(WL), .FL(FL)) activation_quantizer(
    .CLK(CLK),
    .RESET(RESET),
	.in(activation_input),
	.out(activation_input_q),
	.overflow()
);

sigma #(.WL(WL)) activator(
	.in(activation_input_q),
	.out(sigma_out)
);

sigma_prime #(.WL(WL), .FL(FL)) derivative(
	.in(activation_input_q),
	.out(sigma_prime_out)
);

assign prop_error = prod_sum_q * activation_derivative;

stochastic_quantizer #(.WL(WL), .FL(FL)) prop_error_quantizer(
    .CLK(CLK),
    .RESET(RESET),
    .in(prop_error),
    .out(prop_error_q),
    .overflow()
);

always @(mode, prop_error_q, sigma_out, sigma_prime_out) begin
    case (mode)
    FF_MODE: begin
        out1 <= sigma_out;
        out2 <= sigma_prime_out;
    end
    FB_MODE: begin
        out1 <= prop_error_q;
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

assign grad_update = $signed(LEARNING_RATE) * prod_sum_q;

always @(grad_update) begin
    for (i = 0; i < PATCH_LENGTH; i = i + 1) begin
        weight_updates[i] <= (weights[i] << FL) - grad_update;
    end
end

stochastic_quantizer #(.WL(WL), .FL(FL)) weight0_quantizer(
    .CLK(CLK),
    .RESET(RESET),
    .in(weight_updates[0]),
    .out(weight_updates_q[0]),
    .overflow()
);
stochastic_quantizer #(.WL(WL), .FL(FL)) weight1_quantizer(
    .CLK(CLK),
    .RESET(RESET),
    .in(weight_updates[1]),
    .out(weight_updates_q[1]),
    .overflow()
);
stochastic_quantizer #(.WL(WL), .FL(FL)) weight2_quantizer(
    .CLK(CLK),
    .RESET(RESET),
    .in(weight_updates[2]),
    .out(weight_updates_q[2]),
    .overflow()
);
stochastic_quantizer #(.WL(WL), .FL(FL)) weight3_quantizer(
    .CLK(CLK),
    .RESET(RESET),
    .in(weight_updates[3]),
    .out(weight_updates_q[3]),
    .overflow()
);
stochastic_quantizer #(.WL(WL), .FL(FL)) weight4_quantizer(
    .CLK(CLK),
    .RESET(RESET),
    .in(weight_updates[4]),
    .out(weight_updates_q[4]),
    .overflow()
);
stochastic_quantizer #(.WL(WL), .FL(FL)) weight5_quantizer(
    .CLK(CLK),
    .RESET(RESET),
    .in(weight_updates[5]),
    .out(weight_updates_q[5]),
    .overflow()
);
stochastic_quantizer #(.WL(WL), .FL(FL)) weight6_quantizer(
    .CLK(CLK),
    .RESET(RESET),
    .in(weight_updates[6]),
    .out(weight_updates_q[6]),
    .overflow()
);
stochastic_quantizer #(.WL(WL), .FL(FL)) weight7_quantizer(
    .CLK(CLK),
    .RESET(RESET),
    .in(weight_updates[7]),
    .out(weight_updates_q[7]),
    .overflow()
);
stochastic_quantizer #(.WL(WL), .FL(FL)) weight8_quantizer(
    .CLK(CLK),
    .RESET(RESET),
    .in(weight_updates[8]),
    .out(weight_updates_q[8]),
    .overflow()
);

endmodule