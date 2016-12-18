`timescale 1ns / 1ps
// The following code is custom designed for a KERNEL_SIZE=3
module filter3x3 (CLK, RESET, acc, mode, map, activation_derivative, 
    in11, in12, in13, in14, in15, in16, in17, in18, in19, 
    in21, in22, in23, in24, in25, in26, in27, in28, in29,
    out1, out2);

parameter KERNEL_SIZE = 3;
parameter DEPTH = 3;
parameter PATCH_LENGTH = KERNEL_SIZE * KERNEL_SIZE;
parameter VOLUME = PATCH_LENGTH * DEPTH;
parameter LEARNING_RATE = 16'd15552; // 0.95 using WL = 16 and FL = 14
parameter WL = 16;
parameter FL = 14;
parameter LARGEST_NUM = {1'b0, {(WL - 1){1'b1}}};
parameter SMALLEST_NUM = (1 << (WL - 1));
parameter ACC_LARGE_NUM = {1'b00, {(2*WL - 2){1'b1}}};
parameter ACC_SMALL_NUM = (1 << (2*WL - 1)) | (1 << (2 * WL - 2));

parameter FF_MODE = 0, FB_MODE = 1, GR_MODE = 2;
parameter STATE_RESET = 0, STATE_WAIT = 1, STATE_DOT_PRODUCT = 2, STATE_ACCUMULATE = 3, STATE_ACTIVATE = 4, STATE_UPDATE = 5;

integer i;

input CLK, RESET;
input acc;
input [1:0] mode;
input [2:0] map; // Largest possible depth is 8 -> represented by three bits
input signed [(WL - 1):0] activation_derivative;
input signed [(WL - 1):0] in11, in12, in13, in14, in15, in16, in17, in18, in19;
input signed [(WL - 1):0] in21, in22, in23, in24, in25, in26, in27, in28, in29;
output reg signed [(WL - 1):0] out1;
output reg signed [(WL - 1):0] out2;

reg [2:0] CurrentState;
reg [2:0] NextState;

reg signed [(WL - 1):0] weights [0:(VOLUME - 1)];
reg signed [(WL - 1):0] bias;

wire signed [(WL - 1):0] in1 [0:(PATCH_LENGTH - 1)];
wire signed [(WL - 1):0] in2 [0:(PATCH_LENGTH - 1)];
reg signed [(2*WL - 1):0] prod [0:(PATCH_LENGTH - 1)];
wire signed [(2*WL - 1):0] prod_sum [0:(PATCH_LENGTH - 2)];
wire signed [(WL - 1):0] prod_sum_q;
wire signed [(2*WL - 1):0] bias_sum [0:(PATCH_LENGTH - 2)];
reg signed [(2*WL - 1):0] acc_reg;
wire signed [(WL - 1):0] acc_reg_q;
reg signed [(2*WL - 1):0] bias_acc_reg;
wire signed [(WL - 1):0] bias_acc_reg_q;
wire signed [(2*WL - 1):0] activation_input;
wire signed [(WL - 1):0] activation_input_q;
wire signed [(WL - 1):0] sigma_out, sigma_prime_out;
wire signed [(2*WL - 1):0] grad_update;
wire signed [(WL - 1):0] grad_update_q;
wire signed [(2*WL - 1):0] bias_update;
wire signed [(WL - 1):0] bias_update_q;
wire signed [(2*WL - 1):0] prop_error;
wire signed [(WL - 1):0] prop_error_q;
reg acc_overflow;
reg prev_acc_sign;
reg acc_sign;
reg prev_acc_in_sign;
reg acc_in_sign;
reg acc_overflow_dir; // 1 is negative
wire final_acc_in_sign;
wire final_acc_overflow_dir; // 1 is negative
reg bias_acc_overflow;
reg prev_bias_acc_sign;
reg bias_acc_sign;
reg prev_bias_acc_in_sign;
reg bias_acc_in_sign;
reg bias_acc_overflow_dir; // 1 is negative
wire final_bias_acc_in_sign;
wire final_bias_acc_overflow_dir; // 1 is negative
wire [(WL - 1):0] new_weights [0:(PATCH_LENGTH - 1)];
wire [(WL - 1):0] new_bias;

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

always @(CurrentState, mode, acc) begin
    case (CurrentState)
    STATE_RESET: NextState <= STATE_WAIT;
    STATE_WAIT: begin
        if (acc == 1'b1) NextState <= STATE_DOT_PRODUCT;
        else NextState <= STATE_WAIT;
    end
    STATE_DOT_PRODUCT: begin
        NextState <= STATE_ACCUMULATE;
    end
    STATE_ACCUMULATE: begin
        if (acc == 1'b1) NextState <= STATE_DOT_PRODUCT;
        else begin
            if (mode == FF_MODE) NextState <= STATE_ACTIVATE;
            else if (mode == GR_MODE) NextState <= STATE_UPDATE;
            else NextState <= STATE_WAIT;
        end
    end
    STATE_ACTIVATE: NextState <= STATE_WAIT;
    STATE_UPDATE: NextState <= STATE_WAIT;
    endcase
end

always @(CurrentState, prod_sum[PATCH_LENGTH - 2]) begin
    if (CurrentState == STATE_DOT_PRODUCT) begin
        acc_reg <= acc_reg;
        acc_sign <= acc_reg[2*WL - 1] | acc_reg[2*WL - 2];
        acc_in_sign <= prod_sum[PATCH_LENGTH - 2][2*WL - 1] | prod_sum[PATCH_LENGTH - 2][2*WL - 2];
        prev_acc_sign <= prev_acc_sign;
        prev_acc_in_sign <= prev_acc_in_sign;
        acc_overflow <= ~(prev_acc_sign ^ prev_acc_in_sign) & (prev_acc_sign ^ (acc_reg[2*WL - 1] | acc_reg[2*WL - 2]));
        acc_overflow_dir <= prev_acc_sign;
    end
    else if (CurrentState == STATE_ACCUMULATE) begin
        acc_reg <= acc_overflow ? (acc_overflow_dir ? ACC_SMALL_NUM : ACC_LARGE_NUM) : (acc_reg + prod_sum[PATCH_LENGTH - 2]);
        acc_sign <= acc_sign;
        acc_in_sign <= acc_in_sign;
        prev_acc_sign <= acc_sign;
        prev_acc_in_sign <= acc_in_sign;
        acc_overflow <= acc_overflow;
        acc_overflow_dir <= acc_overflow_dir;
    end
    else if (CurrentState == STATE_WAIT) begin
        acc_reg <= 0;
        acc_sign <= 0;
        acc_in_sign <= 0;
        prev_acc_sign <= 0;
        prev_acc_in_sign <= 0;
        acc_overflow <= 0;
        acc_overflow_dir <= 0;
    end
    else begin
        acc_reg <= acc_reg;
        acc_sign <= acc_sign;
        acc_in_sign <= acc_in_sign;
        prev_acc_sign <= prev_acc_sign;
        prev_acc_in_sign <= prev_acc_in_sign;
        acc_overflow <= acc_overflow;
        acc_overflow_dir <= acc_overflow_dir;
    end
end

always @(CurrentState, bias_sum[PATCH_LENGTH - 2]) begin
    if (CurrentState == STATE_DOT_PRODUCT) begin
        bias_acc_reg <= bias_acc_reg;
        bias_acc_sign <= bias_acc_reg[2*WL - 1] | bias_acc_reg[2*WL - 2];
        bias_acc_in_sign <= bias_sum[PATCH_LENGTH - 2][2*WL - 1] | bias_sum[PATCH_LENGTH - 2][2*WL - 2];
        prev_bias_acc_sign <= prev_bias_acc_sign;
        prev_bias_acc_in_sign <= prev_bias_acc_in_sign;
        bias_acc_overflow <= ~(prev_bias_acc_sign ^ prev_bias_acc_in_sign) & (prev_bias_acc_sign ^ (bias_acc_reg[2*WL - 1] | bias_acc_reg[2*WL - 2]));
        bias_acc_overflow_dir <= prev_bias_acc_sign;
    end
    else if (CurrentState == STATE_ACCUMULATE) begin
        bias_acc_reg <= bias_acc_overflow ? (bias_acc_overflow_dir ? ACC_SMALL_NUM : ACC_LARGE_NUM) : (bias_acc_reg + bias_sum[PATCH_LENGTH - 2]);
        bias_acc_sign <= bias_acc_sign;
        bias_acc_in_sign <= bias_acc_in_sign;
        prev_bias_acc_sign <= bias_acc_sign;
        prev_bias_acc_in_sign <= bias_acc_in_sign;
        bias_acc_overflow <= bias_acc_overflow;
        bias_acc_overflow_dir <= bias_acc_overflow_dir;
    end
    else if (CurrentState == STATE_WAIT) begin
        bias_acc_reg <= 0;
        bias_acc_sign <= 0;
        bias_acc_in_sign <= 0;
        prev_bias_acc_sign <= 0;
        prev_bias_acc_in_sign <= 0;
        bias_acc_overflow <= 0;
        bias_acc_overflow_dir <= 0;
    end
    else begin
        bias_acc_reg <= bias_acc_reg;
        bias_acc_sign <= bias_acc_sign;
        bias_acc_in_sign <= bias_acc_in_sign;
        prev_bias_acc_sign <= prev_bias_acc_sign;
        prev_bias_acc_in_sign <= prev_bias_acc_in_sign;
        bias_acc_overflow <= bias_acc_overflow;
        bias_acc_overflow_dir <= bias_acc_overflow_dir;
    end
end

stochastic_quantizer #(.WL(WL), .FL(FL)) acc_reg_quantizer(
    .CLK(CLK),
    .RESET(RESET),
    .in(acc_reg),
    .out(acc_reg_q),
    .overflow()
);

stochastic_quantizer #(.WL(WL), .FL(FL)) bias_acc_reg_quantizer(
    .CLK(CLK),
    .RESET(RESET),
    .in(bias_acc_reg),
    .out(bias_acc_reg_q),
    .overflow()
);

always @(CurrentState, grad_update_q, bias_update_q) begin
    if (CurrentState == STATE_RESET) begin
        for (i = 0; i < VOLUME; i = i + 1) begin
            weights[i] <= (1 << (FL - 3));
        end
        bias <= (1 << (FL - 3));
    end
    else if (CurrentState == STATE_UPDATE) begin
        for (i = 0; i < PATCH_LENGTH; i = i + 1) begin
            weights[i + map * PATCH_LENGTH] <= new_weights[i];
        end
        bias <= new_bias;
    end
    else begin
        for (i = 0; i < VOLUME; i = i + 1) begin
            weights[i] <= weights[i];
        end
        bias <= bias;
    end
end

always @(mode, in1, in2, weights, map) begin
    case (mode)
    FF_MODE: begin
        for (i = 0; i < PATCH_LENGTH; i = i + 1) begin
            prod[i] <= in1[i] * weights[i + (map * PATCH_LENGTH)];
        end
    end
    FB_MODE: begin
        for (i = 0; i < PATCH_LENGTH; i = i + 1) begin
            prod[i] <= in1[i] * weights[(map * PATCH_LENGTH) + PATCH_LENGTH - 1 - i];
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

assign bias_sum[0] = (in2[0] << FL) + (in2[1] << FL);
assign bias_sum[1] = (in2[2] << FL) + (in2[3] << FL);
assign bias_sum[2] = (in2[4] << FL) + (in2[5] << FL);
assign bias_sum[3] = (in2[6] << FL) + (in2[7] << FL);
assign bias_sum[4] = bias_sum[0] + bias_sum[1];
assign bias_sum[5] = bias_sum[2] + bias_sum[3];
assign bias_sum[6] = bias_sum[4] + bias_sum[5];
assign bias_sum[7] = bias_sum[6] + (in2[8] << FL);

stochastic_quantizer #(.WL(WL), .FL(FL)) prod_sum_quantizer(
    .CLK(CLK),
    .RESET(RESET),
    .in(prod_sum[PATCH_LENGTH - 2]),
    .out(prod_sum_q),
    .overflow()
);

assign activation_input = acc_reg + (bias << FL);

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


assign final_acc_overflow = ~(prev_acc_sign ^ prev_acc_in_sign) & (prev_acc_sign ^ (acc_reg[2*WL - 1] | acc_reg[2*WL - 2]));
assign final_acc_overflow_dir = prev_acc_sign;
assign final_bias_acc_overflow = ~(prev_bias_acc_sign ^ prev_bias_acc_in_sign) & (prev_bias_acc_sign ^ (bias_acc_reg[2*WL - 1] | bias_acc_reg[2*WL - 2]));
assign final_bias_acc_overflow_dir = prev_bias_acc_sign;
assign grad_update = $signed(LEARNING_RATE) * (final_acc_overflow ? (final_acc_overflow_dir ? SMALLEST_NUM : LARGEST_NUM) : acc_reg_q);
assign bias_update = $signed(LEARNING_RATE) * (final_bias_acc_overflow ? (final_bias_acc_overflow_dir ? SMALLEST_NUM : LARGEST_NUM) : bias_acc_reg_q);

stochastic_quantizer #(.WL(WL), .FL(FL)) grad_update_quantizer(
    .CLK(CLK),
    .RESET(RESET),
    .in(grad_update),
    .out(grad_update_q),
    .overflow()
);

stochastic_quantizer #(.WL(WL), .FL(FL)) bias_update_quantizer(
    .CLK(CLK),
    .RESET(RESET),
    .in(bias_update),
    .out(bias_update_q),
    .overflow()
);

sub_with_saturate #(.WL(WL), .FL(FL)) weight_subtractor0(
    .a(weights[0 + map * PATCH_LENGTH]),
    .b(grad_update_q),
    .out(new_weights[0])
);
sub_with_saturate #(.WL(WL), .FL(FL)) weight_subtractor1(
    .a(weights[1 + map * PATCH_LENGTH]),
    .b(grad_update_q),
    .out(new_weights[1])
);
sub_with_saturate #(.WL(WL), .FL(FL)) weight_subtractor2(
    .a(weights[2 + map * PATCH_LENGTH]),
    .b(grad_update_q),
    .out(new_weights[2])
);
sub_with_saturate #(.WL(WL), .FL(FL)) weight_subtractor3(
    .a(weights[3 + map * PATCH_LENGTH]),
    .b(grad_update_q),
    .out(new_weights[3])
);
sub_with_saturate #(.WL(WL), .FL(FL)) weight_subtractor4(
    .a(weights[4 + map * PATCH_LENGTH]),
    .b(grad_update_q),
    .out(new_weights[4])
);
sub_with_saturate #(.WL(WL), .FL(FL)) weight_subtractor5(
    .a(weights[5 + map * PATCH_LENGTH]),
    .b(grad_update_q),
    .out(new_weights[5])
);
sub_with_saturate #(.WL(WL), .FL(FL)) weight_subtractor6(
    .a(weights[6 + map * PATCH_LENGTH]),
    .b(grad_update_q),
    .out(new_weights[6])
);
sub_with_saturate #(.WL(WL), .FL(FL)) weight_subtractor7(
    .a(weights[7 + map * PATCH_LENGTH]),
    .b(grad_update_q),
    .out(new_weights[7])
);
sub_with_saturate #(.WL(WL), .FL(FL)) weight_subtractor8(
    .a(weights[8 + map * PATCH_LENGTH]),
    .b(grad_update_q),
    .out(new_weights[8])
);
sub_with_saturate #(.WL(WL), .FL(FL)) bias_subtractor(
    .a(bias),
    .b(bias_update_q),
    .out(new_bias)
);

endmodule