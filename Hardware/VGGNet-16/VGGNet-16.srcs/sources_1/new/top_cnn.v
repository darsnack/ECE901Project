`timescale 1ns / 1ps
module top_cnn(CLK, RESET, Start, Done, 
    in1, in2, in3, in4, in5, in6, in7, in8, in9, 
    label1, label2, label3, label4, label5, label6, label7, label8, label9, label10,
    out1, out2, out3, out4, out5, out6, out7, out8, out9, out10);

integer i;
integer j;

parameter WL = 16;
parameter FL = 14;
parameter KERNEL_SIZE = 3;
parameter PATCH_LENGTH = KERNEL_SIZE * KERNEL_SIZE;
parameter POOL_LENGTH = 4;
parameter LEARNING_RATE = 16'd15552; // 0.95 using WL = 16 and FL = 14
parameter MAP_BITS = 4;
parameter DIM_BITS = 10;

parameter INPUT_DEPTH = 3;

parameter LAYER_1_DEPTH = 4;
parameter LAYER_1_WIDTH = 32;
parameter LAYER_1_HEIGHT = 32;
parameter LAYER_1_VOLUME = LAYER_1_WIDTH * LAYER_1_HEIGHT * LAYER_1_DEPTH;

parameter LAYER_2_DEPTH = 4;
parameter LAYER_2_WIDTH = 16;
parameter LAYER_2_HEIGHT = 16;
parameter LAYER_2_VOLUME = LAYER_2_WIDTH * LAYER_2_HEIGHT * LAYER_2_DEPTH;

parameter LAYER_3_DEPTH = 10;
parameter LAYER_3_WIDTH = 1;
parameter LAYER_3_HEIGHT = 1;
parameter LAYER_3_VOLUME = LAYER_3_WIDTH * LAYER_3_HEIGHT * LAYER_3_DEPTH;

parameter FF_MODE = 0, FB_MODE = 1, GR_MODE = 2;

parameter STATE_RESET = 0, STATE_WAIT = 1;
parameter STATE_L1_START = 2, STATE_L1_DATA = 3, STATE_L1_COMPUTE = 4, STATE_L1_OUTPUT = 5;
parameter STATE_L2_DATA = 6, STATE_L2_OUTPUT = 7;
parameter STATE_L3_DATA = 8, STATE_L3_ACC = 9, STATE_L3_OUTPUT = 10;
parameter STATE_ERROR = 11;
parameter STATE_L3_ERR_DATA = 12, STATE_L3_ERR_ACC = 13, STATE_L3_ERR_INCR = 14;
parameter STATE_L3_GR_DATA = 15, STATE_L3_GR_UPDATE = 16;
parameter STATE_FINAL_OUTPUT = 17;
 
input CLK, RESET, Start;
input signed [(WL - 1):0] in1, in2, in3, in4, in5, in6, in7, in8, in9;
input signed [(WL - 1):0] label1, label2, label3, label4, label5, label6, label7, label8, label9, label10;
output signed [(WL - 1):0] out1, out2, out3, out4, out5, out6, out7, out8, out9, out10;
output reg Done;

reg [4:0] CurrentState, NextState;

reg signed [(WL - 1):0] in1_layer_1 [0:(PATCH_LENGTH * LAYER_1_DEPTH - 1)];
reg signed [(WL - 1):0] in2_layer_1 [0:(PATCH_LENGTH * LAYER_1_DEPTH - 1)];
reg signed [(WL - 1):0] act_der_layer_1 [0:(LAYER_1_DEPTH - 1)];
wire signed [(WL - 1):0] out1_layer_1 [0:(LAYER_1_DEPTH - 1)];
wire signed [(WL - 1):0] out2_layer_1 [0:(LAYER_1_DEPTH - 1)];
reg signed [(WL - 1):0] out_layer_1 [0:(LAYER_1_VOLUME - 1)];
reg signed [(WL - 1):0] der_layer_1 [0:(LAYER_1_VOLUME - 1)];
reg [(MAP_BITS - 1):0] layer_1_depth_counter;
reg [(MAP_BITS - 1):0] layer_1_map;
reg [(MAP_BITS - 1):0] layer_1_next_map;
reg [(DIM_BITS - 1):0] layer_1_index;
reg [(DIM_BITS - 1):0] layer_1_next_index;
reg [1:0] layer_1_mode;
reg layer_1_acc;

reg signed [(WL - 1):0] in_layer_2 [0:(POOL_LENGTH * LAYER_2_DEPTH - 1)];
wire signed [(WL - 1):0] out1_layer_2 [0:(LAYER_2_DEPTH - 1)];
wire signed [(WL - 1):0] out2_layer_2 [0:(LAYER_2_DEPTH - 1)];
reg signed [(WL - 1):0] out_layer_2 [0:(LAYER_2_VOLUME - 1)];
reg signed [(WL - 1):0] der_layer_2 [0:(LAYER_2_VOLUME - 1)];
reg [(DIM_BITS - 1):0] layer_2_row, layer_2_col;
reg [(DIM_BITS - 1):0] layer_2_next_row, layer_2_next_col;
reg [(DIM_BITS - 1):0] layer_1_corner;

reg signed [(WL - 1):0] in1_layer_3 [0:(LAYER_2_WIDTH * LAYER_3_DEPTH - 1)];
reg signed [(WL - 1):0] in2_layer_3 [0:(LAYER_2_WIDTH * LAYER_3_DEPTH - 1)];
wire signed [(WL - 1):0] out1_layer_3 [0:(LAYER_3_DEPTH - 1)];
reg signed [(WL - 1):0] acc_in1_layer_3 [0:(LAYER_3_DEPTH - 1)];
reg signed [(WL - 1):0] acc_in2_layer_3 [0:(LAYER_3_DEPTH - 1)];
wire signed [(WL - 1):0] acc_out_layer_3 [0:(LAYER_3_DEPTH - 1)];
reg signed [(WL - 1):0] out_layer_3 [0:(LAYER_3_VOLUME - 1)];
wire signed [(WL - 1):0] der_layer_3 [0:(LAYER_3_VOLUME - 1)];
reg signed [(WL - 1):0] layer_3_weights [0:(LAYER_2_VOLUME * LAYER_3_DEPTH - 1)];
reg [(DIM_BITS - 1):0] layer_3_index;
reg [(DIM_BITS - 1):0] layer_3_next_index;

wire signed [(WL - 1):0] classification_error [0:(LAYER_3_DEPTH - 1)];
wire signed [(WL - 1):0] output_error_q [0:(LAYER_3_DEPTH - 1)];
reg signed [(WL - 1):0] output_error [0:(LAYER_3_DEPTH - 1)];

reg signed [(WL - 1):0] layer_3_der_patch [0:(LAYER_2_WIDTH - 1)];
reg signed [(WL - 1):0] layer_3_err_weight_patch [0:(LAYER_2_WIDTH - 1)];
wire signed [(WL - 1):0] layer_3_error_patch [0:(LAYER_2_WIDTH - 1)];
reg signed [(WL - 1):0] layer_3_acc [0:(LAYER_2_WIDTH - 1)];
wire signed [(WL - 1):0] layer_3_new_acc [0:(LAYER_2_WIDTH - 1)];
reg signed [(WL - 1):0] layer_3_error [0:(LAYER_2_VOLUME - 1)];
reg [(DIM_BITS - 1):0] layer_3_err_index;
reg [(DIM_BITS - 1):0] layer_3_err_next_index;
reg [(MAP_BITS - 1):0] layer_3_err_depth_counter;
reg [(MAP_BITS - 1):0] layer_3_err_next_depth_counter;

reg signed [(WL - 1):0] layer_3_gr_in_patch [0:(LAYER_2_WIDTH - 1)];
reg signed [(WL - 1):0] layer_3_weight_patch [0:(LAYER_2_WIDTH * LAYER_3_DEPTH - 1)];
wire signed [(WL - 1):0] layer_3_gr_update_patch [0:(LAYER_2_WIDTH * LAYER_3_DEPTH - 1)];
reg [(DIM_BITS - 1):0] layer_3_gr_index;
reg [(DIM_BITS - 1):0] layer_3_gr_next_index;

always @(posedge CLK) begin
    if (RESET == 1'b0) CurrentState <= STATE_RESET;
    else CurrentState <= NextState;
end

always @(CurrentState, Start) begin
    case(CurrentState)
    STATE_RESET: NextState <= STATE_WAIT;
    STATE_WAIT: begin
        if (Start == 1'b1) NextState <= STATE_L1_START;
        else NextState <= STATE_WAIT;
    end
    STATE_L1_START: NextState <= STATE_L1_DATA;
    STATE_L1_DATA: NextState <= STATE_L1_COMPUTE;
    STATE_L1_COMPUTE: begin
        if (layer_1_depth_counter < INPUT_DEPTH) NextState <= STATE_L1_DATA;
        else NextState <= STATE_L1_OUTPUT;
    end
    STATE_L1_OUTPUT: begin
        if (layer_1_index < (LAYER_1_WIDTH * LAYER_1_HEIGHT - 1)) NextState <= STATE_L1_START;
        else NextState <= STATE_L2_DATA;
    end
    STATE_L2_DATA: NextState <= STATE_L2_OUTPUT;
    STATE_L2_OUTPUT: begin
        if ((layer_2_row < LAYER_2_HEIGHT) && (layer_2_col < LAYER_2_WIDTH)) NextState <= STATE_L2_DATA;
        else NextState <= STATE_L3_DATA;
    end
    STATE_L3_DATA: NextState <= STATE_L3_ACC;
    STATE_L3_ACC: NextState <= STATE_L3_OUTPUT;
    STATE_L3_OUTPUT: begin
        if (layer_3_index < (LAYER_2_HEIGHT * LAYER_2_DEPTH - 1)) NextState <= STATE_L3_DATA;
        else NextState <= STATE_ERROR;
    end
    STATE_ERROR: NextState <= STATE_L3_ERR_DATA;
    STATE_L3_ERR_DATA: NextState <= STATE_L3_ERR_ACC;
    STATE_L3_ERR_ACC: NextState <= STATE_L3_ERR_INCR;
    STATE_L3_ERR_INCR: begin
        if ((layer_3_err_depth_counter >= (LAYER_3_DEPTH - 1)) && (layer_3_err_index >= (LAYER_2_HEIGHT * LAYER_2_DEPTH - 1)))
            NextState <= STATE_L3_GR_DATA;
        else NextState <= STATE_L3_ERR_DATA;
    end
    STATE_L3_GR_DATA: NextState <= STATE_L3_GR_UPDATE;
    STATE_L3_GR_UPDATE: begin
        if (layer_3_gr_index < (LAYER_2_HEIGHT * LAYER_2_DEPTH - 1)) NextState <= STATE_L3_GR_DATA;
        else NextState <= STATE_FINAL_OUTPUT;
    end
    STATE_FINAL_OUTPUT: begin
        if (Start == 1'b0) NextState <= STATE_WAIT;
        else NextState <= STATE_FINAL_OUTPUT;
    end
    default: NextState <= STATE_RESET;
    endcase
end

always @(CurrentState) begin
    case(CurrentState)
    STATE_RESET: begin
        for (i = 0; i < (PATCH_LENGTH * LAYER_1_DEPTH); i = i + 1) begin
            in1_layer_1[i] <= 0;
            in2_layer_1[i] <= 0;
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= 0;
        end

        for (i = 0; i < LAYER_1_VOLUME; i = i + 1) begin
            out_layer_1[i] <= 0;
            der_layer_1[i] <= 0;
        end

        layer_1_depth_counter <= 0;
        layer_1_map <= 0;
        layer_1_next_map <= 0;
        layer_1_mode <= 0;
        layer_1_acc <= 0;
        layer_1_index <= 0;
        layer_1_next_index <= 0;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            for (j = 0; j < POOL_LENGTH; j = j + 1) begin
                in_layer_2[i * POOL_LENGTH + j] <= 0;
            end
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            out_layer_2[i] <= 0;
            der_layer_2[i] <= 0;
        end

        layer_1_corner <= 0;
        layer_2_row <= 0;
        layer_2_col <= 0;
        layer_2_next_row <= 0;
        layer_2_next_col <= 0;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= 0;
                in2_layer_3[i * LAYER_2_WIDTH + j] <= 0;
            end

            acc_in1_layer_3[i] <= 0;
            acc_in2_layer_3[i] <= 0;

            out_layer_3[i] <= 0;
        end

        layer_3_index <= 0;
        layer_3_next_index <= 0;

        for (i = 0; i < (LAYER_2_VOLUME * LAYER_3_DEPTH); i = i + 1) begin
            layer_3_weights[i] <= (1 << (FL - 4));
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= 0;
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= 0;
            layer_3_err_weight_patch[i] <= 0;
            layer_3_acc[i] <= 0;
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            layer_3_error[i] <= 0;
        end

        layer_3_err_depth_counter <= 0;
        layer_3_err_next_depth_counter <= 0;
        layer_3_err_index <= 0;
        layer_3_err_next_index <= 0;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= 0;
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= 0;
            end
        end

        layer_3_gr_index <= 0;
        layer_3_gr_next_index <= 0;
        
        Done <= 0;
    end
    STATE_WAIT: begin
        for (i = 0; i < (PATCH_LENGTH * LAYER_1_DEPTH); i = i + 1) begin
            in1_layer_1[i] <= in1_layer_1[i];
            in2_layer_1[i] <= in2_layer_1[i];
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= act_der_layer_1[i];
        end

        for (i = 0; i < LAYER_1_VOLUME; i = i + 1) begin
            out_layer_1[i] <= out_layer_1[i];
            der_layer_1[i] <= der_layer_1[i];
        end

        layer_1_depth_counter <= 0;
        layer_1_map <= 0;
        layer_1_next_map <= 0;
        layer_1_mode <= 0;
        layer_1_acc <= 0;
        layer_1_index <= 0;
        layer_1_next_index <= 0;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            for (j = 0; j < POOL_LENGTH; j = j + 1) begin
                in_layer_2[i * POOL_LENGTH + j] <= in_layer_2[i * POOL_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            out_layer_2[i] <= out_layer_2[i];
            der_layer_2[i] <= der_layer_2[i];
        end

        layer_1_corner <= 0;
        layer_2_row <= 0;
        layer_2_col <= 0;
        layer_2_next_row <= 0;
        layer_2_next_col <= 0;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= in1_layer_3[i * LAYER_2_WIDTH + j];
                in2_layer_3[i * LAYER_2_WIDTH + j] <= in2_layer_3[i * LAYER_2_WIDTH + j];
            end

            acc_in1_layer_3[i] <= 0;
            acc_in2_layer_3[i] <= 0;

            out_layer_3[i] <= 0;
        end

        layer_3_index <= 0;
        layer_3_next_index <= 0;

        for (i = 0; i < (LAYER_2_VOLUME * LAYER_3_DEPTH); i = i + 1) begin
            layer_3_weights[i] <= layer_3_weights[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= output_error[i];
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= layer_3_der_patch[i];
            layer_3_err_weight_patch[i] <= layer_3_err_weight_patch[i];
            layer_3_acc[i] <= layer_3_acc[i];
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            layer_3_error[i] <= layer_3_error[i];
        end

        layer_3_err_depth_counter <= 0;
        layer_3_err_next_depth_counter <= 0;
        layer_3_err_index <= 0;
        layer_3_err_next_index <= 0;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= layer_3_gr_in_patch[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= layer_3_weight_patch[i * LAYER_2_WIDTH + j];
            end
        end

        layer_3_gr_index <= 0;
        layer_3_gr_next_index <= 0;
        
        Done <= 0;
    end
    STATE_L1_START: begin
        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            in1_layer_1[i * PATCH_LENGTH + 0] <= in1;
            in1_layer_1[i * PATCH_LENGTH + 1] <= in2;
            in1_layer_1[i * PATCH_LENGTH + 2] <= in3;
            in1_layer_1[i * PATCH_LENGTH + 3] <= in4;
            in1_layer_1[i * PATCH_LENGTH + 4] <= in5;
            in1_layer_1[i * PATCH_LENGTH + 5] <= in6;
            in1_layer_1[i * PATCH_LENGTH + 6] <= in7;
            in1_layer_1[i * PATCH_LENGTH + 7] <= in8;
            in1_layer_1[i * PATCH_LENGTH + 8] <= in9;
            in2_layer_1[i * PATCH_LENGTH + 0] <= in2_layer_1[i * PATCH_LENGTH + 0];
            in2_layer_1[i * PATCH_LENGTH + 1] <= in2_layer_1[i * PATCH_LENGTH + 1];
            in2_layer_1[i * PATCH_LENGTH + 2] <= in2_layer_1[i * PATCH_LENGTH + 2];
            in2_layer_1[i * PATCH_LENGTH + 3] <= in2_layer_1[i * PATCH_LENGTH + 3];
            in2_layer_1[i * PATCH_LENGTH + 4] <= in2_layer_1[i * PATCH_LENGTH + 4];
            in2_layer_1[i * PATCH_LENGTH + 5] <= in2_layer_1[i * PATCH_LENGTH + 5];
            in2_layer_1[i * PATCH_LENGTH + 6] <= in2_layer_1[i * PATCH_LENGTH + 6];
            in2_layer_1[i * PATCH_LENGTH + 7] <= in2_layer_1[i * PATCH_LENGTH + 7];
            in2_layer_1[i * PATCH_LENGTH + 8] <= in2_layer_1[i * PATCH_LENGTH + 8];
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= act_der_layer_1[i];
        end

        for (i = 0; i < LAYER_1_VOLUME; i = i + 1) begin
            out_layer_1[i] <= out_layer_1[i];
            der_layer_1[i] <= der_layer_1[i];
        end

        layer_1_depth_counter <= 0;
        layer_1_map <= 0;
        layer_1_next_map <= 0;
        layer_1_mode <= FF_MODE;
        layer_1_acc <= 1;
        layer_1_index <= layer_1_next_index;
        layer_1_next_index <= layer_1_next_index;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            for (j = 0; j < POOL_LENGTH; j = j + 1) begin
                in_layer_2[i * POOL_LENGTH + j] <= in_layer_2[i * POOL_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            out_layer_2[i] <= out_layer_2[i];
            der_layer_2[i] <= der_layer_2[i];
        end

        layer_1_corner <= layer_1_corner;
        layer_2_row <= layer_2_row;
        layer_2_col <= layer_2_col;
        layer_2_next_row <= layer_2_next_row;
        layer_2_next_col <= layer_2_next_col;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= 0;
                in2_layer_3[i * LAYER_2_WIDTH + j] <= 0;
            end

            acc_in1_layer_3[i] <= 0;
            acc_in2_layer_3[i] <= 0;

            out_layer_3[i] <= 0;
        end

        layer_3_index <= layer_3_index;
        layer_3_next_index <= layer_3_next_index;

        for (i = 0; i < (LAYER_2_VOLUME * LAYER_3_DEPTH); i = i + 1) begin
            layer_3_weights[i] <= layer_3_weights[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= output_error[i];
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= layer_3_der_patch[i];
            layer_3_err_weight_patch[i] <= layer_3_err_weight_patch[i];
            layer_3_acc[i] <= layer_3_acc[i];
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            layer_3_error[i] <= layer_3_error[i];
        end

        layer_3_err_depth_counter <= layer_3_err_depth_counter;
        layer_3_err_next_depth_counter <= layer_3_err_next_depth_counter;
        layer_3_err_index <= layer_3_err_index;
        layer_3_err_next_index <= layer_3_err_next_index;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= layer_3_gr_in_patch[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= layer_3_weight_patch[i * LAYER_2_WIDTH + j];
            end
        end

        layer_3_gr_index <= layer_3_gr_next_index;
        layer_3_gr_next_index <= layer_3_gr_next_index;
        
        Done <= 0;
    end
    STATE_L1_DATA: begin
        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            in1_layer_1[i * PATCH_LENGTH + 0] <= in1;
            in1_layer_1[i * PATCH_LENGTH + 1] <= in2;
            in1_layer_1[i * PATCH_LENGTH + 2] <= in3;
            in1_layer_1[i * PATCH_LENGTH + 3] <= in4;
            in1_layer_1[i * PATCH_LENGTH + 4] <= in5;
            in1_layer_1[i * PATCH_LENGTH + 5] <= in6;
            in1_layer_1[i * PATCH_LENGTH + 6] <= in7;
            in1_layer_1[i * PATCH_LENGTH + 7] <= in8;
            in1_layer_1[i * PATCH_LENGTH + 8] <= in9;
            in2_layer_1[i * PATCH_LENGTH + 0] <= in2_layer_1[i * PATCH_LENGTH + 0];
            in2_layer_1[i * PATCH_LENGTH + 1] <= in2_layer_1[i * PATCH_LENGTH + 1];
            in2_layer_1[i * PATCH_LENGTH + 2] <= in2_layer_1[i * PATCH_LENGTH + 2];
            in2_layer_1[i * PATCH_LENGTH + 3] <= in2_layer_1[i * PATCH_LENGTH + 3];
            in2_layer_1[i * PATCH_LENGTH + 4] <= in2_layer_1[i * PATCH_LENGTH + 4];
            in2_layer_1[i * PATCH_LENGTH + 5] <= in2_layer_1[i * PATCH_LENGTH + 5];
            in2_layer_1[i * PATCH_LENGTH + 6] <= in2_layer_1[i * PATCH_LENGTH + 6];
            in2_layer_1[i * PATCH_LENGTH + 7] <= in2_layer_1[i * PATCH_LENGTH + 7];
            in2_layer_1[i * PATCH_LENGTH + 8] <= in2_layer_1[i * PATCH_LENGTH + 8];
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= act_der_layer_1[i];
        end

        for (i = 0; i < LAYER_1_VOLUME; i = i + 1) begin
            out_layer_1[i] <= out_layer_1[i];
            der_layer_1[i] <= der_layer_1[i];
        end

        layer_1_depth_counter <= layer_1_depth_counter + 1;
        layer_1_map <= layer_1_next_map;
        layer_1_mode <= FF_MODE;
        layer_1_acc <= 1;
        layer_1_index <= layer_1_index;
        layer_1_next_index <= layer_1_next_index;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            for (j = 0; j < POOL_LENGTH; j = j + 1) begin
                in_layer_2[i * POOL_LENGTH + j] <= in_layer_2[i * POOL_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            out_layer_2[i] <= out_layer_2[i];
            der_layer_2[i] <= der_layer_2[i];
        end

        layer_1_corner <= layer_1_corner;
        layer_2_row <= layer_2_row;
        layer_2_col <= layer_2_col;
        layer_2_next_row <= layer_2_next_row;
        layer_2_next_col <= layer_2_next_col;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= in1_layer_3[i * LAYER_2_WIDTH + j];
                in2_layer_3[i * LAYER_2_WIDTH + j] <= in2_layer_3[i * LAYER_2_WIDTH + j];
            end

            acc_in1_layer_3[i] <= acc_in1_layer_3[i];
            acc_in2_layer_3[i] <= acc_in2_layer_3[i];

            out_layer_3[i] <= out_layer_3[i];
        end

        layer_3_index <= layer_3_index;
        layer_3_next_index <= layer_3_next_index;

        for (i = 0; i < (LAYER_2_VOLUME * LAYER_3_DEPTH); i = i + 1) begin
            layer_3_weights[i] <= layer_3_weights[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= output_error[i];
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= layer_3_der_patch[i];
            layer_3_err_weight_patch[i] <= layer_3_err_weight_patch[i];
            layer_3_acc[i] <= layer_3_acc[i];
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            layer_3_error[i] <= layer_3_error[i];
        end

        layer_3_err_depth_counter <= layer_3_err_depth_counter;
        layer_3_err_next_depth_counter <= layer_3_err_next_depth_counter;
        layer_3_err_index <= layer_3_err_index;
        layer_3_err_next_index <= layer_3_err_next_index;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= layer_3_gr_in_patch[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= layer_3_weight_patch[i * LAYER_2_WIDTH + j];
            end
        end

        layer_3_gr_index <= layer_3_gr_next_index;
        layer_3_gr_next_index <= layer_3_gr_next_index;
        
        Done <= 0;
    end
    STATE_L1_COMPUTE: begin
        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            for (j = 0; j < PATCH_LENGTH; j = j + 1) begin
                in1_layer_1[i * PATCH_LENGTH + j] <= in1_layer_1[i * PATCH_LENGTH + j];
                in2_layer_1[i * PATCH_LENGTH + j] <= in2_layer_1[i * PATCH_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= act_der_layer_1[i];
        end

        for (i = 0; i < LAYER_1_VOLUME; i = i + 1) begin
            out_layer_1[i] <= out_layer_1[i];
            der_layer_1[i] <= der_layer_1[i];
        end

        layer_1_depth_counter <= layer_1_depth_counter;
        layer_1_map <= layer_1_map;
        layer_1_next_map <= layer_1_map + 1;
        layer_1_mode <= FF_MODE;
        layer_1_acc <= (layer_1_depth_counter < (LAYER_1_DEPTH - 1)) ? 1 : 0;
        layer_1_index <= layer_1_index;
        layer_1_next_index <= layer_1_next_index;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            for (j = 0; j < POOL_LENGTH; j = j + 1) begin
                in_layer_2[i * POOL_LENGTH + j] <= in_layer_2[i * POOL_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            out_layer_2[i] <= out_layer_2[i];
            der_layer_2[i] <= der_layer_2[i];
        end

        layer_1_corner <= layer_1_corner;
        layer_2_row <= layer_2_row;
        layer_2_col <= layer_2_col;
        layer_2_next_row <= layer_2_next_row;
        layer_2_next_col <= layer_2_next_col;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= in1_layer_3[i * LAYER_2_WIDTH + j];
                in2_layer_3[i * LAYER_2_WIDTH + j] <= in2_layer_3[i * LAYER_2_WIDTH + j];
            end

            acc_in1_layer_3[i] <= acc_in1_layer_3[i];
            acc_in2_layer_3[i] <= acc_in2_layer_3[i];

            out_layer_3[i] <= out_layer_3[i];
        end

        layer_3_index <= layer_3_index;
        layer_3_next_index <= layer_3_next_index;

        for (i = 0; i < (LAYER_2_VOLUME * LAYER_3_DEPTH); i = i + 1) begin
            layer_3_weights[i] <= layer_3_weights[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= output_error[i];
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= layer_3_der_patch[i];
            layer_3_err_weight_patch[i] <= layer_3_err_weight_patch[i];
            layer_3_acc[i] <= layer_3_acc[i];
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            layer_3_error[i] <= layer_3_error[i];
        end

        layer_3_err_depth_counter <= layer_3_err_depth_counter;
        layer_3_err_next_depth_counter <= layer_3_err_next_depth_counter;
        layer_3_err_index <= layer_3_err_index;
        layer_3_err_next_index <= layer_3_err_next_index;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= layer_3_gr_in_patch[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= layer_3_weight_patch[i * LAYER_2_WIDTH + j];
            end
        end

        layer_3_gr_index <= layer_3_gr_next_index;
        layer_3_gr_next_index <= layer_3_gr_next_index;
        
        Done <= 0;
    end
    STATE_L1_OUTPUT: begin
        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            for (j = 0; j < PATCH_LENGTH; j = j + 1) begin
                in1_layer_1[i * PATCH_LENGTH + j] <= in1_layer_1[i * PATCH_LENGTH + j];
                in2_layer_1[i * PATCH_LENGTH + j] <= in2_layer_1[i * PATCH_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= act_der_layer_1[i];
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            out_layer_1[layer_1_index + i * (LAYER_1_WIDTH * LAYER_1_HEIGHT)] <= out1_layer_1[i];
            der_layer_1[layer_1_index + i * (LAYER_1_WIDTH * LAYER_1_HEIGHT)] <= out2_layer_1[i];
        end

        layer_1_depth_counter <= layer_1_depth_counter;
        layer_1_map <= layer_1_map;
        layer_1_next_map <= layer_1_next_map;
        layer_1_mode <= FF_MODE;
        layer_1_acc <= 0;
        layer_1_index <= layer_1_index;
        layer_1_next_index <= layer_1_next_index + 1;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            for (j = 0; j < POOL_LENGTH; j = j + 1) begin
                in_layer_2[i * POOL_LENGTH + j] <= in_layer_2[i * POOL_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            out_layer_2[i] <= out_layer_2[i];
            der_layer_2[i] <= der_layer_2[i];
        end

        layer_1_corner <= layer_1_corner;
        layer_2_row <= layer_2_row;
        layer_2_col <= layer_2_col;
        layer_2_next_row <= layer_2_next_row;
        layer_2_next_col <= layer_2_next_col;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= in1_layer_3[i * LAYER_2_WIDTH + j];
                in2_layer_3[i * LAYER_2_WIDTH + j] <= in2_layer_3[i * LAYER_2_WIDTH + j];
            end

            acc_in1_layer_3[i] <= acc_in1_layer_3[i];
            acc_in2_layer_3[i] <= acc_in2_layer_3[i];

            out_layer_3[i] <= out_layer_3[i];
        end

        layer_3_index <= layer_3_index;
        layer_3_next_index <= layer_3_next_index;

        for (i = 0; i < (LAYER_2_VOLUME * LAYER_3_DEPTH); i = i + 1) begin
            layer_3_weights[i] <= layer_3_weights[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= output_error[i];
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= layer_3_der_patch[i];
            layer_3_err_weight_patch[i] <= layer_3_err_weight_patch[i];
            layer_3_acc[i] <= layer_3_acc[i];
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            layer_3_error[i] <= layer_3_error[i];
        end

        layer_3_err_depth_counter <= layer_3_err_depth_counter;
        layer_3_err_next_depth_counter <= layer_3_err_next_depth_counter;
        layer_3_err_index <= layer_3_err_index;
        layer_3_err_next_index <= layer_3_err_next_index;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= layer_3_gr_in_patch[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= layer_3_weight_patch[i * LAYER_2_WIDTH + j];
            end
        end

        layer_3_gr_index <= layer_3_gr_next_index;
        layer_3_gr_next_index <= layer_3_gr_next_index;
        
        Done <= 0;
    end
    STATE_L2_DATA: begin
        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            for (j = 0; j < PATCH_LENGTH; j = j + 1) begin
                in1_layer_1[i * PATCH_LENGTH + j] <= in1_layer_1[i * PATCH_LENGTH + j];
                in2_layer_1[i * PATCH_LENGTH + j] <= in2_layer_1[i * PATCH_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= act_der_layer_1[i];
        end

        for (i = 0; i < LAYER_1_VOLUME; i = i + 1) begin
            out_layer_1[i] <= out_layer_1[i];
            der_layer_1[i] <= der_layer_1[i];
        end

        layer_1_depth_counter <= layer_1_depth_counter;
        layer_1_map <= layer_1_map;
        layer_1_next_map <= layer_1_next_map;
        layer_1_mode <= layer_1_mode;
        layer_1_acc <= layer_1_acc;
        layer_1_index <= layer_1_index;
        layer_1_next_index <= layer_1_next_index;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            in_layer_2[i * POOL_LENGTH + 0] <= out_layer_1[i * (LAYER_1_WIDTH * LAYER_1_HEIGHT) + layer_1_corner + 0];
            in_layer_2[i * POOL_LENGTH + 1] <= out_layer_1[i * (LAYER_1_WIDTH * LAYER_1_HEIGHT) + layer_1_corner + 1];
            in_layer_2[i * POOL_LENGTH + 2] <= out_layer_1[i * (LAYER_1_WIDTH * LAYER_1_HEIGHT) + layer_1_corner + LAYER_1_WIDTH];
            in_layer_2[i * POOL_LENGTH + 3] <= out_layer_1[i * (LAYER_1_WIDTH * LAYER_1_HEIGHT) + layer_1_corner + LAYER_1_WIDTH + 1];
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            out_layer_2[i] <= out_layer_2[i];
            der_layer_2[i] <= der_layer_2[i];
        end

        layer_1_corner <= layer_1_corner;
        layer_2_row <= layer_2_next_row;
        layer_2_col <= layer_2_next_col;
        layer_2_next_row <= layer_2_next_row;
        layer_2_next_col <= layer_2_next_col;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= in1_layer_3[i * LAYER_2_WIDTH + j];
                in2_layer_3[i * LAYER_2_WIDTH + j] <= in2_layer_3[i * LAYER_2_WIDTH + j];
            end

            acc_in1_layer_3[i] <= acc_in1_layer_3[i];
            acc_in2_layer_3[i] <= acc_in2_layer_3[i];

            out_layer_3[i] <= out_layer_3[i];
        end

        layer_3_index <= layer_3_index;
        layer_3_next_index <= layer_3_next_index;

        for (i = 0; i < (LAYER_2_VOLUME * LAYER_3_DEPTH); i = i + 1) begin
            layer_3_weights[i] <= layer_3_weights[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= output_error[i];
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= layer_3_der_patch[i];
            layer_3_err_weight_patch[i] <= layer_3_err_weight_patch[i];
            layer_3_acc[i] <= layer_3_acc[i];
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            layer_3_error[i] <= layer_3_error[i];
        end

        layer_3_err_depth_counter <= layer_3_err_depth_counter;
        layer_3_err_next_depth_counter <= layer_3_err_next_depth_counter;
        layer_3_err_index <= layer_3_err_index;
        layer_3_err_next_index <= layer_3_err_next_index;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= layer_3_gr_in_patch[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= layer_3_weight_patch[i * LAYER_2_WIDTH + j];
            end
        end

        layer_3_gr_index <= layer_3_gr_next_index;
        layer_3_gr_next_index <= layer_3_gr_next_index;

        Done <= 0;
    end
    STATE_L2_OUTPUT: begin
        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            for (j = 0; j < PATCH_LENGTH; j = j + 1) begin
                in1_layer_1[i * PATCH_LENGTH + j] <= in1_layer_1[i * PATCH_LENGTH + j];
                in2_layer_1[i * PATCH_LENGTH + j] <= in2_layer_1[i * PATCH_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= act_der_layer_1[i];
        end

        for (i = 0; i < LAYER_1_VOLUME; i = i + 1) begin
            out_layer_1[i] <= out_layer_1[i];
            der_layer_1[i] <= der_layer_1[i];
        end

        layer_1_depth_counter <= layer_1_depth_counter;
        layer_1_map <= layer_1_map;
        layer_1_next_map <= layer_1_next_map;
        layer_1_mode <= layer_1_mode;
        layer_1_acc <= layer_1_acc;
        layer_1_index <= layer_1_index;
        layer_1_next_index <= layer_1_next_index;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            for (j = 0; j < POOL_LENGTH; j = j + 1) begin
                in_layer_2[i * POOL_LENGTH + j] <= in_layer_2[i * POOL_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            out_layer_2[i * (LAYER_2_WIDTH * LAYER_2_HEIGHT) + layer_2_row * LAYER_2_WIDTH + layer_2_col] <= out1_layer_2[i];
            der_layer_2[i * (LAYER_2_WIDTH * LAYER_2_HEIGHT) + layer_2_row * LAYER_2_WIDTH + layer_2_col] <= out2_layer_2[i];
        end

        layer_1_corner <= layer_2_row * 2 * LAYER_1_WIDTH + layer_2_col * 2;
        layer_2_row <= layer_2_row;
        layer_2_col <= layer_2_col;
        layer_2_next_row <= (layer_2_col == (LAYER_2_WIDTH - 1)) ? (layer_2_row + 1) : layer_2_row;
        layer_2_next_col <= (layer_2_col == (LAYER_2_WIDTH - 1)) ? 0 : layer_2_col + 1;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= in1_layer_3[i * LAYER_2_WIDTH + j];
                in2_layer_3[i * LAYER_2_WIDTH + j] <= in2_layer_3[i * LAYER_2_WIDTH + j];
            end

            acc_in1_layer_3[i] <= acc_in1_layer_3[i];
            acc_in2_layer_3[i] <= acc_in2_layer_3[i];

            out_layer_3[i] <= out_layer_3[i];
        end

        layer_3_index <= layer_3_index;
        layer_3_next_index <= layer_3_next_index;

        for (i = 0; i < (LAYER_2_VOLUME * LAYER_3_DEPTH); i = i + 1) begin
            layer_3_weights[i] <= layer_3_weights[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= output_error[i];
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= layer_3_der_patch[i];
            layer_3_err_weight_patch[i] <= layer_3_err_weight_patch[i];
            layer_3_acc[i] <= layer_3_acc[i];
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            layer_3_error[i] <= layer_3_error[i];
        end

        layer_3_err_depth_counter <= layer_3_err_depth_counter;
        layer_3_err_next_depth_counter <= layer_3_err_next_depth_counter;
        layer_3_err_index <= layer_3_err_index;
        layer_3_err_next_index <= layer_3_err_next_index;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= layer_3_gr_in_patch[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= layer_3_weight_patch[i * LAYER_2_WIDTH + j];
            end
        end

        layer_3_gr_index <= layer_3_gr_next_index;
        layer_3_gr_next_index <= layer_3_gr_next_index;

        Done <= 0;
    end
    STATE_L3_DATA: begin
        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            for (j = 0; j < PATCH_LENGTH; j = j + 1) begin
                in1_layer_1[i * PATCH_LENGTH + j] <= in1_layer_1[i * PATCH_LENGTH + j];
                in2_layer_1[i * PATCH_LENGTH + j] <= in2_layer_1[i * PATCH_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= act_der_layer_1[i];
        end

        for (i = 0; i < LAYER_1_VOLUME; i = i + 1) begin
            out_layer_1[i] <= out_layer_1[i];
            der_layer_1[i] <= der_layer_1[i];
        end

        layer_1_depth_counter <= layer_1_depth_counter;
        layer_1_map <= layer_1_map;
        layer_1_next_map <= layer_1_next_map;
        layer_1_mode <= layer_1_mode;
        layer_1_acc <= layer_1_acc;
        layer_1_index <= layer_1_index;
        layer_1_next_index <= layer_1_next_index;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            for (j = 0; j < POOL_LENGTH; j = j + 1) begin
                in_layer_2[i * POOL_LENGTH + j] <= in_layer_2[i * POOL_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            out_layer_2[i] <= out_layer_2[i];
            der_layer_2[i] <= der_layer_2[i];
        end

        layer_1_corner <= layer_1_corner;
        layer_2_row <= layer_2_row;
        layer_2_col <= layer_2_col;
        layer_2_next_row <= layer_2_next_row;
        layer_2_next_col <= layer_2_next_col;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= out_layer_2[layer_3_index * LAYER_2_WIDTH + j];
                in2_layer_3[i * LAYER_2_WIDTH + j] <= layer_3_weights[i * LAYER_2_VOLUME + layer_3_index * LAYER_2_WIDTH + j];
            end

            acc_in1_layer_3[i] <= acc_in1_layer_3[i];
            acc_in2_layer_3[i] <= acc_in2_layer_3[i];

            out_layer_3[i] <= out_layer_3[i];
        end

        layer_3_index <= layer_3_next_index;
        layer_3_next_index <= layer_3_next_index;

        for (i = 0; i < (LAYER_2_VOLUME * LAYER_3_DEPTH); i = i + 1) begin
            layer_3_weights[i] <= layer_3_weights[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= output_error[i];
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= layer_3_der_patch[i];
            layer_3_err_weight_patch[i] <= layer_3_err_weight_patch[i];
            layer_3_acc[i] <= layer_3_acc[i];
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            layer_3_error[i] <= layer_3_error[i];
        end

        layer_3_err_depth_counter <= layer_3_err_depth_counter;
        layer_3_err_next_depth_counter <= layer_3_err_next_depth_counter;
        layer_3_err_index <= layer_3_err_index;
        layer_3_err_next_index <= layer_3_err_next_index;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= layer_3_gr_in_patch[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= layer_3_weight_patch[i * LAYER_2_WIDTH + j];
            end
        end

        layer_3_gr_index <= layer_3_gr_next_index;
        layer_3_gr_next_index <= layer_3_gr_next_index;

        Done <= 0;
    end
    STATE_L3_ACC: begin
        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            for (j = 0; j < PATCH_LENGTH; j = j + 1) begin
                in1_layer_1[i * PATCH_LENGTH + j] <= in1_layer_1[i * PATCH_LENGTH + j];
                in2_layer_1[i * PATCH_LENGTH + j] <= in2_layer_1[i * PATCH_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= act_der_layer_1[i];
        end

        for (i = 0; i < LAYER_1_VOLUME; i = i + 1) begin
            out_layer_1[i] <= out_layer_1[i];
            der_layer_1[i] <= der_layer_1[i];
        end

        layer_1_depth_counter <= layer_1_depth_counter;
        layer_1_map <= layer_1_map;
        layer_1_next_map <= layer_1_next_map;
        layer_1_mode <= layer_1_mode;
        layer_1_acc <= layer_1_acc;
        layer_1_index <= layer_1_index;
        layer_1_next_index <= layer_1_next_index;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            for (j = 0; j < POOL_LENGTH; j = j + 1) begin
                in_layer_2[i * POOL_LENGTH + j] <= in_layer_2[i * POOL_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            out_layer_2[i] <= out_layer_2[i];
            der_layer_2[i] <= der_layer_2[i];
        end

        layer_1_corner <= layer_1_corner;
        layer_2_row <= layer_2_row;
        layer_2_col <= layer_2_col;
        layer_2_next_row <= layer_2_next_row;
        layer_2_next_col <= layer_2_next_col;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= in1_layer_3[i * LAYER_2_WIDTH + j];
                in2_layer_3[i * LAYER_2_WIDTH + j] <= in2_layer_3[i * LAYER_2_WIDTH + j];
            end

            acc_in1_layer_3[i] <= out_layer_3[i];
            acc_in2_layer_3[i] <= out1_layer_3[i];

            out_layer_3[i] <= out_layer_3[i];
        end

        layer_3_index <= layer_3_index;
        layer_3_next_index <= layer_3_next_index;

        for (i = 0; i < (LAYER_2_VOLUME * LAYER_3_DEPTH); i = i + 1) begin
            layer_3_weights[i] <= layer_3_weights[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= output_error[i];
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= layer_3_der_patch[i];
            layer_3_err_weight_patch[i] <= layer_3_err_weight_patch[i];
            layer_3_acc[i] <= layer_3_acc[i];
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            layer_3_error[i] <= layer_3_error[i];
        end

        layer_3_err_depth_counter <= layer_3_err_depth_counter;
        layer_3_err_next_depth_counter <= layer_3_err_next_depth_counter;
        layer_3_err_index <= layer_3_err_index;
        layer_3_err_next_index <= layer_3_err_next_index;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= layer_3_gr_in_patch[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= layer_3_weight_patch[i * LAYER_2_WIDTH + j];
            end
        end

        layer_3_gr_index <= layer_3_gr_next_index;
        layer_3_gr_next_index <= layer_3_gr_next_index;

        Done <= 0;
    end
    STATE_L3_OUTPUT: begin
        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            for (j = 0; j < PATCH_LENGTH; j = j + 1) begin
                in1_layer_1[i * PATCH_LENGTH + j] <= in1_layer_1[i * PATCH_LENGTH + j];
                in2_layer_1[i * PATCH_LENGTH + j] <= in2_layer_1[i * PATCH_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= act_der_layer_1[i];
        end

        for (i = 0; i < LAYER_1_VOLUME; i = i + 1) begin
            out_layer_1[i] <= out_layer_1[i];
            der_layer_1[i] <= der_layer_1[i];
        end

        layer_1_depth_counter <= layer_1_depth_counter;
        layer_1_map <= layer_1_map;
        layer_1_next_map <= layer_1_next_map;
        layer_1_mode <= layer_1_mode;
        layer_1_acc <= layer_1_acc;
        layer_1_index <= layer_1_index;
        layer_1_next_index <= layer_1_next_index;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            for (j = 0; j < POOL_LENGTH; j = j + 1) begin
                in_layer_2[i * POOL_LENGTH + j] <= in_layer_2[i * POOL_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            out_layer_2[i] <= out_layer_2[i];
            der_layer_2[i] <= der_layer_2[i];
        end

        layer_1_corner <= layer_1_corner;
        layer_2_row <= layer_2_row;
        layer_2_col <= layer_2_col;
        layer_2_next_row <= layer_2_next_row;
        layer_2_next_col <= layer_2_next_col;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= in1_layer_3[i * LAYER_2_WIDTH + j];
                in2_layer_3[i * LAYER_2_WIDTH + j] <= in2_layer_3[i * LAYER_2_WIDTH + j];
            end

            acc_in1_layer_3[i] <= acc_in1_layer_3[i];
            acc_in2_layer_3[i] <= acc_in2_layer_3[i];

            out_layer_3[i] <= acc_out_layer_3[i];
        end

        layer_3_index <= layer_3_index;
        layer_3_next_index <= layer_3_index + 1;

        for (i = 0; i < (LAYER_2_VOLUME * LAYER_3_DEPTH); i = i + 1) begin
            layer_3_weights[i] <= layer_3_weights[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= output_error[i];
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= layer_3_der_patch[i];
            layer_3_err_weight_patch[i] <= layer_3_err_weight_patch[i];
            layer_3_acc[i] <= layer_3_acc[i];
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            layer_3_error[i] <= layer_3_error[i];
        end

        layer_3_err_depth_counter <= layer_3_err_depth_counter;
        layer_3_err_next_depth_counter <= layer_3_err_next_depth_counter;
        layer_3_err_index <= layer_3_err_index;
        layer_3_err_next_index <= layer_3_err_next_index;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= layer_3_gr_in_patch[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= layer_3_weight_patch[i * LAYER_2_WIDTH + j];
            end
        end

        layer_3_gr_index <= layer_3_gr_next_index;
        layer_3_gr_next_index <= layer_3_gr_next_index;

        Done <= 0;
    end
    STATE_ERROR: begin
        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            for (j = 0; j < PATCH_LENGTH; j = j + 1) begin
                in1_layer_1[i * PATCH_LENGTH + j] <= in1_layer_1[i * PATCH_LENGTH + j];
                in2_layer_1[i * PATCH_LENGTH + j] <= in2_layer_1[i * PATCH_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= act_der_layer_1[i];
        end

        for (i = 0; i < LAYER_1_VOLUME; i = i + 1) begin
            out_layer_1[i] <= out_layer_1[i];
            der_layer_1[i] <= der_layer_1[i];
        end

        layer_1_depth_counter <= layer_1_depth_counter;
        layer_1_map <= layer_1_map;
        layer_1_next_map <= layer_1_next_map;
        layer_1_mode <= layer_1_mode;
        layer_1_acc <= layer_1_acc;
        layer_1_index <= layer_1_index;
        layer_1_next_index <= layer_1_next_index;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            for (j = 0; j < POOL_LENGTH; j = j + 1) begin
                in_layer_2[i * POOL_LENGTH + j] <= in_layer_2[i * POOL_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            out_layer_2[i] <= out_layer_2[i];
            der_layer_2[i] <= der_layer_2[i];
        end

        layer_1_corner <= layer_1_corner;
        layer_2_row <= layer_2_row;
        layer_2_col <= layer_2_col;
        layer_2_next_row <= layer_2_next_row;
        layer_2_next_col <= layer_2_next_col;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= in1_layer_3[i * LAYER_2_WIDTH + j];
                in2_layer_3[i * LAYER_2_WIDTH + j] <= in2_layer_3[i * LAYER_2_WIDTH + j];
            end

            acc_in1_layer_3[i] <= acc_in1_layer_3[i];
            acc_in2_layer_3[i] <= acc_in2_layer_3[i];

            out_layer_3[i] <= out_layer_3[i];
        end

        layer_3_index <= layer_3_index;
        layer_3_next_index <= layer_3_next_index;

        for (i = 0; i < (LAYER_2_VOLUME * LAYER_3_DEPTH); i = i + 1) begin
            layer_3_weights[i] <= layer_3_weights[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= output_error_q[i];
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= layer_3_der_patch[i];
            layer_3_err_weight_patch[i] <= layer_3_err_weight_patch[i];
            layer_3_acc[i] <= layer_3_acc[i];
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            layer_3_error[i] <= layer_3_error[i];
        end

        layer_3_err_depth_counter <= layer_3_err_depth_counter;
        layer_3_err_next_depth_counter <= layer_3_err_next_depth_counter;
        layer_3_err_index <= layer_3_err_index;
        layer_3_err_next_index <= layer_3_err_next_index;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= layer_3_gr_in_patch[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= layer_3_weight_patch[i * LAYER_2_WIDTH + j];
            end
        end

        layer_3_gr_index <= layer_3_gr_next_index;
        layer_3_gr_next_index <= layer_3_gr_next_index;
        
        Done <= 0;
    end
    STATE_L3_ERR_DATA: begin
        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            for (j = 0; j < PATCH_LENGTH; j = j + 1) begin
                in1_layer_1[i * PATCH_LENGTH + j] <= in1_layer_1[i * PATCH_LENGTH + j];
                in2_layer_1[i * PATCH_LENGTH + j] <= in2_layer_1[i * PATCH_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= act_der_layer_1[i];
        end

        for (i = 0; i < LAYER_1_VOLUME; i = i + 1) begin
            out_layer_1[i] <= out_layer_1[i];
            der_layer_1[i] <= der_layer_1[i];
        end

        layer_1_depth_counter <= layer_1_depth_counter;
        layer_1_map <= layer_1_map;
        layer_1_next_map <= layer_1_next_map;
        layer_1_mode <= layer_1_mode;
        layer_1_acc <= layer_1_acc;
        layer_1_index <= layer_1_index;
        layer_1_next_index <= layer_1_next_index;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            for (j = 0; j < POOL_LENGTH; j = j + 1) begin
                in_layer_2[i * POOL_LENGTH + j] <= in_layer_2[i * POOL_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            out_layer_2[i] <= out_layer_2[i];
            der_layer_2[i] <= der_layer_2[i];
        end

        layer_1_corner <= layer_1_corner;
        layer_2_row <= layer_2_row;
        layer_2_col <= layer_2_col;
        layer_2_next_row <= layer_2_next_row;
        layer_2_next_col <= layer_2_next_col;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= in1_layer_3[i * LAYER_2_WIDTH + j];
                in2_layer_3[i * LAYER_2_WIDTH + j] <= in2_layer_3[i * LAYER_2_WIDTH + j];
            end

            acc_in1_layer_3[i] <= acc_in1_layer_3[i];
            acc_in2_layer_3[i] <= acc_in2_layer_3[i];

            out_layer_3[i] <= out_layer_3[i];
        end

        layer_3_index <= layer_3_index;
        layer_3_next_index <= layer_3_next_index;

        for (i = 0; i < (LAYER_2_VOLUME * LAYER_3_DEPTH); i = i + 1) begin
            layer_3_weights[i] <= layer_3_weights[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= output_error[i];
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= der_layer_2[layer_3_err_index * LAYER_2_WIDTH + i];
            layer_3_err_weight_patch[i] <= layer_3_weights[layer_3_err_depth_counter * LAYER_2_VOLUME + layer_3_err_index * LAYER_2_WIDTH + i];
            layer_3_acc[i] <= (layer_3_err_depth_counter == 0) ? 0 : layer_3_acc[i];
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            layer_3_error[i] <= layer_3_error[i];
        end

        layer_3_err_depth_counter <= layer_3_err_depth_counter;
        layer_3_err_next_depth_counter <= (layer_3_err_depth_counter == (LAYER_3_DEPTH - 1)) ? 0 : (layer_3_err_depth_counter + 1);
        layer_3_err_index <= layer_3_err_index;
        layer_3_err_next_index <= (layer_3_err_depth_counter == (LAYER_3_DEPTH - 1)) ? (layer_3_err_index + 1) : layer_3_err_next_index;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= layer_3_gr_in_patch[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= layer_3_weight_patch[i * LAYER_2_WIDTH + j];
            end
        end

        layer_3_gr_index <= layer_3_gr_next_index;
        layer_3_gr_next_index <= layer_3_gr_next_index;
        
        Done <= 0;
    end
    STATE_L3_ERR_ACC: begin
        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            for (j = 0; j < PATCH_LENGTH; j = j + 1) begin
                in1_layer_1[i * PATCH_LENGTH + j] <= in1_layer_1[i * PATCH_LENGTH + j];
                in2_layer_1[i * PATCH_LENGTH + j] <= in2_layer_1[i * PATCH_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= act_der_layer_1[i];
        end

        for (i = 0; i < LAYER_1_VOLUME; i = i + 1) begin
            out_layer_1[i] <= out_layer_1[i];
            der_layer_1[i] <= der_layer_1[i];
        end

        layer_1_depth_counter <= layer_1_depth_counter;
        layer_1_map <= layer_1_map;
        layer_1_next_map <= layer_1_next_map;
        layer_1_mode <= layer_1_mode;
        layer_1_acc <= layer_1_acc;
        layer_1_index <= layer_1_index;
        layer_1_next_index <= layer_1_next_index;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            for (j = 0; j < POOL_LENGTH; j = j + 1) begin
                in_layer_2[i * POOL_LENGTH + j] <= in_layer_2[i * POOL_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            out_layer_2[i] <= out_layer_2[i];
            der_layer_2[i] <= der_layer_2[i];
        end

        layer_1_corner <= layer_1_corner;
        layer_2_row <= layer_2_row;
        layer_2_col <= layer_2_col;
        layer_2_next_row <= layer_2_next_row;
        layer_2_next_col <= layer_2_next_col;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= in1_layer_3[i * LAYER_2_WIDTH + j];
                in2_layer_3[i * LAYER_2_WIDTH + j] <= in2_layer_3[i * LAYER_2_WIDTH + j];
            end

            acc_in1_layer_3[i] <= acc_in1_layer_3[i];
            acc_in2_layer_3[i] <= acc_in2_layer_3[i];

            out_layer_3[i] <= out_layer_3[i];
        end

        layer_3_index <= layer_3_index;
        layer_3_next_index <= layer_3_next_index;

        for (i = 0; i < (LAYER_2_VOLUME * LAYER_3_DEPTH); i = i + 1) begin
            layer_3_weights[i] <= layer_3_weights[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= output_error[i];
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= layer_3_der_patch[i];
            layer_3_err_weight_patch[i] <= layer_3_err_weight_patch[i];
            layer_3_acc[i] <= layer_3_new_acc[i];
            layer_3_error[layer_3_err_index * LAYER_2_WIDTH + i] <= layer_3_new_acc[i];
        end

        layer_3_err_depth_counter <= layer_3_err_depth_counter;
        layer_3_err_next_depth_counter <= layer_3_err_next_depth_counter;
        layer_3_err_index <= layer_3_err_index;
        layer_3_err_next_index <= layer_3_err_next_index;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= layer_3_gr_in_patch[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= layer_3_weight_patch[i * LAYER_2_WIDTH + j];
            end
        end

        layer_3_gr_index <= layer_3_gr_next_index;
        layer_3_gr_next_index <= layer_3_gr_next_index;
        
        Done <= 0;
    end
    STATE_L3_ERR_INCR: begin
        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            for (j = 0; j < PATCH_LENGTH; j = j + 1) begin
                in1_layer_1[i * PATCH_LENGTH + j] <= in1_layer_1[i * PATCH_LENGTH + j];
                in2_layer_1[i * PATCH_LENGTH + j] <= in2_layer_1[i * PATCH_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= act_der_layer_1[i];
        end

        for (i = 0; i < LAYER_1_VOLUME; i = i + 1) begin
            out_layer_1[i] <= out_layer_1[i];
            der_layer_1[i] <= der_layer_1[i];
        end

        layer_1_depth_counter <= layer_1_depth_counter;
        layer_1_map <= layer_1_map;
        layer_1_next_map <= layer_1_next_map;
        layer_1_mode <= layer_1_mode;
        layer_1_acc <= layer_1_acc;
        layer_1_index <= layer_1_index;
        layer_1_next_index <= layer_1_next_index;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            for (j = 0; j < POOL_LENGTH; j = j + 1) begin
                in_layer_2[i * POOL_LENGTH + j] <= in_layer_2[i * POOL_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            out_layer_2[i] <= out_layer_2[i];
            der_layer_2[i] <= der_layer_2[i];
        end

        layer_1_corner <= layer_1_corner;
        layer_2_row <= layer_2_row;
        layer_2_col <= layer_2_col;
        layer_2_next_row <= layer_2_next_row;
        layer_2_next_col <= layer_2_next_col;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= in1_layer_3[i * LAYER_2_WIDTH + j];
                in2_layer_3[i * LAYER_2_WIDTH + j] <= in2_layer_3[i * LAYER_2_WIDTH + j];
            end

            acc_in1_layer_3[i] <= acc_in1_layer_3[i];
            acc_in2_layer_3[i] <= acc_in2_layer_3[i];

            out_layer_3[i] <= out_layer_3[i];
        end

        layer_3_index <= layer_3_index;
        layer_3_next_index <= layer_3_next_index;

        for (i = 0; i < (LAYER_2_VOLUME * LAYER_3_DEPTH); i = i + 1) begin
            layer_3_weights[i] <= layer_3_weights[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= output_error[i];
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= layer_3_der_patch[i];
            layer_3_err_weight_patch[i] <= layer_3_err_weight_patch[i];
            layer_3_acc[i] <= layer_3_acc[i];
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            layer_3_error[i] <= layer_3_error[i];
        end

        layer_3_err_depth_counter <= layer_3_err_next_depth_counter;
        layer_3_err_next_depth_counter <= layer_3_err_next_depth_counter;
        layer_3_err_index <= layer_3_err_next_index;
        layer_3_err_next_index <= layer_3_err_next_index;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= layer_3_gr_in_patch[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= layer_3_weight_patch[i * LAYER_2_WIDTH + j];
            end
        end

        layer_3_gr_index <= layer_3_gr_next_index;
        layer_3_gr_next_index <= layer_3_gr_next_index;
        
        Done <= 0;
    end
    STATE_L3_GR_DATA: begin
        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            for (j = 0; j < PATCH_LENGTH; j = j + 1) begin
                in1_layer_1[i * PATCH_LENGTH + j] <= in1_layer_1[i * PATCH_LENGTH + j];
                in2_layer_1[i * PATCH_LENGTH + j] <= in2_layer_1[i * PATCH_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= act_der_layer_1[i];
        end

        for (i = 0; i < LAYER_1_VOLUME; i = i + 1) begin
            out_layer_1[i] <= out_layer_1[i];
            der_layer_1[i] <= der_layer_1[i];
        end

        layer_1_depth_counter <= layer_1_depth_counter;
        layer_1_map <= layer_1_map;
        layer_1_next_map <= layer_1_next_map;
        layer_1_mode <= layer_1_mode;
        layer_1_acc <= layer_1_acc;
        layer_1_index <= layer_1_index;
        layer_1_next_index <= layer_1_next_index;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            for (j = 0; j < POOL_LENGTH; j = j + 1) begin
                in_layer_2[i * POOL_LENGTH + j] <= in_layer_2[i * POOL_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            out_layer_2[i] <= out_layer_2[i];
            der_layer_2[i] <= der_layer_2[i];
        end

        layer_1_corner <= layer_1_corner;
        layer_2_row <= layer_2_row;
        layer_2_col <= layer_2_col;
        layer_2_next_row <= layer_2_next_row;
        layer_2_next_col <= layer_2_next_col;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= in1_layer_3[i * LAYER_2_WIDTH + j];
                in2_layer_3[i * LAYER_2_WIDTH + j] <= in2_layer_3[i * LAYER_2_WIDTH + j];
            end

            acc_in1_layer_3[i] <= acc_in1_layer_3[i];
            acc_in2_layer_3[i] <= acc_in2_layer_3[i];

            out_layer_3[i] <= out_layer_3[i];
        end

        layer_3_index <= layer_3_index;
        layer_3_next_index <= layer_3_next_index;

        for (i = 0; i < (LAYER_2_VOLUME * LAYER_3_DEPTH); i = i + 1) begin
            layer_3_weights[i] <= layer_3_weights[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= output_error[i];
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= layer_3_der_patch[i];
            layer_3_err_weight_patch[i] <= layer_3_err_weight_patch[i];
            layer_3_acc[i] <= layer_3_acc[i];
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            layer_3_error[i] <= layer_3_error[i];
        end

        layer_3_err_depth_counter <= layer_3_err_depth_counter;
        layer_3_err_next_depth_counter <= layer_3_err_next_depth_counter;
        layer_3_err_index <= layer_3_err_index;
        layer_3_err_next_index <= layer_3_err_next_index;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= out_layer_2[layer_3_gr_index * LAYER_2_WIDTH + i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= layer_3_weights[i * LAYER_2_VOLUME + layer_3_gr_index * LAYER_2_WIDTH + j];
            end
        end

        layer_3_gr_index <= layer_3_gr_next_index;
        layer_3_gr_next_index <= layer_3_gr_index + 1;
        
        Done <= 0;
    end
    STATE_L3_GR_UPDATE: begin
        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            for (j = 0; j < PATCH_LENGTH; j = j + 1) begin
                in1_layer_1[i * PATCH_LENGTH + j] <= in1_layer_1[i * PATCH_LENGTH + j];
                in2_layer_1[i * PATCH_LENGTH + j] <= in2_layer_1[i * PATCH_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= act_der_layer_1[i];
        end

        for (i = 0; i < LAYER_1_VOLUME; i = i + 1) begin
            out_layer_1[i] <= out_layer_1[i];
            der_layer_1[i] <= der_layer_1[i];
        end

        layer_1_depth_counter <= layer_1_depth_counter;
        layer_1_map <= layer_1_map;
        layer_1_next_map <= layer_1_next_map;
        layer_1_mode <= layer_1_mode;
        layer_1_acc <= layer_1_acc;
        layer_1_index <= layer_1_index;
        layer_1_next_index <= layer_1_next_index;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            for (j = 0; j < POOL_LENGTH; j = j + 1) begin
                in_layer_2[i * POOL_LENGTH + j] <= in_layer_2[i * POOL_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            out_layer_2[i] <= out_layer_2[i];
            der_layer_2[i] <= der_layer_2[i];
        end

        layer_1_corner <= layer_1_corner;
        layer_2_row <= layer_2_row;
        layer_2_col <= layer_2_col;
        layer_2_next_row <= layer_2_next_row;
        layer_2_next_col <= layer_2_next_col;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= in1_layer_3[i * LAYER_2_WIDTH + j];
                in2_layer_3[i * LAYER_2_WIDTH + j] <= in2_layer_3[i * LAYER_2_WIDTH + j];
            end

            acc_in1_layer_3[i] <= acc_in1_layer_3[i];
            acc_in2_layer_3[i] <= acc_in2_layer_3[i];

            out_layer_3[i] <= out_layer_3[i];
        end

        layer_3_index <= layer_3_index;
        layer_3_next_index <= layer_3_next_index;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= output_error[i];
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= layer_3_der_patch[i];
            layer_3_err_weight_patch[i] <= layer_3_err_weight_patch[i];
            layer_3_acc[i] <= layer_3_acc[i];
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            layer_3_error[i] <= layer_3_error[i];
        end

        layer_3_err_depth_counter <= layer_3_err_depth_counter;
        layer_3_err_next_depth_counter <= layer_3_err_next_depth_counter;
        layer_3_err_index <= layer_3_err_index;
        layer_3_err_next_index <= layer_3_err_next_index;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= layer_3_gr_in_patch[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= layer_3_weight_patch[i * LAYER_2_WIDTH + j];
                layer_3_weights[i * LAYER_2_VOLUME + layer_3_gr_index * LAYER_2_WIDTH + j] <= layer_3_gr_update_patch[i * LAYER_2_WIDTH + j];
            end
        end

        layer_3_gr_index <= layer_3_gr_next_index;
        layer_3_gr_next_index <= layer_3_gr_next_index;
        
        Done <= 0;
    end
    STATE_FINAL_OUTPUT: begin
        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            for (j = 0; j < PATCH_LENGTH; j = j + 1) begin
                in1_layer_1[i * PATCH_LENGTH + j] <= in1_layer_1[i * PATCH_LENGTH + j];
                in2_layer_1[i * PATCH_LENGTH + j] <= in2_layer_1[i * PATCH_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_1_DEPTH; i = i + 1) begin
            act_der_layer_1[i] <= act_der_layer_1[i];
        end

        for (i = 0; i < LAYER_1_VOLUME; i = i + 1) begin
            out_layer_1[i] <= out_layer_1[i];
            der_layer_1[i] <= der_layer_1[i];
        end

        layer_1_depth_counter <= layer_1_depth_counter;
        layer_1_map <= layer_1_map;
        layer_1_next_map <= layer_1_next_map;
        layer_1_mode <= layer_1_mode;
        layer_1_acc <= layer_1_acc;
        layer_1_index <= layer_1_index;
        layer_1_next_index <= layer_1_next_index;

        for (i = 0; i < LAYER_2_DEPTH; i = i + 1) begin
            for (j = 0; j < POOL_LENGTH; j = j + 1) begin
                in_layer_2[i * POOL_LENGTH + j] <= in_layer_2[i * POOL_LENGTH + j];
            end
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            out_layer_2[i] <= out_layer_2[i];
            der_layer_2[i] <= der_layer_2[i];
        end

        layer_1_corner <= layer_1_corner;
        layer_2_row <= layer_2_row;
        layer_2_col <= layer_2_col;
        layer_2_next_row <= layer_2_next_row;
        layer_2_next_col <= layer_2_next_col;

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                in1_layer_3[i * LAYER_2_WIDTH + j] <= in1_layer_3[i * LAYER_2_WIDTH + j];
                in2_layer_3[i * LAYER_2_WIDTH + j] <= in2_layer_3[i * LAYER_2_WIDTH + j];
            end

            acc_in1_layer_3[i] <= acc_in1_layer_3[i];
            acc_in2_layer_3[i] <= acc_in2_layer_3[i];

            out_layer_3[i] <= out_layer_3[i];
        end

        layer_3_index <= layer_3_index;
        layer_3_next_index <= layer_3_next_index;

        for (i = 0; i < (LAYER_2_VOLUME * LAYER_3_DEPTH); i = i + 1) begin
            layer_3_weights[i] <= layer_3_weights[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            output_error[i] <= output_error[i];
        end

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_der_patch[i] <= layer_3_der_patch[i];
            layer_3_err_weight_patch[i] <= layer_3_err_weight_patch[i];
            layer_3_acc[i] <= layer_3_acc[i];
        end

        for (i = 0; i < LAYER_2_VOLUME; i = i + 1) begin
            layer_3_error[i] <= layer_3_error[i];
        end

        layer_3_err_depth_counter <= layer_3_err_depth_counter;
        layer_3_err_next_depth_counter <= layer_3_err_next_depth_counter;
        layer_3_err_index <= layer_3_err_index;
        layer_3_err_next_index <= layer_3_err_next_index;

        for (i = 0; i < LAYER_2_WIDTH; i = i + 1) begin
            layer_3_gr_in_patch[i] <= layer_3_gr_in_patch[i];
        end

        for (i = 0; i < LAYER_3_DEPTH; i = i + 1) begin
            for (j = 0; j < LAYER_2_WIDTH; j = j + 1) begin
                layer_3_weight_patch[i * LAYER_2_WIDTH + j] <= layer_3_weight_patch[i * LAYER_2_WIDTH + j];
            end
        end

        layer_3_gr_index <= layer_3_gr_next_index;
        layer_3_gr_next_index <= layer_3_gr_next_index;
        
        Done <= 1;
    end
    endcase
end

filter3x3 #(.KERNEL_SIZE(KERNEL_SIZE), .DEPTH(LAYER_1_DEPTH), .LEARNING_RATE(LEARNING_RATE), .WL(WL), .FL(FL)) filter_layer11(
    .CLK(CLK), 
    .RESET(RESET), 
    .acc(layer_1_acc), 
    .mode(layer_1_mode), 
    .map(layer_1_map), 
    .activation_derivative(act_der_layer_1[0]), 
    .in11(in1_layer_1[0]), 
    .in12(in1_layer_1[1]), 
    .in13(in1_layer_1[2]), 
    .in14(in1_layer_1[3]), 
    .in15(in1_layer_1[4]), 
    .in16(in1_layer_1[5]), 
    .in17(in1_layer_1[6]), 
    .in18(in1_layer_1[7]), 
    .in19(in1_layer_1[8]), 
    .in21(in2_layer_1[0]),
    .in22(in2_layer_1[1]),
    .in23(in2_layer_1[2]),
    .in24(in2_layer_1[3]),
    .in25(in2_layer_1[4]),
    .in26(in2_layer_1[5]),
    .in27(in2_layer_1[6]),
    .in28(in2_layer_1[7]),
    .in29(in2_layer_1[8]),
    .out1(out1_layer_1[0]), 
    .out2(out2_layer_1[0])
);

filter3x3 #(.KERNEL_SIZE(KERNEL_SIZE), .DEPTH(INPUT_DEPTH), .LEARNING_RATE(LEARNING_RATE), .WL(WL), .FL(FL)) filter_layer12(
    .CLK(CLK), 
    .RESET(RESET), 
    .acc(layer_1_acc), 
    .mode(layer_1_mode), 
    .map(layer_1_map), 
    .activation_derivative(act_der_layer_1[1]), 
    .in11(in1_layer_1[1 * PATCH_LENGTH + 0]), 
    .in12(in1_layer_1[1 * PATCH_LENGTH + 1]), 
    .in13(in1_layer_1[1 * PATCH_LENGTH + 2]), 
    .in14(in1_layer_1[1 * PATCH_LENGTH + 3]), 
    .in15(in1_layer_1[1 * PATCH_LENGTH + 4]), 
    .in16(in1_layer_1[1 * PATCH_LENGTH + 5]), 
    .in17(in1_layer_1[1 * PATCH_LENGTH + 6]), 
    .in18(in1_layer_1[1 * PATCH_LENGTH + 7]), 
    .in19(in1_layer_1[1 * PATCH_LENGTH + 8]), 
    .in21(in2_layer_1[1 * PATCH_LENGTH + 0]),
    .in22(in2_layer_1[1 * PATCH_LENGTH + 1]),
    .in23(in2_layer_1[1 * PATCH_LENGTH + 2]),
    .in24(in2_layer_1[1 * PATCH_LENGTH + 3]),
    .in25(in2_layer_1[1 * PATCH_LENGTH + 4]),
    .in26(in2_layer_1[1 * PATCH_LENGTH + 5]),
    .in27(in2_layer_1[1 * PATCH_LENGTH + 6]),
    .in28(in2_layer_1[1 * PATCH_LENGTH + 7]),
    .in29(in2_layer_1[1 * PATCH_LENGTH + 8]),
    .out1(out1_layer_1[1]), 
    .out2(out2_layer_1[1])
);

filter3x3 #(.KERNEL_SIZE(KERNEL_SIZE), .DEPTH(INPUT_DEPTH), .LEARNING_RATE(LEARNING_RATE), .WL(WL), .FL(FL)) filter_layer13(
    .CLK(CLK), 
    .RESET(RESET), 
    .acc(layer_1_acc), 
    .mode(layer_1_mode), 
    .map(layer_1_map), 
    .activation_derivative(act_der_layer_1[2]), 
    .in11(in1_layer_1[2 * PATCH_LENGTH + 0]), 
    .in12(in1_layer_1[2 * PATCH_LENGTH + 1]), 
    .in13(in1_layer_1[2 * PATCH_LENGTH + 2]), 
    .in14(in1_layer_1[2 * PATCH_LENGTH + 3]), 
    .in15(in1_layer_1[2 * PATCH_LENGTH + 4]), 
    .in16(in1_layer_1[2 * PATCH_LENGTH + 5]), 
    .in17(in1_layer_1[2 * PATCH_LENGTH + 6]), 
    .in18(in1_layer_1[2 * PATCH_LENGTH + 7]), 
    .in19(in1_layer_1[2 * PATCH_LENGTH + 8]), 
    .in21(in2_layer_1[2 * PATCH_LENGTH + 0]),
    .in22(in2_layer_1[2 * PATCH_LENGTH + 1]),
    .in23(in2_layer_1[2 * PATCH_LENGTH + 2]),
    .in24(in2_layer_1[2 * PATCH_LENGTH + 3]),
    .in25(in2_layer_1[2 * PATCH_LENGTH + 4]),
    .in26(in2_layer_1[2 * PATCH_LENGTH + 5]),
    .in27(in2_layer_1[2 * PATCH_LENGTH + 6]),
    .in28(in2_layer_1[2 * PATCH_LENGTH + 7]),
    .in29(in2_layer_1[2 * PATCH_LENGTH + 8]),
    .out1(out1_layer_1[2]), 
    .out2(out2_layer_1[2])
);

filter3x3 #(.KERNEL_SIZE(KERNEL_SIZE), .DEPTH(INPUT_DEPTH), .LEARNING_RATE(LEARNING_RATE), .WL(WL), .FL(FL)) filter_layer14(
    .CLK(CLK), 
    .RESET(RESET), 
    .acc(layer_1_acc), 
    .mode(layer_1_mode), 
    .map(layer_1_map), 
    .activation_derivative(act_der_layer_1[3]), 
    .in11(in1_layer_1[3 * PATCH_LENGTH + 0]), 
    .in12(in1_layer_1[3 * PATCH_LENGTH + 1]), 
    .in13(in1_layer_1[3 * PATCH_LENGTH + 2]), 
    .in14(in1_layer_1[3 * PATCH_LENGTH + 3]), 
    .in15(in1_layer_1[3 * PATCH_LENGTH + 4]), 
    .in16(in1_layer_1[3 * PATCH_LENGTH + 5]), 
    .in17(in1_layer_1[3 * PATCH_LENGTH + 6]), 
    .in18(in1_layer_1[3 * PATCH_LENGTH + 7]), 
    .in19(in1_layer_1[3 * PATCH_LENGTH + 8]), 
    .in21(in2_layer_1[3 * PATCH_LENGTH + 0]),
    .in22(in2_layer_1[3 * PATCH_LENGTH + 1]),
    .in23(in2_layer_1[3 * PATCH_LENGTH + 2]),
    .in24(in2_layer_1[3 * PATCH_LENGTH + 3]),
    .in25(in2_layer_1[3 * PATCH_LENGTH + 4]),
    .in26(in2_layer_1[3 * PATCH_LENGTH + 5]),
    .in27(in2_layer_1[3 * PATCH_LENGTH + 6]),
    .in28(in2_layer_1[3 * PATCH_LENGTH + 7]),
    .in29(in2_layer_1[3 * PATCH_LENGTH + 8]),
    .out1(out1_layer_1[3]), 
    .out2(out2_layer_1[3])
);

pool2x2 #(.WL(WL), .FL(FL)) pool11(
    .in1(in_layer_2[0]),
    .in2(in_layer_2[1]),
    .in3(in_layer_2[2]),
    .in4(in_layer_2[3]),
    .out(out1_layer_2[0])
);

pool2x2 #(.WL(WL), .FL(FL)) pool12(
    .in1(in_layer_2[1 * POOL_LENGTH + 0]),
    .in2(in_layer_2[1 * POOL_LENGTH + 1]),
    .in3(in_layer_2[1 * POOL_LENGTH + 2]),
    .in4(in_layer_2[1 * POOL_LENGTH + 3]),
    .out(out1_layer_2[1])
);

pool2x2 #(.WL(WL), .FL(FL)) pool13(
    .in1(in_layer_2[2 * POOL_LENGTH + 0]),
    .in2(in_layer_2[2 * POOL_LENGTH + 1]),
    .in3(in_layer_2[2 * POOL_LENGTH + 2]),
    .in4(in_layer_2[2 * POOL_LENGTH + 3]),
    .out(out1_layer_2[2])
);

pool2x2 #(.WL(WL), .FL(FL)) pool14(
    .in1(in_layer_2[3 * POOL_LENGTH + 0]),
    .in2(in_layer_2[3 * POOL_LENGTH + 1]),
    .in3(in_layer_2[3 * POOL_LENGTH + 2]),
    .in4(in_layer_2[3 * POOL_LENGTH + 3]),
    .out(out1_layer_2[3])
);

sigma_prime #(.WL(WL), .FL(FL)) act_der_layer_2_1(
    .in(out1_layer_2[0]),
    .out(out2_layer_2[0])
);

sigma_prime #(.WL(WL), .FL(FL)) act_der_layer_2_2(
    .in(out1_layer_2[1]),
    .out(out2_layer_2[1])
);

sigma_prime #(.WL(WL), .FL(FL)) act_der_layer_2_3(
    .in(out1_layer_2[2]),
    .out(out2_layer_2[3])
);

sigma_prime #(.WL(WL), .FL(FL)) act_der_layer_2_4(
    .in(out1_layer_2[3]),
    .out(out2_layer_2[4])
);

fc16x1 #(.WL(WL), .FL(WL), .PATCH_LENGTH(LAYER_2_WIDTH)) fc1(
    .CLK(CLK),
    .RESET(RESET),
    .in11(in1_layer_3[0]),
    .in12(in1_layer_3[1]),
    .in13(in1_layer_3[2]),
    .in14(in1_layer_3[3]),
    .in15(in1_layer_3[4]),
    .in16(in1_layer_3[5]),
    .in17(in1_layer_3[6]),
    .in18(in1_layer_3[7]),
    .in19(in1_layer_3[8]),
    .in110(in1_layer_3[9]),
    .in111(in1_layer_3[10]),
    .in112(in1_layer_3[11]),
    .in113(in1_layer_3[12]),
    .in114(in1_layer_3[13]),
    .in115(in1_layer_3[14]),
    .in116(in1_layer_3[15]),
    .in21(in2_layer_3[0]),
    .in22(in2_layer_3[1]),
    .in23(in2_layer_3[2]),
    .in24(in2_layer_3[3]),
    .in25(in2_layer_3[4]),
    .in26(in2_layer_3[5]),
    .in27(in2_layer_3[6]),
    .in28(in2_layer_3[7]),
    .in29(in2_layer_3[8]),
    .in210(in2_layer_3[9]),
    .in211(in2_layer_3[10]),
    .in212(in2_layer_3[11]),
    .in213(in2_layer_3[12]),
    .in214(in2_layer_3[13]),
    .in215(in2_layer_3[14]),
    .in216(in2_layer_3[15]),
    .out(out1_layer_3[0])
);

fc16x1 #(.WL(WL), .FL(WL), .PATCH_LENGTH(LAYER_2_WIDTH)) fc2(
    .CLK(CLK),
    .RESET(RESET),
    .in11(in1_layer_3[1 * LAYER_2_WIDTH + 0]),
    .in12(in1_layer_3[1 * LAYER_2_WIDTH + 1]),
    .in13(in1_layer_3[1 * LAYER_2_WIDTH + 2]),
    .in14(in1_layer_3[1 * LAYER_2_WIDTH + 3]),
    .in15(in1_layer_3[1 * LAYER_2_WIDTH + 4]),
    .in16(in1_layer_3[1 * LAYER_2_WIDTH + 5]),
    .in17(in1_layer_3[1 * LAYER_2_WIDTH + 6]),
    .in18(in1_layer_3[1 * LAYER_2_WIDTH + 7]),
    .in19(in1_layer_3[1 * LAYER_2_WIDTH + 8]),
    .in110(in1_layer_3[1 * LAYER_2_WIDTH + 9]),
    .in111(in1_layer_3[1 * LAYER_2_WIDTH + 10]),
    .in112(in1_layer_3[1 * LAYER_2_WIDTH + 11]),
    .in113(in1_layer_3[1 * LAYER_2_WIDTH + 12]),
    .in114(in1_layer_3[1 * LAYER_2_WIDTH + 13]),
    .in115(in1_layer_3[1 * LAYER_2_WIDTH + 14]),
    .in116(in1_layer_3[1 * LAYER_2_WIDTH + 15]),
    .in21(in2_layer_3[1 * LAYER_2_WIDTH + 0]),
    .in22(in2_layer_3[1 * LAYER_2_WIDTH + 1]),
    .in23(in2_layer_3[1 * LAYER_2_WIDTH + 2]),
    .in24(in2_layer_3[1 * LAYER_2_WIDTH + 3]),
    .in25(in2_layer_3[1 * LAYER_2_WIDTH + 4]),
    .in26(in2_layer_3[1 * LAYER_2_WIDTH + 5]),
    .in27(in2_layer_3[1 * LAYER_2_WIDTH + 6]),
    .in28(in2_layer_3[1 * LAYER_2_WIDTH + 7]),
    .in29(in2_layer_3[1 * LAYER_2_WIDTH + 8]),
    .in210(in2_layer_3[1 * LAYER_2_WIDTH + 9]),
    .in211(in2_layer_3[1 * LAYER_2_WIDTH + 10]),
    .in212(in2_layer_3[1 * LAYER_2_WIDTH + 11]),
    .in213(in2_layer_3[1 * LAYER_2_WIDTH + 12]),
    .in214(in2_layer_3[1 * LAYER_2_WIDTH + 13]),
    .in215(in2_layer_3[1 * LAYER_2_WIDTH + 14]),
    .in216(in2_layer_3[1 * LAYER_2_WIDTH + 15]),
    .out(out1_layer_3[1])
);

fc16x1 #(.WL(WL), .FL(WL), .PATCH_LENGTH(LAYER_2_WIDTH)) fc3(
    .CLK(CLK),
    .RESET(RESET),
    .in11(in1_layer_3[2 * LAYER_2_WIDTH + 0]),
    .in12(in1_layer_3[2 * LAYER_2_WIDTH + 1]),
    .in13(in1_layer_3[2 * LAYER_2_WIDTH + 2]),
    .in14(in1_layer_3[2 * LAYER_2_WIDTH + 3]),
    .in15(in1_layer_3[2 * LAYER_2_WIDTH + 4]),
    .in16(in1_layer_3[2 * LAYER_2_WIDTH + 5]),
    .in17(in1_layer_3[2 * LAYER_2_WIDTH + 6]),
    .in18(in1_layer_3[2 * LAYER_2_WIDTH + 7]),
    .in19(in1_layer_3[2 * LAYER_2_WIDTH + 8]),
    .in110(in1_layer_3[2 * LAYER_2_WIDTH + 9]),
    .in111(in1_layer_3[2 * LAYER_2_WIDTH + 10]),
    .in112(in1_layer_3[2 * LAYER_2_WIDTH + 11]),
    .in113(in1_layer_3[2 * LAYER_2_WIDTH + 12]),
    .in114(in1_layer_3[2 * LAYER_2_WIDTH + 13]),
    .in115(in1_layer_3[2 * LAYER_2_WIDTH + 14]),
    .in116(in1_layer_3[2 * LAYER_2_WIDTH + 15]),
    .in21(in2_layer_3[2 * LAYER_2_WIDTH + 0]),
    .in22(in2_layer_3[2 * LAYER_2_WIDTH + 1]),
    .in23(in2_layer_3[2 * LAYER_2_WIDTH + 2]),
    .in24(in2_layer_3[2 * LAYER_2_WIDTH + 3]),
    .in25(in2_layer_3[2 * LAYER_2_WIDTH + 4]),
    .in26(in2_layer_3[2 * LAYER_2_WIDTH + 5]),
    .in27(in2_layer_3[2 * LAYER_2_WIDTH + 6]),
    .in28(in2_layer_3[2 * LAYER_2_WIDTH + 7]),
    .in29(in2_layer_3[2 * LAYER_2_WIDTH + 8]),
    .in210(in2_layer_3[2 * LAYER_2_WIDTH + 9]),
    .in211(in2_layer_3[2 * LAYER_2_WIDTH + 10]),
    .in212(in2_layer_3[2 * LAYER_2_WIDTH + 11]),
    .in213(in2_layer_3[2 * LAYER_2_WIDTH + 12]),
    .in214(in2_layer_3[2 * LAYER_2_WIDTH + 13]),
    .in215(in2_layer_3[2 * LAYER_2_WIDTH + 14]),
    .in216(in2_layer_3[2 * LAYER_2_WIDTH + 15]),
    .out(out1_layer_3[2])
);

fc16x1 #(.WL(WL), .FL(WL), .PATCH_LENGTH(LAYER_2_WIDTH)) fc4(
    .CLK(CLK),
    .RESET(RESET),
    .in11(in1_layer_3[3 * LAYER_2_WIDTH + 0]),
    .in12(in1_layer_3[3 * LAYER_2_WIDTH + 1]),
    .in13(in1_layer_3[3 * LAYER_2_WIDTH + 2]),
    .in14(in1_layer_3[3 * LAYER_2_WIDTH + 3]),
    .in15(in1_layer_3[3 * LAYER_2_WIDTH + 4]),
    .in16(in1_layer_3[3 * LAYER_2_WIDTH + 5]),
    .in17(in1_layer_3[3 * LAYER_2_WIDTH + 6]),
    .in18(in1_layer_3[3 * LAYER_2_WIDTH + 7]),
    .in19(in1_layer_3[3 * LAYER_2_WIDTH + 8]),
    .in110(in1_layer_3[3 * LAYER_2_WIDTH + 9]),
    .in111(in1_layer_3[3 * LAYER_2_WIDTH + 10]),
    .in112(in1_layer_3[3 * LAYER_2_WIDTH + 11]),
    .in113(in1_layer_3[3 * LAYER_2_WIDTH + 12]),
    .in114(in1_layer_3[3 * LAYER_2_WIDTH + 13]),
    .in115(in1_layer_3[3 * LAYER_2_WIDTH + 14]),
    .in116(in1_layer_3[3 * LAYER_2_WIDTH + 15]),
    .in21(in2_layer_3[3 * LAYER_2_WIDTH + 0]),
    .in22(in2_layer_3[3 * LAYER_2_WIDTH + 1]),
    .in23(in2_layer_3[3 * LAYER_2_WIDTH + 2]),
    .in24(in2_layer_3[3 * LAYER_2_WIDTH + 3]),
    .in25(in2_layer_3[3 * LAYER_2_WIDTH + 4]),
    .in26(in2_layer_3[3 * LAYER_2_WIDTH + 5]),
    .in27(in2_layer_3[3 * LAYER_2_WIDTH + 6]),
    .in28(in2_layer_3[3 * LAYER_2_WIDTH + 7]),
    .in29(in2_layer_3[3 * LAYER_2_WIDTH + 8]),
    .in210(in2_layer_3[3 * LAYER_2_WIDTH + 9]),
    .in211(in2_layer_3[3 * LAYER_2_WIDTH + 10]),
    .in212(in2_layer_3[3 * LAYER_2_WIDTH + 11]),
    .in213(in2_layer_3[3 * LAYER_2_WIDTH + 12]),
    .in214(in2_layer_3[3 * LAYER_2_WIDTH + 13]),
    .in215(in2_layer_3[3 * LAYER_2_WIDTH + 14]),
    .in216(in2_layer_3[3 * LAYER_2_WIDTH + 15]),
    .out(out1_layer_3[3])
);

fc16x1 #(.WL(WL), .FL(WL), .PATCH_LENGTH(LAYER_2_WIDTH)) fc5(
    .CLK(CLK),
    .RESET(RESET),
    .in11(in1_layer_3[4 * LAYER_2_WIDTH + 0]),
    .in12(in1_layer_3[4 * LAYER_2_WIDTH + 1]),
    .in13(in1_layer_3[4 * LAYER_2_WIDTH + 2]),
    .in14(in1_layer_3[4 * LAYER_2_WIDTH + 3]),
    .in15(in1_layer_3[4 * LAYER_2_WIDTH + 4]),
    .in16(in1_layer_3[4 * LAYER_2_WIDTH + 5]),
    .in17(in1_layer_3[4 * LAYER_2_WIDTH + 6]),
    .in18(in1_layer_3[4 * LAYER_2_WIDTH + 7]),
    .in19(in1_layer_3[4 * LAYER_2_WIDTH + 8]),
    .in110(in1_layer_3[4 * LAYER_2_WIDTH + 9]),
    .in111(in1_layer_3[4 * LAYER_2_WIDTH + 10]),
    .in112(in1_layer_3[4 * LAYER_2_WIDTH + 11]),
    .in113(in1_layer_3[4 * LAYER_2_WIDTH + 12]),
    .in114(in1_layer_3[4 * LAYER_2_WIDTH + 13]),
    .in115(in1_layer_3[4 * LAYER_2_WIDTH + 14]),
    .in116(in1_layer_3[4 * LAYER_2_WIDTH + 15]),
    .in21(in2_layer_3[4 * LAYER_2_WIDTH + 0]),
    .in22(in2_layer_3[4 * LAYER_2_WIDTH + 1]),
    .in23(in2_layer_3[4 * LAYER_2_WIDTH + 2]),
    .in24(in2_layer_3[4 * LAYER_2_WIDTH + 3]),
    .in25(in2_layer_3[4 * LAYER_2_WIDTH + 4]),
    .in26(in2_layer_3[4 * LAYER_2_WIDTH + 5]),
    .in27(in2_layer_3[4 * LAYER_2_WIDTH + 6]),
    .in28(in2_layer_3[4 * LAYER_2_WIDTH + 7]),
    .in29(in2_layer_3[4 * LAYER_2_WIDTH + 8]),
    .in210(in2_layer_3[4 * LAYER_2_WIDTH + 9]),
    .in211(in2_layer_3[4 * LAYER_2_WIDTH + 10]),
    .in212(in2_layer_3[4 * LAYER_2_WIDTH + 11]),
    .in213(in2_layer_3[4 * LAYER_2_WIDTH + 12]),
    .in214(in2_layer_3[4 * LAYER_2_WIDTH + 13]),
    .in215(in2_layer_3[4 * LAYER_2_WIDTH + 14]),
    .in216(in2_layer_3[4 * LAYER_2_WIDTH + 15]),
    .out(out1_layer_3[4])
);

fc16x1 #(.WL(WL), .FL(WL), .PATCH_LENGTH(LAYER_2_WIDTH)) fc6(
    .CLK(CLK),
    .RESET(RESET),
    .in11(in1_layer_3[5 * LAYER_2_WIDTH + 0]),
    .in12(in1_layer_3[5 * LAYER_2_WIDTH + 1]),
    .in13(in1_layer_3[5 * LAYER_2_WIDTH + 2]),
    .in14(in1_layer_3[5 * LAYER_2_WIDTH + 3]),
    .in15(in1_layer_3[5 * LAYER_2_WIDTH + 4]),
    .in16(in1_layer_3[5 * LAYER_2_WIDTH + 5]),
    .in17(in1_layer_3[5 * LAYER_2_WIDTH + 6]),
    .in18(in1_layer_3[5 * LAYER_2_WIDTH + 7]),
    .in19(in1_layer_3[5 * LAYER_2_WIDTH + 8]),
    .in110(in1_layer_3[5 * LAYER_2_WIDTH + 9]),
    .in111(in1_layer_3[5 * LAYER_2_WIDTH + 10]),
    .in112(in1_layer_3[5 * LAYER_2_WIDTH + 11]),
    .in113(in1_layer_3[5 * LAYER_2_WIDTH + 12]),
    .in114(in1_layer_3[5 * LAYER_2_WIDTH + 13]),
    .in115(in1_layer_3[5 * LAYER_2_WIDTH + 14]),
    .in116(in1_layer_3[5 * LAYER_2_WIDTH + 15]),
    .in21(in2_layer_3[5 * LAYER_2_WIDTH + 0]),
    .in22(in2_layer_3[5 * LAYER_2_WIDTH + 1]),
    .in23(in2_layer_3[5 * LAYER_2_WIDTH + 2]),
    .in24(in2_layer_3[5 * LAYER_2_WIDTH + 3]),
    .in25(in2_layer_3[5 * LAYER_2_WIDTH + 4]),
    .in26(in2_layer_3[5 * LAYER_2_WIDTH + 5]),
    .in27(in2_layer_3[5 * LAYER_2_WIDTH + 6]),
    .in28(in2_layer_3[5 * LAYER_2_WIDTH + 7]),
    .in29(in2_layer_3[5 * LAYER_2_WIDTH + 8]),
    .in210(in2_layer_3[5 * LAYER_2_WIDTH + 9]),
    .in211(in2_layer_3[5 * LAYER_2_WIDTH + 10]),
    .in212(in2_layer_3[5 * LAYER_2_WIDTH + 11]),
    .in213(in2_layer_3[5 * LAYER_2_WIDTH + 12]),
    .in214(in2_layer_3[5 * LAYER_2_WIDTH + 13]),
    .in215(in2_layer_3[5 * LAYER_2_WIDTH + 14]),
    .in216(in2_layer_3[5 * LAYER_2_WIDTH + 15]),
    .out(out1_layer_3[5])
);

fc16x1 #(.WL(WL), .FL(WL), .PATCH_LENGTH(LAYER_2_WIDTH)) fc7(
    .CLK(CLK),
    .RESET(RESET),
    .in11(in1_layer_3[6 * LAYER_2_WIDTH + 0]),
    .in12(in1_layer_3[6 * LAYER_2_WIDTH + 1]),
    .in13(in1_layer_3[6 * LAYER_2_WIDTH + 2]),
    .in14(in1_layer_3[6 * LAYER_2_WIDTH + 3]),
    .in15(in1_layer_3[6 * LAYER_2_WIDTH + 4]),
    .in16(in1_layer_3[6 * LAYER_2_WIDTH + 5]),
    .in17(in1_layer_3[6 * LAYER_2_WIDTH + 6]),
    .in18(in1_layer_3[6 * LAYER_2_WIDTH + 7]),
    .in19(in1_layer_3[6 * LAYER_2_WIDTH + 8]),
    .in110(in1_layer_3[6 * LAYER_2_WIDTH + 9]),
    .in111(in1_layer_3[6 * LAYER_2_WIDTH + 10]),
    .in112(in1_layer_3[6 * LAYER_2_WIDTH + 11]),
    .in113(in1_layer_3[6 * LAYER_2_WIDTH + 12]),
    .in114(in1_layer_3[6 * LAYER_2_WIDTH + 13]),
    .in115(in1_layer_3[6 * LAYER_2_WIDTH + 14]),
    .in116(in1_layer_3[6 * LAYER_2_WIDTH + 15]),
    .in21(in2_layer_3[6 * LAYER_2_WIDTH + 0]),
    .in22(in2_layer_3[6 * LAYER_2_WIDTH + 1]),
    .in23(in2_layer_3[6 * LAYER_2_WIDTH + 2]),
    .in24(in2_layer_3[6 * LAYER_2_WIDTH + 3]),
    .in25(in2_layer_3[6 * LAYER_2_WIDTH + 4]),
    .in26(in2_layer_3[6 * LAYER_2_WIDTH + 5]),
    .in27(in2_layer_3[6 * LAYER_2_WIDTH + 6]),
    .in28(in2_layer_3[6 * LAYER_2_WIDTH + 7]),
    .in29(in2_layer_3[6 * LAYER_2_WIDTH + 8]),
    .in210(in2_layer_3[6 * LAYER_2_WIDTH + 9]),
    .in211(in2_layer_3[6 * LAYER_2_WIDTH + 10]),
    .in212(in2_layer_3[6 * LAYER_2_WIDTH + 11]),
    .in213(in2_layer_3[6 * LAYER_2_WIDTH + 12]),
    .in214(in2_layer_3[6 * LAYER_2_WIDTH + 13]),
    .in215(in2_layer_3[6 * LAYER_2_WIDTH + 14]),
    .in216(in2_layer_3[6 * LAYER_2_WIDTH + 15]),
    .out(out1_layer_3[6])
);

fc16x1 #(.WL(WL), .FL(WL), .PATCH_LENGTH(LAYER_2_WIDTH)) fc8(
    .CLK(CLK),
    .RESET(RESET),
    .in11(in1_layer_3[7 * LAYER_2_WIDTH + 0]),
    .in12(in1_layer_3[7 * LAYER_2_WIDTH + 1]),
    .in13(in1_layer_3[7 * LAYER_2_WIDTH + 2]),
    .in14(in1_layer_3[7 * LAYER_2_WIDTH + 3]),
    .in15(in1_layer_3[7 * LAYER_2_WIDTH + 4]),
    .in16(in1_layer_3[7 * LAYER_2_WIDTH + 5]),
    .in17(in1_layer_3[7 * LAYER_2_WIDTH + 6]),
    .in18(in1_layer_3[7 * LAYER_2_WIDTH + 7]),
    .in19(in1_layer_3[7 * LAYER_2_WIDTH + 8]),
    .in110(in1_layer_3[7 * LAYER_2_WIDTH + 9]),
    .in111(in1_layer_3[7 * LAYER_2_WIDTH + 10]),
    .in112(in1_layer_3[7 * LAYER_2_WIDTH + 11]),
    .in113(in1_layer_3[7 * LAYER_2_WIDTH + 12]),
    .in114(in1_layer_3[7 * LAYER_2_WIDTH + 13]),
    .in115(in1_layer_3[7 * LAYER_2_WIDTH + 14]),
    .in116(in1_layer_3[7 * LAYER_2_WIDTH + 15]),
    .in21(in2_layer_3[7 * LAYER_2_WIDTH + 0]),
    .in22(in2_layer_3[7 * LAYER_2_WIDTH + 1]),
    .in23(in2_layer_3[7 * LAYER_2_WIDTH + 2]),
    .in24(in2_layer_3[7 * LAYER_2_WIDTH + 3]),
    .in25(in2_layer_3[7 * LAYER_2_WIDTH + 4]),
    .in26(in2_layer_3[7 * LAYER_2_WIDTH + 5]),
    .in27(in2_layer_3[7 * LAYER_2_WIDTH + 6]),
    .in28(in2_layer_3[7 * LAYER_2_WIDTH + 7]),
    .in29(in2_layer_3[7 * LAYER_2_WIDTH + 8]),
    .in210(in2_layer_3[7 * LAYER_2_WIDTH + 9]),
    .in211(in2_layer_3[7 * LAYER_2_WIDTH + 10]),
    .in212(in2_layer_3[7 * LAYER_2_WIDTH + 11]),
    .in213(in2_layer_3[7 * LAYER_2_WIDTH + 12]),
    .in214(in2_layer_3[7 * LAYER_2_WIDTH + 13]),
    .in215(in2_layer_3[7 * LAYER_2_WIDTH + 14]),
    .in216(in2_layer_3[7 * LAYER_2_WIDTH + 15]),
    .out(out1_layer_3[7])
);

fc16x1 #(.WL(WL), .FL(WL), .PATCH_LENGTH(LAYER_2_WIDTH)) fc9(
    .CLK(CLK),
    .RESET(RESET),
    .in11(in1_layer_3[8 * LAYER_2_WIDTH + 0]),
    .in12(in1_layer_3[8 * LAYER_2_WIDTH + 1]),
    .in13(in1_layer_3[8 * LAYER_2_WIDTH + 2]),
    .in14(in1_layer_3[8 * LAYER_2_WIDTH + 3]),
    .in15(in1_layer_3[8 * LAYER_2_WIDTH + 4]),
    .in16(in1_layer_3[8 * LAYER_2_WIDTH + 5]),
    .in17(in1_layer_3[8 * LAYER_2_WIDTH + 6]),
    .in18(in1_layer_3[8 * LAYER_2_WIDTH + 7]),
    .in19(in1_layer_3[8 * LAYER_2_WIDTH + 8]),
    .in110(in1_layer_3[8 * LAYER_2_WIDTH + 9]),
    .in111(in1_layer_3[8 * LAYER_2_WIDTH + 10]),
    .in112(in1_layer_3[8 * LAYER_2_WIDTH + 11]),
    .in113(in1_layer_3[8 * LAYER_2_WIDTH + 12]),
    .in114(in1_layer_3[8 * LAYER_2_WIDTH + 13]),
    .in115(in1_layer_3[8 * LAYER_2_WIDTH + 14]),
    .in116(in1_layer_3[8 * LAYER_2_WIDTH + 15]),
    .in21(in2_layer_3[8 * LAYER_2_WIDTH + 0]),
    .in22(in2_layer_3[8 * LAYER_2_WIDTH + 1]),
    .in23(in2_layer_3[8 * LAYER_2_WIDTH + 2]),
    .in24(in2_layer_3[8 * LAYER_2_WIDTH + 3]),
    .in25(in2_layer_3[8 * LAYER_2_WIDTH + 4]),
    .in26(in2_layer_3[8 * LAYER_2_WIDTH + 5]),
    .in27(in2_layer_3[8 * LAYER_2_WIDTH + 6]),
    .in28(in2_layer_3[8 * LAYER_2_WIDTH + 7]),
    .in29(in2_layer_3[8 * LAYER_2_WIDTH + 8]),
    .in210(in2_layer_3[8 * LAYER_2_WIDTH + 9]),
    .in211(in2_layer_3[8 * LAYER_2_WIDTH + 10]),
    .in212(in2_layer_3[8 * LAYER_2_WIDTH + 11]),
    .in213(in2_layer_3[8 * LAYER_2_WIDTH + 12]),
    .in214(in2_layer_3[8 * LAYER_2_WIDTH + 13]),
    .in215(in2_layer_3[8 * LAYER_2_WIDTH + 14]),
    .in216(in2_layer_3[8 * LAYER_2_WIDTH + 15]),
    .out(out1_layer_3[8])
);

fc16x1 #(.WL(WL), .FL(WL), .PATCH_LENGTH(LAYER_2_WIDTH)) fc10(
    .CLK(CLK),
    .RESET(RESET),
    .in11(in1_layer_3[9 * LAYER_2_WIDTH + 0]),
    .in12(in1_layer_3[9 * LAYER_2_WIDTH + 1]),
    .in13(in1_layer_3[9 * LAYER_2_WIDTH + 2]),
    .in14(in1_layer_3[9 * LAYER_2_WIDTH + 3]),
    .in15(in1_layer_3[9 * LAYER_2_WIDTH + 4]),
    .in16(in1_layer_3[9 * LAYER_2_WIDTH + 5]),
    .in17(in1_layer_3[9 * LAYER_2_WIDTH + 6]),
    .in18(in1_layer_3[9 * LAYER_2_WIDTH + 7]),
    .in19(in1_layer_3[9 * LAYER_2_WIDTH + 8]),
    .in110(in1_layer_3[9 * LAYER_2_WIDTH + 9]),
    .in111(in1_layer_3[9 * LAYER_2_WIDTH + 10]),
    .in112(in1_layer_3[9 * LAYER_2_WIDTH + 11]),
    .in113(in1_layer_3[9 * LAYER_2_WIDTH + 12]),
    .in114(in1_layer_3[9 * LAYER_2_WIDTH + 13]),
    .in115(in1_layer_3[9 * LAYER_2_WIDTH + 14]),
    .in116(in1_layer_3[9 * LAYER_2_WIDTH + 15]),
    .in21(in2_layer_3[9 * LAYER_2_WIDTH + 0]),
    .in22(in2_layer_3[9 * LAYER_2_WIDTH + 1]),
    .in23(in2_layer_3[9 * LAYER_2_WIDTH + 2]),
    .in24(in2_layer_3[9 * LAYER_2_WIDTH + 3]),
    .in25(in2_layer_3[9 * LAYER_2_WIDTH + 4]),
    .in26(in2_layer_3[9 * LAYER_2_WIDTH + 5]),
    .in27(in2_layer_3[9 * LAYER_2_WIDTH + 6]),
    .in28(in2_layer_3[9 * LAYER_2_WIDTH + 7]),
    .in29(in2_layer_3[9 * LAYER_2_WIDTH + 8]),
    .in210(in2_layer_3[9 * LAYER_2_WIDTH + 9]),
    .in211(in2_layer_3[9 * LAYER_2_WIDTH + 10]),
    .in212(in2_layer_3[9 * LAYER_2_WIDTH + 11]),
    .in213(in2_layer_3[9 * LAYER_2_WIDTH + 12]),
    .in214(in2_layer_3[9 * LAYER_2_WIDTH + 13]),
    .in215(in2_layer_3[9 * LAYER_2_WIDTH + 14]),
    .in216(in2_layer_3[9 * LAYER_2_WIDTH + 15]),
    .out(out1_layer_3[9])
);

add_with_saturate #(.WL(WL), .FL(FL)) acc1(
    .a(acc_in1_layer_3[0]),
    .b(acc_in2_layer_3[0]),
    .out(acc_out_layer_3[0])
);

add_with_saturate #(.WL(WL), .FL(FL)) acc2(
    .a(acc_in1_layer_3[1]),
    .b(acc_in2_layer_3[1]),
    .out(acc_out_layer_3[1])
);

add_with_saturate #(.WL(WL), .FL(FL)) acc3(
    .a(acc_in1_layer_3[2]),
    .b(acc_in2_layer_3[2]),
    .out(acc_out_layer_3[2])
);

add_with_saturate #(.WL(WL), .FL(FL)) acc4(
    .a(acc_in1_layer_3[3]),
    .b(acc_in2_layer_3[3]),
    .out(acc_out_layer_3[3])
);

add_with_saturate #(.WL(WL), .FL(FL)) acc5(
    .a(acc_in1_layer_3[4]),
    .b(acc_in2_layer_3[4]),
    .out(acc_out_layer_3[4])
);

add_with_saturate #(.WL(WL), .FL(FL)) acc6(
    .a(acc_in1_layer_3[5]),
    .b(acc_in2_layer_3[5]),
    .out(acc_out_layer_3[5])
);

add_with_saturate #(.WL(WL), .FL(FL)) acc7(
    .a(acc_in1_layer_3[6]),
    .b(acc_in2_layer_3[6]),
    .out(acc_out_layer_3[6])
);

add_with_saturate #(.WL(WL), .FL(FL)) acc8(
    .a(acc_in1_layer_3[7]),
    .b(acc_in2_layer_3[7]),
    .out(acc_out_layer_3[7])
);

add_with_saturate #(.WL(WL), .FL(FL)) acc9(
    .a(acc_in1_layer_3[8]),
    .b(acc_in2_layer_3[8]),
    .out(acc_out_layer_3[8])
);

add_with_saturate #(.WL(WL), .FL(FL)) acc10(
    .a(acc_in1_layer_3[9]),
    .b(acc_in2_layer_3[9]),
    .out(acc_out_layer_3[9])
);

sigma #(.WL(WL)) activator1(
    .in(out_layer_3[0]),
    .out(out1)
);

sigma_prime #(.WL(WL), .FL(FL)) derivative1(
    .in(out_layer_3[0]),
    .out(der_layer_3[0])
);

sigma #(.WL(WL)) activator2(
    .in(out_layer_3[1]),
    .out(out2)
);

sigma_prime #(.WL(WL), .FL(FL)) derivative2(
    .in(out_layer_3[1]),
    .out(der_layer_3[1])
);

sigma #(.WL(WL)) activator3(
    .in(out_layer_3[2]),
    .out(out3)
);

sigma_prime #(.WL(WL), .FL(FL)) derivative3(
    .in(out_layer_3[2]),
    .out(der_layer_3[2])
);

sigma #(.WL(WL)) activator4(
    .in(out_layer_3[3]),
    .out(out4)
);

sigma_prime #(.WL(WL), .FL(FL)) derivative4(
    .in(out_layer_3[3]),
    .out(der_layer_3[3])
);

sigma #(.WL(WL)) activator5(
    .in(out_layer_3[4]),
    .out(out5)
);

sigma_prime #(.WL(WL), .FL(FL)) derivative5(
    .in(out_layer_3[4]),
    .out(der_layer_3[4])
);

sigma #(.WL(WL)) activator6(
    .in(out_layer_3[5]),
    .out(out6)
);

sigma_prime #(.WL(WL), .FL(FL)) derivative6(
    .in(out_layer_3[5]),
    .out(der_layer_3[5])
);

sigma #(.WL(WL)) activator7(
    .in(out_layer_3[6]),
    .out(out7)
);

sigma_prime #(.WL(WL), .FL(FL)) derivative7(
    .in(out_layer_3[6]),
    .out(der_layer_3[6])
);

sigma #(.WL(WL)) activator8(
    .in(out_layer_3[7]),
    .out(out8)
);

sigma_prime #(.WL(WL), .FL(FL)) derivative8(
    .in(out_layer_3[7]),
    .out(der_layer_3[7])
);

sigma #(.WL(WL)) activator9(
    .in(out_layer_3[8]),
    .out(out9)
);

sigma_prime #(.WL(WL), .FL(FL)) derivative9(
    .in(out_layer_3[8]),
    .out(der_layer_3[8])
);

sigma #(.WL(WL)) activator10(
    .in(out_layer_3[9]),
    .out(out10)
);

sigma_prime #(.WL(WL), .FL(FL)) derivative10(
    .in(out_layer_3[9]),
    .out(der_layer_3[9])
);

sub_with_saturate #(.WL(WL), .FL(FL)) error_sub1(
    .a(out1),
    .b(label1),
    .out(classification_error[0])
);

sub_with_saturate #(.WL(WL), .FL(FL)) error_sub2(
    .a(out2),
    .b(label2),
    .out(classification_error[1])
);

sub_with_saturate #(.WL(WL), .FL(FL)) error_sub3(
    .a(out3),
    .b(label3),
    .out(classification_error[2])
);

sub_with_saturate #(.WL(WL), .FL(FL)) error_sub4(
    .a(out4),
    .b(label4),
    .out(classification_error[3])
);

sub_with_saturate #(.WL(WL), .FL(FL)) error_sub5(
    .a(out5),
    .b(label5),
    .out(classification_error[4])
);

sub_with_saturate #(.WL(WL), .FL(FL)) error_sub6(
    .a(out6),
    .b(label6),
    .out(classification_error[5])
);

sub_with_saturate #(.WL(WL), .FL(FL)) error_sub7(
    .a(out7),
    .b(label7),
    .out(classification_error[6])
);

sub_with_saturate #(.WL(WL), .FL(FL)) error_sub8(
    .a(out8),
    .b(label8),
    .out(classification_error[7])
);

sub_with_saturate #(.WL(WL), .FL(FL)) error_sub9(
    .a(out9),
    .b(label9),
    .out(classification_error[8])
);

sub_with_saturate #(.WL(WL), .FL(FL)) error_sub10(
    .a(out10),
    .b(label10),
    .out(classification_error[9])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_error_quantizer1(
    .CLK(CLK),
    .RESET(RESET),
    .in(classification_error[0] * der_layer_3[0]),
    .out(output_error_q[0])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_error_quantizer2(
    .CLK(CLK),
    .RESET(RESET),
    .in(classification_error[1] * der_layer_3[1]),
    .out(output_error_q[1])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_error_quantizer3(
    .CLK(CLK),
    .RESET(RESET),
    .in(classification_error[2] * der_layer_3[2]),
    .out(output_error_q[2])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_error_quantizer4(
    .CLK(CLK),
    .RESET(RESET),
    .in(classification_error[3] * der_layer_3[3]),
    .out(output_error_q[3])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_error_quantizer5(
    .CLK(CLK),
    .RESET(RESET),
    .in(classification_error[4] * der_layer_3[4]),
    .out(output_error_q[4])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_error_quantizer6(
    .CLK(CLK),
    .RESET(RESET),
    .in(classification_error[5] * der_layer_3[5]),
    .out(output_error_q[5])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_error_quantizer7(
    .CLK(CLK),
    .RESET(RESET),
    .in(classification_error[6] * der_layer_3[6]),
    .out(output_error_q[6])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_error_quantizer8(
    .CLK(CLK),
    .RESET(RESET),
    .in(classification_error[7] * der_layer_3[7]),
    .out(output_error_q[7])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_error_quantizer9(
    .CLK(CLK),
    .RESET(RESET),
    .in(classification_error[8] * der_layer_3[8]),
    .out(output_error_q[8])
);

stochastic_quantizer #(.WL(WL), .FL(FL)) output_error_quantizer10(
    .CLK(CLK),
    .RESET(RESET),
    .in(classification_error[9] * der_layer_3[9]),
    .out(output_error_q[9])
);

fc_gr_update16x1 #(.WL(WL), .FL(FL), .LEARNING_RATE(LEARNING_RATE)) fc_bp1(
    .CLK(CLK),
    .RESET(RESET),
    .err_factor(output_error[0]),
    .in1(layer_3_gr_in_patch[0]),
    .in2(layer_3_gr_in_patch[1]),
    .in3(layer_3_gr_in_patch[2]),
    .in4(layer_3_gr_in_patch[3]),
    .in5(layer_3_gr_in_patch[4]),
    .in6(layer_3_gr_in_patch[5]),
    .in7(layer_3_gr_in_patch[6]),
    .in8(layer_3_gr_in_patch[7]),
    .in9(layer_3_gr_in_patch[8]),
    .in10(layer_3_gr_in_patch[9]),
    .in11(layer_3_gr_in_patch[10]),
    .in12(layer_3_gr_in_patch[11]),
    .in13(layer_3_gr_in_patch[12]),
    .in14(layer_3_gr_in_patch[13]),
    .in15(layer_3_gr_in_patch[14]),
    .in16(layer_3_gr_in_patch[15]),
    .wght1(layer_3_weight_patch[0]),
    .wght2(layer_3_weight_patch[1]),
    .wght3(layer_3_weight_patch[2]),
    .wght4(layer_3_weight_patch[3]),
    .wght5(layer_3_weight_patch[4]),
    .wght6(layer_3_weight_patch[5]),
    .wght7(layer_3_weight_patch[6]),
    .wght8(layer_3_weight_patch[7]),
    .wght9(layer_3_weight_patch[8]),
    .wght10(layer_3_weight_patch[9]),
    .wght11(layer_3_weight_patch[10]),
    .wght12(layer_3_weight_patch[11]),
    .wght13(layer_3_weight_patch[12]),
    .wght14(layer_3_weight_patch[13]),
    .wght15(layer_3_weight_patch[14]),
    .wght16(layer_3_weight_patch[15]),
    .out1(layer_3_gr_update_patch[0]),
    .out2(layer_3_gr_update_patch[1]),
    .out3(layer_3_gr_update_patch[2]),
    .out4(layer_3_gr_update_patch[3]),
    .out5(layer_3_gr_update_patch[4]),
    .out6(layer_3_gr_update_patch[5]),
    .out7(layer_3_gr_update_patch[6]),
    .out8(layer_3_gr_update_patch[7]),
    .out9(layer_3_gr_update_patch[8]),
    .out10(layer_3_gr_update_patch[9]),
    .out11(layer_3_gr_update_patch[10]),
    .out12(layer_3_gr_update_patch[11]),
    .out13(layer_3_gr_update_patch[12]),
    .out14(layer_3_gr_update_patch[13]),
    .out15(layer_3_gr_update_patch[14]),
    .out16(layer_3_gr_update_patch[15])
);

fc_gr_update16x1 #(.WL(WL), .FL(FL), .LEARNING_RATE(LEARNING_RATE)) fc_bp2(
    .CLK(CLK),
    .RESET(RESET),
    .err_factor(output_error[1]),
    .in1(layer_3_gr_in_patch[0]),
    .in2(layer_3_gr_in_patch[1]),
    .in3(layer_3_gr_in_patch[2]),
    .in4(layer_3_gr_in_patch[3]),
    .in5(layer_3_gr_in_patch[4]),
    .in6(layer_3_gr_in_patch[5]),
    .in7(layer_3_gr_in_patch[6]),
    .in8(layer_3_gr_in_patch[7]),
    .in9(layer_3_gr_in_patch[8]),
    .in10(layer_3_gr_in_patch[9]),
    .in11(layer_3_gr_in_patch[10]),
    .in12(layer_3_gr_in_patch[11]),
    .in13(layer_3_gr_in_patch[12]),
    .in14(layer_3_gr_in_patch[13]),
    .in15(layer_3_gr_in_patch[14]),
    .in16(layer_3_gr_in_patch[15]),
    .wght1(layer_3_weight_patch[1 * LAYER_2_WIDTH + 0]),
    .wght2(layer_3_weight_patch[1 * LAYER_2_WIDTH + 1]),
    .wght3(layer_3_weight_patch[1 * LAYER_2_WIDTH + 2]),
    .wght4(layer_3_weight_patch[1 * LAYER_2_WIDTH + 3]),
    .wght5(layer_3_weight_patch[1 * LAYER_2_WIDTH + 4]),
    .wght6(layer_3_weight_patch[1 * LAYER_2_WIDTH + 5]),
    .wght7(layer_3_weight_patch[1 * LAYER_2_WIDTH + 6]),
    .wght8(layer_3_weight_patch[1 * LAYER_2_WIDTH + 7]),
    .wght9(layer_3_weight_patch[1 * LAYER_2_WIDTH + 8]),
    .wght10(layer_3_weight_patch[1 * LAYER_2_WIDTH + 9]),
    .wght11(layer_3_weight_patch[1 * LAYER_2_WIDTH + 10]),
    .wght12(layer_3_weight_patch[1 * LAYER_2_WIDTH + 11]),
    .wght13(layer_3_weight_patch[1 * LAYER_2_WIDTH + 12]),
    .wght14(layer_3_weight_patch[1 * LAYER_2_WIDTH + 13]),
    .wght15(layer_3_weight_patch[1 * LAYER_2_WIDTH + 14]),
    .wght16(layer_3_weight_patch[1 * LAYER_2_WIDTH + 15]),
    .out1(layer_3_gr_update_patch[1 * LAYER_2_WIDTH + 0]),
    .out2(layer_3_gr_update_patch[1 * LAYER_2_WIDTH + 1]),
    .out3(layer_3_gr_update_patch[1 * LAYER_2_WIDTH + 2]),
    .out4(layer_3_gr_update_patch[1 * LAYER_2_WIDTH + 3]),
    .out5(layer_3_gr_update_patch[1 * LAYER_2_WIDTH + 4]),
    .out6(layer_3_gr_update_patch[1 * LAYER_2_WIDTH + 5]),
    .out7(layer_3_gr_update_patch[1 * LAYER_2_WIDTH + 6]),
    .out8(layer_3_gr_update_patch[1 * LAYER_2_WIDTH + 7]),
    .out9(layer_3_gr_update_patch[1 * LAYER_2_WIDTH + 8]),
    .out10(layer_3_gr_update_patch[1 * LAYER_2_WIDTH + 9]),
    .out11(layer_3_gr_update_patch[1 * LAYER_2_WIDTH + 10]),
    .out12(layer_3_gr_update_patch[1 * LAYER_2_WIDTH + 11]),
    .out13(layer_3_gr_update_patch[1 * LAYER_2_WIDTH + 12]),
    .out14(layer_3_gr_update_patch[1 * LAYER_2_WIDTH + 13]),
    .out15(layer_3_gr_update_patch[1 * LAYER_2_WIDTH + 14]),
    .out16(layer_3_gr_update_patch[1 * LAYER_2_WIDTH + 15])
);

fc_gr_update16x1 #(.WL(WL), .FL(FL), .LEARNING_RATE(LEARNING_RATE)) fc_bp3(
    .CLK(CLK),
    .RESET(RESET),
    .err_factor(output_error[2]),
    .in1(layer_3_gr_in_patch[0]),
    .in2(layer_3_gr_in_patch[1]),
    .in3(layer_3_gr_in_patch[2]),
    .in4(layer_3_gr_in_patch[3]),
    .in5(layer_3_gr_in_patch[4]),
    .in6(layer_3_gr_in_patch[5]),
    .in7(layer_3_gr_in_patch[6]),
    .in8(layer_3_gr_in_patch[7]),
    .in9(layer_3_gr_in_patch[8]),
    .in10(layer_3_gr_in_patch[9]),
    .in11(layer_3_gr_in_patch[10]),
    .in12(layer_3_gr_in_patch[11]),
    .in13(layer_3_gr_in_patch[12]),
    .in14(layer_3_gr_in_patch[13]),
    .in15(layer_3_gr_in_patch[14]),
    .in16(layer_3_gr_in_patch[15]),
    .wght1(layer_3_weight_patch[2 * LAYER_2_WIDTH + 0]),
    .wght2(layer_3_weight_patch[2 * LAYER_2_WIDTH + 1]),
    .wght3(layer_3_weight_patch[2 * LAYER_2_WIDTH + 2]),
    .wght4(layer_3_weight_patch[2 * LAYER_2_WIDTH + 3]),
    .wght5(layer_3_weight_patch[2 * LAYER_2_WIDTH + 4]),
    .wght6(layer_3_weight_patch[2 * LAYER_2_WIDTH + 5]),
    .wght7(layer_3_weight_patch[2 * LAYER_2_WIDTH + 6]),
    .wght8(layer_3_weight_patch[2 * LAYER_2_WIDTH + 7]),
    .wght9(layer_3_weight_patch[2 * LAYER_2_WIDTH + 8]),
    .wght10(layer_3_weight_patch[2 * LAYER_2_WIDTH + 9]),
    .wght11(layer_3_weight_patch[2 * LAYER_2_WIDTH + 10]),
    .wght12(layer_3_weight_patch[2 * LAYER_2_WIDTH + 11]),
    .wght13(layer_3_weight_patch[2 * LAYER_2_WIDTH + 12]),
    .wght14(layer_3_weight_patch[2 * LAYER_2_WIDTH + 13]),
    .wght15(layer_3_weight_patch[2 * LAYER_2_WIDTH + 14]),
    .wght16(layer_3_weight_patch[2 * LAYER_2_WIDTH + 15]),
    .out1(layer_3_gr_update_patch[2 * LAYER_2_WIDTH + 0]),
    .out2(layer_3_gr_update_patch[2 * LAYER_2_WIDTH + 1]),
    .out3(layer_3_gr_update_patch[2 * LAYER_2_WIDTH + 2]),
    .out4(layer_3_gr_update_patch[2 * LAYER_2_WIDTH + 3]),
    .out5(layer_3_gr_update_patch[2 * LAYER_2_WIDTH + 4]),
    .out6(layer_3_gr_update_patch[2 * LAYER_2_WIDTH + 5]),
    .out7(layer_3_gr_update_patch[2 * LAYER_2_WIDTH + 6]),
    .out8(layer_3_gr_update_patch[2 * LAYER_2_WIDTH + 7]),
    .out9(layer_3_gr_update_patch[2 * LAYER_2_WIDTH + 8]),
    .out10(layer_3_gr_update_patch[2 * LAYER_2_WIDTH + 9]),
    .out11(layer_3_gr_update_patch[2 * LAYER_2_WIDTH + 10]),
    .out12(layer_3_gr_update_patch[2 * LAYER_2_WIDTH + 11]),
    .out13(layer_3_gr_update_patch[2 * LAYER_2_WIDTH + 12]),
    .out14(layer_3_gr_update_patch[2 * LAYER_2_WIDTH + 13]),
    .out15(layer_3_gr_update_patch[2 * LAYER_2_WIDTH + 14]),
    .out16(layer_3_gr_update_patch[2 * LAYER_2_WIDTH + 15])
);

fc_gr_update16x1 #(.WL(WL), .FL(FL), .LEARNING_RATE(LEARNING_RATE)) fc_bp4(
    .CLK(CLK),
    .RESET(RESET),
    .err_factor(output_error[3]),
    .in1(layer_3_gr_in_patch[0]),
    .in2(layer_3_gr_in_patch[1]),
    .in3(layer_3_gr_in_patch[2]),
    .in4(layer_3_gr_in_patch[3]),
    .in5(layer_3_gr_in_patch[4]),
    .in6(layer_3_gr_in_patch[5]),
    .in7(layer_3_gr_in_patch[6]),
    .in8(layer_3_gr_in_patch[7]),
    .in9(layer_3_gr_in_patch[8]),
    .in10(layer_3_gr_in_patch[9]),
    .in11(layer_3_gr_in_patch[10]),
    .in12(layer_3_gr_in_patch[11]),
    .in13(layer_3_gr_in_patch[12]),
    .in14(layer_3_gr_in_patch[13]),
    .in15(layer_3_gr_in_patch[14]),
    .in16(layer_3_gr_in_patch[15]),
    .wght1(layer_3_weight_patch[3 * LAYER_2_WIDTH + 0]),
    .wght2(layer_3_weight_patch[3 * LAYER_2_WIDTH + 1]),
    .wght3(layer_3_weight_patch[3 * LAYER_2_WIDTH + 2]),
    .wght4(layer_3_weight_patch[3 * LAYER_2_WIDTH + 3]),
    .wght5(layer_3_weight_patch[3 * LAYER_2_WIDTH + 4]),
    .wght6(layer_3_weight_patch[3 * LAYER_2_WIDTH + 5]),
    .wght7(layer_3_weight_patch[3 * LAYER_2_WIDTH + 6]),
    .wght8(layer_3_weight_patch[3 * LAYER_2_WIDTH + 7]),
    .wght9(layer_3_weight_patch[3 * LAYER_2_WIDTH + 8]),
    .wght10(layer_3_weight_patch[3 * LAYER_2_WIDTH + 9]),
    .wght11(layer_3_weight_patch[3 * LAYER_2_WIDTH + 10]),
    .wght12(layer_3_weight_patch[3 * LAYER_2_WIDTH + 11]),
    .wght13(layer_3_weight_patch[3 * LAYER_2_WIDTH + 12]),
    .wght14(layer_3_weight_patch[3 * LAYER_2_WIDTH + 13]),
    .wght15(layer_3_weight_patch[3 * LAYER_2_WIDTH + 14]),
    .wght16(layer_3_weight_patch[3 * LAYER_2_WIDTH + 15]),
    .out1(layer_3_gr_update_patch[3 * LAYER_2_WIDTH + 0]),
    .out2(layer_3_gr_update_patch[3 * LAYER_2_WIDTH + 1]),
    .out3(layer_3_gr_update_patch[3 * LAYER_2_WIDTH + 2]),
    .out4(layer_3_gr_update_patch[3 * LAYER_2_WIDTH + 3]),
    .out5(layer_3_gr_update_patch[3 * LAYER_2_WIDTH + 4]),
    .out6(layer_3_gr_update_patch[3 * LAYER_2_WIDTH + 5]),
    .out7(layer_3_gr_update_patch[3 * LAYER_2_WIDTH + 6]),
    .out8(layer_3_gr_update_patch[3 * LAYER_2_WIDTH + 7]),
    .out9(layer_3_gr_update_patch[3 * LAYER_2_WIDTH + 8]),
    .out10(layer_3_gr_update_patch[3 * LAYER_2_WIDTH + 9]),
    .out11(layer_3_gr_update_patch[3 * LAYER_2_WIDTH + 10]),
    .out12(layer_3_gr_update_patch[3 * LAYER_2_WIDTH + 11]),
    .out13(layer_3_gr_update_patch[3 * LAYER_2_WIDTH + 12]),
    .out14(layer_3_gr_update_patch[3 * LAYER_2_WIDTH + 13]),
    .out15(layer_3_gr_update_patch[3 * LAYER_2_WIDTH + 14]),
    .out16(layer_3_gr_update_patch[3 * LAYER_2_WIDTH + 15])
);

fc_gr_update16x1 #(.WL(WL), .FL(FL), .LEARNING_RATE(LEARNING_RATE)) fc_bp5(
    .CLK(CLK),
    .RESET(RESET),
    .err_factor(output_error[4]),
    .in1(layer_3_gr_in_patch[0]),
    .in2(layer_3_gr_in_patch[1]),
    .in3(layer_3_gr_in_patch[2]),
    .in4(layer_3_gr_in_patch[3]),
    .in5(layer_3_gr_in_patch[4]),
    .in6(layer_3_gr_in_patch[5]),
    .in7(layer_3_gr_in_patch[6]),
    .in8(layer_3_gr_in_patch[7]),
    .in9(layer_3_gr_in_patch[8]),
    .in10(layer_3_gr_in_patch[9]),
    .in11(layer_3_gr_in_patch[10]),
    .in12(layer_3_gr_in_patch[11]),
    .in13(layer_3_gr_in_patch[12]),
    .in14(layer_3_gr_in_patch[13]),
    .in15(layer_3_gr_in_patch[14]),
    .in16(layer_3_gr_in_patch[15]),
    .wght1(layer_3_weight_patch[4 * LAYER_2_WIDTH + 0]),
    .wght2(layer_3_weight_patch[4 * LAYER_2_WIDTH + 1]),
    .wght3(layer_3_weight_patch[4 * LAYER_2_WIDTH + 2]),
    .wght4(layer_3_weight_patch[4 * LAYER_2_WIDTH + 3]),
    .wght5(layer_3_weight_patch[4 * LAYER_2_WIDTH + 4]),
    .wght6(layer_3_weight_patch[4 * LAYER_2_WIDTH + 5]),
    .wght7(layer_3_weight_patch[4 * LAYER_2_WIDTH + 6]),
    .wght8(layer_3_weight_patch[4 * LAYER_2_WIDTH + 7]),
    .wght9(layer_3_weight_patch[4 * LAYER_2_WIDTH + 8]),
    .wght10(layer_3_weight_patch[4 * LAYER_2_WIDTH + 9]),
    .wght11(layer_3_weight_patch[4 * LAYER_2_WIDTH + 10]),
    .wght12(layer_3_weight_patch[4 * LAYER_2_WIDTH + 11]),
    .wght13(layer_3_weight_patch[4 * LAYER_2_WIDTH + 12]),
    .wght14(layer_3_weight_patch[4 * LAYER_2_WIDTH + 13]),
    .wght15(layer_3_weight_patch[4 * LAYER_2_WIDTH + 14]),
    .wght16(layer_3_weight_patch[4 * LAYER_2_WIDTH + 15]),
    .out1(layer_3_gr_update_patch[4 * LAYER_2_WIDTH + 0]),
    .out2(layer_3_gr_update_patch[4 * LAYER_2_WIDTH + 1]),
    .out3(layer_3_gr_update_patch[4 * LAYER_2_WIDTH + 2]),
    .out4(layer_3_gr_update_patch[4 * LAYER_2_WIDTH + 3]),
    .out5(layer_3_gr_update_patch[4 * LAYER_2_WIDTH + 4]),
    .out6(layer_3_gr_update_patch[4 * LAYER_2_WIDTH + 5]),
    .out7(layer_3_gr_update_patch[4 * LAYER_2_WIDTH + 6]),
    .out8(layer_3_gr_update_patch[4 * LAYER_2_WIDTH + 7]),
    .out9(layer_3_gr_update_patch[4 * LAYER_2_WIDTH + 8]),
    .out10(layer_3_gr_update_patch[4 * LAYER_2_WIDTH + 9]),
    .out11(layer_3_gr_update_patch[4 * LAYER_2_WIDTH + 10]),
    .out12(layer_3_gr_update_patch[4 * LAYER_2_WIDTH + 11]),
    .out13(layer_3_gr_update_patch[4 * LAYER_2_WIDTH + 12]),
    .out14(layer_3_gr_update_patch[4 * LAYER_2_WIDTH + 13]),
    .out15(layer_3_gr_update_patch[4 * LAYER_2_WIDTH + 14]),
    .out16(layer_3_gr_update_patch[4 * LAYER_2_WIDTH + 15])
);

fc_gr_update16x1 #(.WL(WL), .FL(FL), .LEARNING_RATE(LEARNING_RATE)) fc_bp6(
    .CLK(CLK),
    .RESET(RESET),
    .err_factor(output_error[5]),
    .in1(layer_3_gr_in_patch[0]),
    .in2(layer_3_gr_in_patch[1]),
    .in3(layer_3_gr_in_patch[2]),
    .in4(layer_3_gr_in_patch[3]),
    .in5(layer_3_gr_in_patch[4]),
    .in6(layer_3_gr_in_patch[5]),
    .in7(layer_3_gr_in_patch[6]),
    .in8(layer_3_gr_in_patch[7]),
    .in9(layer_3_gr_in_patch[8]),
    .in10(layer_3_gr_in_patch[9]),
    .in11(layer_3_gr_in_patch[10]),
    .in12(layer_3_gr_in_patch[11]),
    .in13(layer_3_gr_in_patch[12]),
    .in14(layer_3_gr_in_patch[13]),
    .in15(layer_3_gr_in_patch[14]),
    .in16(layer_3_gr_in_patch[15]),
    .wght1(layer_3_weight_patch[5 * LAYER_2_WIDTH + 0]),
    .wght2(layer_3_weight_patch[5 * LAYER_2_WIDTH + 1]),
    .wght3(layer_3_weight_patch[5 * LAYER_2_WIDTH + 2]),
    .wght4(layer_3_weight_patch[5 * LAYER_2_WIDTH + 3]),
    .wght5(layer_3_weight_patch[5 * LAYER_2_WIDTH + 4]),
    .wght6(layer_3_weight_patch[5 * LAYER_2_WIDTH + 5]),
    .wght7(layer_3_weight_patch[5 * LAYER_2_WIDTH + 6]),
    .wght8(layer_3_weight_patch[5 * LAYER_2_WIDTH + 7]),
    .wght9(layer_3_weight_patch[5 * LAYER_2_WIDTH + 8]),
    .wght10(layer_3_weight_patch[5 * LAYER_2_WIDTH + 9]),
    .wght11(layer_3_weight_patch[5 * LAYER_2_WIDTH + 10]),
    .wght12(layer_3_weight_patch[5 * LAYER_2_WIDTH + 11]),
    .wght13(layer_3_weight_patch[5 * LAYER_2_WIDTH + 12]),
    .wght14(layer_3_weight_patch[5 * LAYER_2_WIDTH + 13]),
    .wght15(layer_3_weight_patch[5 * LAYER_2_WIDTH + 14]),
    .wght16(layer_3_weight_patch[5 * LAYER_2_WIDTH + 15]),
    .out1(layer_3_gr_update_patch[5 * LAYER_2_WIDTH + 0]),
    .out2(layer_3_gr_update_patch[5 * LAYER_2_WIDTH + 1]),
    .out3(layer_3_gr_update_patch[5 * LAYER_2_WIDTH + 2]),
    .out4(layer_3_gr_update_patch[5 * LAYER_2_WIDTH + 3]),
    .out5(layer_3_gr_update_patch[5 * LAYER_2_WIDTH + 4]),
    .out6(layer_3_gr_update_patch[5 * LAYER_2_WIDTH + 5]),
    .out7(layer_3_gr_update_patch[5 * LAYER_2_WIDTH + 6]),
    .out8(layer_3_gr_update_patch[5 * LAYER_2_WIDTH + 7]),
    .out9(layer_3_gr_update_patch[5 * LAYER_2_WIDTH + 8]),
    .out10(layer_3_gr_update_patch[5 * LAYER_2_WIDTH + 9]),
    .out11(layer_3_gr_update_patch[5 * LAYER_2_WIDTH + 10]),
    .out12(layer_3_gr_update_patch[5 * LAYER_2_WIDTH + 11]),
    .out13(layer_3_gr_update_patch[5 * LAYER_2_WIDTH + 12]),
    .out14(layer_3_gr_update_patch[5 * LAYER_2_WIDTH + 13]),
    .out15(layer_3_gr_update_patch[5 * LAYER_2_WIDTH + 14]),
    .out16(layer_3_gr_update_patch[5 * LAYER_2_WIDTH + 15])
);

fc_gr_update16x1 #(.WL(WL), .FL(FL), .LEARNING_RATE(LEARNING_RATE)) fc_bp7(
    .CLK(CLK),
    .RESET(RESET),
    .err_factor(output_error[6]),
    .in1(layer_3_gr_in_patch[0]),
    .in2(layer_3_gr_in_patch[1]),
    .in3(layer_3_gr_in_patch[2]),
    .in4(layer_3_gr_in_patch[3]),
    .in5(layer_3_gr_in_patch[4]),
    .in6(layer_3_gr_in_patch[5]),
    .in7(layer_3_gr_in_patch[6]),
    .in8(layer_3_gr_in_patch[7]),
    .in9(layer_3_gr_in_patch[8]),
    .in10(layer_3_gr_in_patch[9]),
    .in11(layer_3_gr_in_patch[10]),
    .in12(layer_3_gr_in_patch[11]),
    .in13(layer_3_gr_in_patch[12]),
    .in14(layer_3_gr_in_patch[13]),
    .in15(layer_3_gr_in_patch[14]),
    .in16(layer_3_gr_in_patch[15]),
    .wght1(layer_3_weight_patch[6 * LAYER_2_WIDTH + 0]),
    .wght2(layer_3_weight_patch[6 * LAYER_2_WIDTH + 1]),
    .wght3(layer_3_weight_patch[6 * LAYER_2_WIDTH + 2]),
    .wght4(layer_3_weight_patch[6 * LAYER_2_WIDTH + 3]),
    .wght5(layer_3_weight_patch[6 * LAYER_2_WIDTH + 4]),
    .wght6(layer_3_weight_patch[6 * LAYER_2_WIDTH + 5]),
    .wght7(layer_3_weight_patch[6 * LAYER_2_WIDTH + 6]),
    .wght8(layer_3_weight_patch[6 * LAYER_2_WIDTH + 7]),
    .wght9(layer_3_weight_patch[6 * LAYER_2_WIDTH + 8]),
    .wght10(layer_3_weight_patch[6 * LAYER_2_WIDTH + 9]),
    .wght11(layer_3_weight_patch[6 * LAYER_2_WIDTH + 10]),
    .wght12(layer_3_weight_patch[6 * LAYER_2_WIDTH + 11]),
    .wght13(layer_3_weight_patch[6 * LAYER_2_WIDTH + 12]),
    .wght14(layer_3_weight_patch[6 * LAYER_2_WIDTH + 13]),
    .wght15(layer_3_weight_patch[6 * LAYER_2_WIDTH + 14]),
    .wght16(layer_3_weight_patch[6 * LAYER_2_WIDTH + 15]),
    .out1(layer_3_gr_update_patch[6 * LAYER_2_WIDTH + 0]),
    .out2(layer_3_gr_update_patch[6 * LAYER_2_WIDTH + 1]),
    .out3(layer_3_gr_update_patch[6 * LAYER_2_WIDTH + 2]),
    .out4(layer_3_gr_update_patch[6 * LAYER_2_WIDTH + 3]),
    .out5(layer_3_gr_update_patch[6 * LAYER_2_WIDTH + 4]),
    .out6(layer_3_gr_update_patch[6 * LAYER_2_WIDTH + 5]),
    .out7(layer_3_gr_update_patch[6 * LAYER_2_WIDTH + 6]),
    .out8(layer_3_gr_update_patch[6 * LAYER_2_WIDTH + 7]),
    .out9(layer_3_gr_update_patch[6 * LAYER_2_WIDTH + 8]),
    .out10(layer_3_gr_update_patch[6 * LAYER_2_WIDTH + 9]),
    .out11(layer_3_gr_update_patch[6 * LAYER_2_WIDTH + 10]),
    .out12(layer_3_gr_update_patch[6 * LAYER_2_WIDTH + 11]),
    .out13(layer_3_gr_update_patch[6 * LAYER_2_WIDTH + 12]),
    .out14(layer_3_gr_update_patch[6 * LAYER_2_WIDTH + 13]),
    .out15(layer_3_gr_update_patch[6 * LAYER_2_WIDTH + 14]),
    .out16(layer_3_gr_update_patch[6 * LAYER_2_WIDTH + 15])
);

fc_gr_update16x1 #(.WL(WL), .FL(FL), .LEARNING_RATE(LEARNING_RATE)) fc_bp8(
    .CLK(CLK),
    .RESET(RESET),
    .err_factor(output_error[7]),
    .in1(layer_3_gr_in_patch[0]),
    .in2(layer_3_gr_in_patch[1]),
    .in3(layer_3_gr_in_patch[2]),
    .in4(layer_3_gr_in_patch[3]),
    .in5(layer_3_gr_in_patch[4]),
    .in6(layer_3_gr_in_patch[5]),
    .in7(layer_3_gr_in_patch[6]),
    .in8(layer_3_gr_in_patch[7]),
    .in9(layer_3_gr_in_patch[8]),
    .in10(layer_3_gr_in_patch[9]),
    .in11(layer_3_gr_in_patch[10]),
    .in12(layer_3_gr_in_patch[11]),
    .in13(layer_3_gr_in_patch[12]),
    .in14(layer_3_gr_in_patch[13]),
    .in15(layer_3_gr_in_patch[14]),
    .in16(layer_3_gr_in_patch[15]),
    .wght1(layer_3_weight_patch[7 * LAYER_2_WIDTH + 0]),
    .wght2(layer_3_weight_patch[7 * LAYER_2_WIDTH + 1]),
    .wght3(layer_3_weight_patch[7 * LAYER_2_WIDTH + 2]),
    .wght4(layer_3_weight_patch[7 * LAYER_2_WIDTH + 3]),
    .wght5(layer_3_weight_patch[7 * LAYER_2_WIDTH + 4]),
    .wght6(layer_3_weight_patch[7 * LAYER_2_WIDTH + 5]),
    .wght7(layer_3_weight_patch[7 * LAYER_2_WIDTH + 6]),
    .wght8(layer_3_weight_patch[7 * LAYER_2_WIDTH + 7]),
    .wght9(layer_3_weight_patch[7 * LAYER_2_WIDTH + 8]),
    .wght10(layer_3_weight_patch[7 * LAYER_2_WIDTH + 9]),
    .wght11(layer_3_weight_patch[7 * LAYER_2_WIDTH + 10]),
    .wght12(layer_3_weight_patch[7 * LAYER_2_WIDTH + 11]),
    .wght13(layer_3_weight_patch[7 * LAYER_2_WIDTH + 12]),
    .wght14(layer_3_weight_patch[7 * LAYER_2_WIDTH + 13]),
    .wght15(layer_3_weight_patch[7 * LAYER_2_WIDTH + 14]),
    .wght16(layer_3_weight_patch[7 * LAYER_2_WIDTH + 15]),
    .out1(layer_3_gr_update_patch[7 * LAYER_2_WIDTH + 0]),
    .out2(layer_3_gr_update_patch[7 * LAYER_2_WIDTH + 1]),
    .out3(layer_3_gr_update_patch[7 * LAYER_2_WIDTH + 2]),
    .out4(layer_3_gr_update_patch[7 * LAYER_2_WIDTH + 3]),
    .out5(layer_3_gr_update_patch[7 * LAYER_2_WIDTH + 4]),
    .out6(layer_3_gr_update_patch[7 * LAYER_2_WIDTH + 5]),
    .out7(layer_3_gr_update_patch[7 * LAYER_2_WIDTH + 6]),
    .out8(layer_3_gr_update_patch[7 * LAYER_2_WIDTH + 7]),
    .out9(layer_3_gr_update_patch[7 * LAYER_2_WIDTH + 8]),
    .out10(layer_3_gr_update_patch[7 * LAYER_2_WIDTH + 9]),
    .out11(layer_3_gr_update_patch[7 * LAYER_2_WIDTH + 10]),
    .out12(layer_3_gr_update_patch[7 * LAYER_2_WIDTH + 11]),
    .out13(layer_3_gr_update_patch[7 * LAYER_2_WIDTH + 12]),
    .out14(layer_3_gr_update_patch[7 * LAYER_2_WIDTH + 13]),
    .out15(layer_3_gr_update_patch[7 * LAYER_2_WIDTH + 14]),
    .out16(layer_3_gr_update_patch[7 * LAYER_2_WIDTH + 15])
);

fc_gr_update16x1 #(.WL(WL), .FL(FL), .LEARNING_RATE(LEARNING_RATE)) fc_bp9(
    .CLK(CLK),
    .RESET(RESET),
    .err_factor(output_error[8]),
    .in1(layer_3_gr_in_patch[0]),
    .in2(layer_3_gr_in_patch[1]),
    .in3(layer_3_gr_in_patch[2]),
    .in4(layer_3_gr_in_patch[3]),
    .in5(layer_3_gr_in_patch[4]),
    .in6(layer_3_gr_in_patch[5]),
    .in7(layer_3_gr_in_patch[6]),
    .in8(layer_3_gr_in_patch[7]),
    .in9(layer_3_gr_in_patch[8]),
    .in10(layer_3_gr_in_patch[9]),
    .in11(layer_3_gr_in_patch[10]),
    .in12(layer_3_gr_in_patch[11]),
    .in13(layer_3_gr_in_patch[12]),
    .in14(layer_3_gr_in_patch[13]),
    .in15(layer_3_gr_in_patch[14]),
    .in16(layer_3_gr_in_patch[15]),
    .wght1(layer_3_weight_patch[8 * LAYER_2_WIDTH + 0]),
    .wght2(layer_3_weight_patch[8 * LAYER_2_WIDTH + 1]),
    .wght3(layer_3_weight_patch[8 * LAYER_2_WIDTH + 2]),
    .wght4(layer_3_weight_patch[8 * LAYER_2_WIDTH + 3]),
    .wght5(layer_3_weight_patch[8 * LAYER_2_WIDTH + 4]),
    .wght6(layer_3_weight_patch[8 * LAYER_2_WIDTH + 5]),
    .wght7(layer_3_weight_patch[8 * LAYER_2_WIDTH + 6]),
    .wght8(layer_3_weight_patch[8 * LAYER_2_WIDTH + 7]),
    .wght9(layer_3_weight_patch[8 * LAYER_2_WIDTH + 8]),
    .wght10(layer_3_weight_patch[8 * LAYER_2_WIDTH + 9]),
    .wght11(layer_3_weight_patch[8 * LAYER_2_WIDTH + 10]),
    .wght12(layer_3_weight_patch[8 * LAYER_2_WIDTH + 11]),
    .wght13(layer_3_weight_patch[8 * LAYER_2_WIDTH + 12]),
    .wght14(layer_3_weight_patch[8 * LAYER_2_WIDTH + 13]),
    .wght15(layer_3_weight_patch[8 * LAYER_2_WIDTH + 14]),
    .wght16(layer_3_weight_patch[8 * LAYER_2_WIDTH + 15]),
    .out1(layer_3_gr_update_patch[8 * LAYER_2_WIDTH + 0]),
    .out2(layer_3_gr_update_patch[8 * LAYER_2_WIDTH + 1]),
    .out3(layer_3_gr_update_patch[8 * LAYER_2_WIDTH + 2]),
    .out4(layer_3_gr_update_patch[8 * LAYER_2_WIDTH + 3]),
    .out5(layer_3_gr_update_patch[8 * LAYER_2_WIDTH + 4]),
    .out6(layer_3_gr_update_patch[8 * LAYER_2_WIDTH + 5]),
    .out7(layer_3_gr_update_patch[8 * LAYER_2_WIDTH + 6]),
    .out8(layer_3_gr_update_patch[8 * LAYER_2_WIDTH + 7]),
    .out9(layer_3_gr_update_patch[8 * LAYER_2_WIDTH + 8]),
    .out10(layer_3_gr_update_patch[8 * LAYER_2_WIDTH + 9]),
    .out11(layer_3_gr_update_patch[8 * LAYER_2_WIDTH + 10]),
    .out12(layer_3_gr_update_patch[8 * LAYER_2_WIDTH + 11]),
    .out13(layer_3_gr_update_patch[8 * LAYER_2_WIDTH + 12]),
    .out14(layer_3_gr_update_patch[8 * LAYER_2_WIDTH + 13]),
    .out15(layer_3_gr_update_patch[8 * LAYER_2_WIDTH + 14]),
    .out16(layer_3_gr_update_patch[8 * LAYER_2_WIDTH + 15])
);

fc_gr_update16x1 #(.WL(WL), .FL(FL), .LEARNING_RATE(LEARNING_RATE)) fc_bp10(
    .CLK(CLK),
    .RESET(RESET),
    .err_factor(output_error[9]),
    .in1(layer_3_gr_in_patch[0]),
    .in2(layer_3_gr_in_patch[1]),
    .in3(layer_3_gr_in_patch[2]),
    .in4(layer_3_gr_in_patch[3]),
    .in5(layer_3_gr_in_patch[4]),
    .in6(layer_3_gr_in_patch[5]),
    .in7(layer_3_gr_in_patch[6]),
    .in8(layer_3_gr_in_patch[7]),
    .in9(layer_3_gr_in_patch[8]),
    .in10(layer_3_gr_in_patch[9]),
    .in11(layer_3_gr_in_patch[10]),
    .in12(layer_3_gr_in_patch[11]),
    .in13(layer_3_gr_in_patch[12]),
    .in14(layer_3_gr_in_patch[13]),
    .in15(layer_3_gr_in_patch[14]),
    .in16(layer_3_gr_in_patch[15]),
    .wght1(layer_3_weight_patch[9 * LAYER_2_WIDTH + 0]),
    .wght2(layer_3_weight_patch[9 * LAYER_2_WIDTH + 1]),
    .wght3(layer_3_weight_patch[9 * LAYER_2_WIDTH + 2]),
    .wght4(layer_3_weight_patch[9 * LAYER_2_WIDTH + 3]),
    .wght5(layer_3_weight_patch[9 * LAYER_2_WIDTH + 4]),
    .wght6(layer_3_weight_patch[9 * LAYER_2_WIDTH + 5]),
    .wght7(layer_3_weight_patch[9 * LAYER_2_WIDTH + 6]),
    .wght8(layer_3_weight_patch[9 * LAYER_2_WIDTH + 7]),
    .wght9(layer_3_weight_patch[9 * LAYER_2_WIDTH + 8]),
    .wght10(layer_3_weight_patch[9 * LAYER_2_WIDTH + 9]),
    .wght11(layer_3_weight_patch[9 * LAYER_2_WIDTH + 10]),
    .wght12(layer_3_weight_patch[9 * LAYER_2_WIDTH + 11]),
    .wght13(layer_3_weight_patch[9 * LAYER_2_WIDTH + 12]),
    .wght14(layer_3_weight_patch[9 * LAYER_2_WIDTH + 13]),
    .wght15(layer_3_weight_patch[9 * LAYER_2_WIDTH + 14]),
    .wght16(layer_3_weight_patch[9 * LAYER_2_WIDTH + 15]),
    .out1(layer_3_gr_update_patch[9 * LAYER_2_WIDTH + 0]),
    .out2(layer_3_gr_update_patch[9 * LAYER_2_WIDTH + 1]),
    .out3(layer_3_gr_update_patch[9 * LAYER_2_WIDTH + 2]),
    .out4(layer_3_gr_update_patch[9 * LAYER_2_WIDTH + 3]),
    .out5(layer_3_gr_update_patch[9 * LAYER_2_WIDTH + 4]),
    .out6(layer_3_gr_update_patch[9 * LAYER_2_WIDTH + 5]),
    .out7(layer_3_gr_update_patch[9 * LAYER_2_WIDTH + 6]),
    .out8(layer_3_gr_update_patch[9 * LAYER_2_WIDTH + 7]),
    .out9(layer_3_gr_update_patch[9 * LAYER_2_WIDTH + 8]),
    .out10(layer_3_gr_update_patch[9 * LAYER_2_WIDTH + 9]),
    .out11(layer_3_gr_update_patch[9 * LAYER_2_WIDTH + 10]),
    .out12(layer_3_gr_update_patch[9 * LAYER_2_WIDTH + 11]),
    .out13(layer_3_gr_update_patch[9 * LAYER_2_WIDTH + 12]),
    .out14(layer_3_gr_update_patch[9 * LAYER_2_WIDTH + 13]),
    .out15(layer_3_gr_update_patch[9 * LAYER_2_WIDTH + 14]),
    .out16(layer_3_gr_update_patch[9 * LAYER_2_WIDTH + 15])
);

fc_bp16x1 #(.WL(WL), .FL(FL)) fc_error_prop(
    .CLK(CLK),
    .RESET(RESET),
    .err_factor(output_error[0]),
    .der1(layer_3_der_patch[0]),
    .der2(layer_3_der_patch[1]),
    .der3(layer_3_der_patch[2]),
    .der4(layer_3_der_patch[3]),
    .der5(layer_3_der_patch[4]),
    .der6(layer_3_der_patch[5]),
    .der7(layer_3_der_patch[6]),
    .der8(layer_3_der_patch[7]),
    .der9(layer_3_der_patch[8]),
    .der10(layer_3_der_patch[9]),
    .der11(layer_3_der_patch[10]),
    .der12(layer_3_der_patch[11]),
    .der13(layer_3_der_patch[12]),
    .der14(layer_3_der_patch[13]),
    .der15(layer_3_der_patch[14]),
    .der16(layer_3_der_patch[15]),
    .wght1(layer_3_err_weight_patch[0]),
    .wght2(layer_3_err_weight_patch[1]),
    .wght3(layer_3_err_weight_patch[2]),
    .wght4(layer_3_err_weight_patch[3]),
    .wght5(layer_3_err_weight_patch[4]),
    .wght6(layer_3_err_weight_patch[5]),
    .wght7(layer_3_err_weight_patch[6]),
    .wght8(layer_3_err_weight_patch[7]),
    .wght9(layer_3_err_weight_patch[8]),
    .wght10(layer_3_err_weight_patch[9]),
    .wght11(layer_3_err_weight_patch[10]),
    .wght12(layer_3_err_weight_patch[11]),
    .wght13(layer_3_err_weight_patch[12]),
    .wght14(layer_3_err_weight_patch[13]),
    .wght15(layer_3_err_weight_patch[14]),
    .wght16(layer_3_err_weight_patch[15]),
    .out1(layer_3_error_patch[0]),
    .out2(layer_3_error_patch[1]),
    .out3(layer_3_error_patch[2]),
    .out4(layer_3_error_patch[3]),
    .out5(layer_3_error_patch[4]),
    .out6(layer_3_error_patch[5]),
    .out7(layer_3_error_patch[6]),
    .out8(layer_3_error_patch[7]),
    .out9(layer_3_error_patch[8]),
    .out10(layer_3_error_patch[9]),
    .out11(layer_3_error_patch[10]),
    .out12(layer_3_error_patch[11]),
    .out13(layer_3_error_patch[12]),
    .out14(layer_3_error_patch[13]),
    .out15(layer_3_error_patch[14]),
    .out16(layer_3_error_patch[15])
);

add_with_saturate #(.WL(WL), .FL(FL)) layer_3_err_acc1(
    .a(layer_3_error_patch[0]),
    .b(layer_3_acc[0]),
    .out(layer_3_new_acc[0])
);

add_with_saturate #(.WL(WL), .FL(FL)) layer_3_err_acc2(
    .a(layer_3_error_patch[1]),
    .b(layer_3_acc[1]),
    .out(layer_3_new_acc[1])
);

add_with_saturate #(.WL(WL), .FL(FL)) layer_3_err_acc3(
    .a(layer_3_error_patch[2]),
    .b(layer_3_acc[2]),
    .out(layer_3_new_acc[2])
);

add_with_saturate #(.WL(WL), .FL(FL)) layer_3_err_acc4(
    .a(layer_3_error_patch[3]),
    .b(layer_3_acc[3]),
    .out(layer_3_new_acc[3])
);

add_with_saturate #(.WL(WL), .FL(FL)) layer_3_err_acc5(
    .a(layer_3_error_patch[4]),
    .b(layer_3_acc[4]),
    .out(layer_3_new_acc[4])
);

add_with_saturate #(.WL(WL), .FL(FL)) layer_3_err_acc6(
    .a(layer_3_error_patch[5]),
    .b(layer_3_acc[5]),
    .out(layer_3_new_acc[5])
);

add_with_saturate #(.WL(WL), .FL(FL)) layer_3_err_acc7(
    .a(layer_3_error_patch[6]),
    .b(layer_3_acc[6]),
    .out(layer_3_new_acc[6])
);

add_with_saturate #(.WL(WL), .FL(FL)) layer_3_err_acc8(
    .a(layer_3_error_patch[7]),
    .b(layer_3_acc[7]),
    .out(layer_3_new_acc[7])
);

add_with_saturate #(.WL(WL), .FL(FL)) layer_3_err_acc9(
    .a(layer_3_error_patch[8]),
    .b(layer_3_acc[8]),
    .out(layer_3_new_acc[8])
);

add_with_saturate #(.WL(WL), .FL(FL)) layer_3_err_acc10(
    .a(layer_3_error_patch[9]),
    .b(layer_3_acc[9]),
    .out(layer_3_new_acc[9])
);

add_with_saturate #(.WL(WL), .FL(FL)) layer_3_err_acc11(
    .a(layer_3_error_patch[10]),
    .b(layer_3_acc[10]),
    .out(layer_3_new_acc[10])
);

add_with_saturate #(.WL(WL), .FL(FL)) layer_3_err_acc12(
    .a(layer_3_error_patch[11]),
    .b(layer_3_acc[11]),
    .out(layer_3_new_acc[11])
);

add_with_saturate #(.WL(WL), .FL(FL)) layer_3_err_acc13(
    .a(layer_3_error_patch[12]),
    .b(layer_3_acc[12]),
    .out(layer_3_new_acc[12])
);

add_with_saturate #(.WL(WL), .FL(FL)) layer_3_err_acc14(
    .a(layer_3_error_patch[13]),
    .b(layer_3_acc[13]),
    .out(layer_3_new_acc[13])
);

add_with_saturate #(.WL(WL), .FL(FL)) layer_3_err_acc15(
    .a(layer_3_error_patch[14]),
    .b(layer_3_acc[14]),
    .out(layer_3_new_acc[14])
);

add_with_saturate #(.WL(WL), .FL(FL)) layer_3_err_acc16(
    .a(layer_3_error_patch[15]),
    .b(layer_3_acc[15]),
    .out(layer_3_new_acc[15])
);

endmodule
