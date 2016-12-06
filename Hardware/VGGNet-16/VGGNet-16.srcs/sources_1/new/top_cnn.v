`timescale 1ns / 1ps
module top_cnn(CLK, RESET, Start, Done, 
    in1, in2, in3, in4, in5, in6, in7, in8, in9, 
    out1, out2, out3, out4, out5, out6, out7, out8, out9, out10);

integer i;
integer j;

parameter WL = 16;
parameter FL = 14;
parameter KERNEL_SIZE = 3;
parameter PATCH_LENGTH = KERNEL_SIZE * KERNEL_SIZE;
parameter POOL_LENGTH = 4;
parameter LEARNING_RATE = 16'd15552; // 0.95 using WL = 16 and FL = 14
parameter MAP_BITS = 3;
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
parameter STATE_FINAL_OUTPUT = 11;
 
input CLK, RESET, Start;
input [(WL - 1):0] in1, in2, in3, in4, in5, in6, in7, in8, in9;
output [(WL - 1):0] out1, out2, out3, out4, out5, out6, out7, out8, out9, out10;
output reg Done;

reg [3:0] CurrentState, NextState;

reg [(WL - 1):0] in1_layer_1 [0:(PATCH_LENGTH * LAYER_1_DEPTH - 1)];
reg [(WL - 1):0] in2_layer_1 [0:(PATCH_LENGTH * LAYER_1_DEPTH - 1)];
reg [(WL - 1):0] act_der_layer_1 [0:(LAYER_1_DEPTH - 1)];
wire [(WL - 1):0] out1_layer_1 [0:(LAYER_1_DEPTH - 1)];
wire [(WL - 1):0] out2_layer_1 [0:(LAYER_1_DEPTH - 1)];
reg [(WL - 1):0] out_layer_1 [0:(LAYER_1_VOLUME - 1)];
reg [(WL - 1):0] der_layer_1 [0:(LAYER_1_VOLUME - 1)];
reg [(MAP_BITS - 1):0] layer_1_depth_counter;
reg [(MAP_BITS - 1):0] layer_1_map;
reg [(MAP_BITS - 1):0] layer_1_next_map;
reg [(DIM_BITS - 1):0] layer_1_index;
reg [(DIM_BITS - 1):0] layer_1_next_index;
reg [1:0] layer_1_mode;
reg layer_1_acc;

reg [(WL - 1):0] in_layer_2 [0:(POOL_LENGTH * LAYER_2_DEPTH - 1)];
wire [(WL - 1):0] out1_layer_2 [0:(LAYER_2_DEPTH - 1)];
reg [(WL - 1):0] out_layer_2 [0:(LAYER_2_VOLUME - 1)];
reg [(DIM_BITS - 1):0] layer_2_row, layer_2_col;
reg [(DIM_BITS - 1):0] layer_2_next_row, layer_2_next_col;
reg [(DIM_BITS - 1):0] layer_1_corner;

reg [(WL - 1):0] in1_layer_3 [0:(LAYER_2_WIDTH * LAYER_3_DEPTH - 1)];
reg [(WL - 1):0] in2_layer_3 [0:(LAYER_2_WIDTH * LAYER_3_DEPTH - 1)];
wire [(WL - 1):0] out1_layer_3 [0:(LAYER_3_DEPTH - 1)];
reg [(WL - 1):0] acc_in1_layer_3 [0:(LAYER_3_DEPTH - 1)];
reg [(WL - 1):0] acc_in2_layer_3 [0:(LAYER_3_DEPTH - 1)];
wire [(WL - 1):0] acc_out_layer_3 [0:(LAYER_3_DEPTH - 1)];
reg [(WL - 1):0] out_layer_3 [0:(LAYER_3_VOLUME - 1)];
reg [(WL - 1):0] layer_3_weights [0:(LAYER_2_VOLUME * LAYER_3_DEPTH - 1)];
reg [(DIM_BITS - 1):0] layer_3_index;
reg [(DIM_BITS - 1):0] layer_3_next_index;

assign out1 = out_layer_3[0];
assign out2 = out_layer_3[1];
assign out3 = out_layer_3[2];
assign out4 = out_layer_3[3];
assign out5 = out_layer_3[4];
assign out6 = out_layer_3[5];
assign out7 = out_layer_3[6];
assign out8 = out_layer_3[7];
assign out9 = out_layer_3[8];
assign out10 = out_layer_3[9];

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

endmodule
