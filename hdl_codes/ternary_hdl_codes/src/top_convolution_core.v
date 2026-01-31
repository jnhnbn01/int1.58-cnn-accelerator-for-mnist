`timescale 1ns / 1ps

module top_convolution_core#(
    parameter DATA_WIDTH = 2
)(
    input wire [4:0] input_depth,
    input wire clk,
    input wire resetn,
    input wire start,
    input wire [DATA_WIDTH - 1:0] d_in,
    input wire [DATA_WIDTH - 1:0] w_in, // weight
    input wire w_req, // weight requirement
    input wire [4:0] x1,
    input wire [4:0] y1,
    output wire signed [5:0] conv_out,
    output wire s1_en,
    output wire s2_en,
    output wire done
);
    localparam IDLE = 3'b000, LOAD = 3'b001, CONV1 = 3'b010, CONV2 = 3'b011, DONE = 3'b100;
    reg [2:0] state, next_state;

    assign s1_en = (state == LOAD || state == CONV1) ? 1:0;
    assign s2_en = (state == CONV1 || state == CONV2) ? 1:0;

    reg [4:0] wrptr; reg [1:0] buffer_wrmux;
    reg [4:0] rdptr; reg [2:0] buffer_rdmux;

    reg [DATA_WIDTH-1:0] fmap_buffer0 [27:0];
    reg [DATA_WIDTH-1:0] fmap_buffer1 [27:0];
    reg [DATA_WIDTH-1:0] fmap_buffer2 [27:0];
    reg [3*DATA_WIDTH-1:0] buffer_out;

    reg [4:0] line_cnt;
    reg [2:0] target_counter; // counter for loading 3 columns
    reg [3*DATA_WIDTH-1:0] target_buffer [2:0];
    wire [9*DATA_WIDTH-1:0] pixel_target;
    integer i;

    assign done = (state == DONE) ? 1 : 0;
    assign done_led = done;
    assign pixel_target = (line_cnt == 1 && target_counter == 3 || (line_cnt >= 2 && (target_counter == 0 || target_counter == 3))) ? {target_buffer[2], target_buffer[1], target_buffer[0]} : 72'b0;
    convolution conv_core (.clk(clk), .resetn(resetn), .i_data(pixel_target), .w_data(w_in), .w_req(w_req), .o_data(conv_out));

    // FSM
    always @(posedge clk) begin
        if(!resetn) state <= IDLE;
        else state <= next_state;
    end

    always@(*) begin
        case(state)
            IDLE : next_state = (start) ? LOAD : IDLE;
            LOAD : next_state = (y1 == 2 && x1 == input_depth - 1) ? CONV1 : LOAD;
            CONV1 : next_state = (y1 >= input_depth - 1 && x1 == input_depth - 1) ? CONV2 : CONV1;
            CONV2: next_state = (line_cnt >= input_depth - 1 && x1 >= 9) ? DONE : CONV2;
            DONE : next_state = DONE;
            default : next_state = IDLE;
        endcase
    end

    // Write
    always @(posedge clk) begin
        if(!resetn) begin
            for (i = 0; i < 28; i = i + 1) fmap_buffer0[i] <= 2'd0;
            for (i = 0; i < 28; i = i + 1) fmap_buffer1[i] <= 2'd0;
            for (i = 0; i < 28; i = i + 1) fmap_buffer2[i] <= 2'd0;
        end else begin
            if(state == LOAD) begin
                if(buffer_wrmux == 0) fmap_buffer0[wrptr] <= d_in;
                else if(buffer_wrmux == 1) fmap_buffer1[wrptr] <= d_in;
                else if(buffer_wrmux == 2) fmap_buffer2[wrptr] <= d_in;
            end else if(state == CONV1) begin
                if(buffer_wrmux % 3 == 0) fmap_buffer0[wrptr] <= d_in;
                else if(buffer_wrmux % 3 == 1) fmap_buffer1[wrptr] <= d_in;
                else if(buffer_wrmux % 3 == 2) fmap_buffer2[wrptr] <= d_in;
            end
        end
    end

    // Read
    always @(posedge clk) begin
        if(!resetn) begin
        end else begin
            if(state == CONV1 || state == CONV2) begin
                if(buffer_rdmux == 0) buffer_out <= {fmap_buffer2[rdptr], fmap_buffer1[rdptr], fmap_buffer0[rdptr]};
                else if(buffer_rdmux == 1) buffer_out <= {fmap_buffer0[rdptr], fmap_buffer2[rdptr], fmap_buffer1[rdptr]};
                else if(buffer_rdmux == 2) buffer_out <= {fmap_buffer1[rdptr], fmap_buffer0[rdptr], fmap_buffer2[rdptr]};
                else buffer_out <= buffer_out;
            end else buffer_out <= 6'b0;
        end
    end

    // Write Pointer
    always @(posedge clk) begin
        if(!resetn) begin
            wrptr <= 0; buffer_wrmux <= 0;
        end else begin
            if(state == IDLE) begin
                wrptr <= 0; buffer_wrmux <= 0;
            end else if (state == LOAD || state == CONV1 || state == CONV2) begin
                wrptr <= x1;
                buffer_wrmux <= y1 % 3;
            end else if (state == DONE) begin
                wrptr <= 0; buffer_wrmux <= 0;
            end else begin
                wrptr <= wrptr; buffer_wrmux <= buffer_wrmux;
            end
        end
    end

    // Read Pointer
    always @(posedge clk) begin
        if(!resetn) begin
            rdptr <= 0;  line_cnt <= 0; buffer_rdmux <= 0;
        end else begin
            if(state == IDLE) begin rdptr <= 0;  line_cnt <= 0; buffer_rdmux <= 0; end
            else if (state == CONV1 || state == CONV2) begin
                if(line_cnt < input_depth - 1) begin
                    rdptr <= (rdptr < input_depth - 1) ? rdptr + 1 : 0;
                    if(buffer_rdmux < 2) begin
                        if(line_cnt < 1)  buffer_rdmux <= (rdptr == 0) ? 0 : (rdptr == input_depth - 1) ? buffer_rdmux + 1 : buffer_rdmux;
                        else   buffer_rdmux <= (rdptr == input_depth - 1) ? buffer_rdmux + 1 : buffer_rdmux;
                    end else buffer_rdmux <= (rdptr == input_depth - 1) ? 0 :  buffer_rdmux;
                    line_cnt <= (line_cnt < input_depth && rdptr == 0) ? line_cnt + 1 : line_cnt;
                end else begin
                    rdptr <= rdptr; line_cnt <= line_cnt; buffer_rdmux <= buffer_rdmux;
                end
            end else begin rdptr <= 0; buffer_rdmux <= 0; line_cnt <= 0; end
        end
    end

    // target buffer
    always @(posedge clk) begin
        if(!resetn) begin
            target_counter <= 0;
            target_buffer[0] <= 0;
            target_buffer[1] <= 0;
            target_buffer[2] <= 0;
        end else begin
            if(y1 >= 3 && (state == CONV1 || state == CONV2)) begin
                if(target_counter < 3) begin
                    target_buffer[target_counter] <= buffer_out;
                    target_counter <= (rdptr == 0) ? 0 : target_counter + 1;
                end else begin
                    if(rdptr == 0) target_counter <= 0;
                    else target_counter <= target_counter;
                    target_buffer[0] <= target_buffer[1];
                    target_buffer[1] <= target_buffer[2];
                    target_buffer[2] <= buffer_out;
                end
            end else begin
                target_counter <= 0;
                target_buffer[0] <= 0;
                target_buffer[1] <= 0;
                target_buffer[2] <= 0;
            end
        end
    end
endmodule
