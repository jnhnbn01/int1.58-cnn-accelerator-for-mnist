`timescale 1ns / 1ps

module convolution (
    input wire clk,
    input wire resetn,
    input wire [17:0] i_data,
    input wire [1:0] w_data,
    input wire w_req,
    output wire signed [5:0] o_data
);
    // define kernels (-2~2)
    reg signed [1:0] gx [8:0];

    // kernel sum
    reg signed [2:0] sum1[4:0]; // -2 ~ 2 : int2
    reg signed [3:0] sum2[2:0]; 
    reg signed [4:0] sum3[1:0];
    reg signed [5:0] sum; // -9 ~ 9 : int6  
    reg signed [1:0] gx_mul [8:0]; // ternary
    reg [3:0] index = 0;


    always @(posedge clk) begin
        if(!resetn) begin
            index <= 0;
        end else begin
            if(w_req) begin
                gx[index] <= w_data;
                index <= (index < 9) ? index + 1 : index;
            end else index <= 0;
        end
    end

    // pipelining 1 : multiplication
    always @(posedge clk) begin
        (* use_dsp = "yes" *)gx_mul[0] <= (i_data[1:0] == 0) ? 0 : gx[0] * $signed(i_data[1:0]);
        (* use_dsp = "yes" *)gx_mul[1] <= (i_data[3:2] == 0) ? 0 : gx[3] * $signed(i_data[3:2]);
        (* use_dsp = "yes" *)gx_mul[2] <= (i_data[5:4] == 0) ? 0 : gx[6] * $signed(i_data[5:4]);
        (* use_dsp = "yes" *)gx_mul[3] <= (i_data[7:6] == 0) ? 0 : gx[1] * $signed(i_data[7:6]);
        (* use_dsp = "yes" *)gx_mul[4] <= (i_data[9:8] == 0) ? 0 : gx[4] * $signed(i_data[9:8]);
        (* use_dsp = "yes" *)gx_mul[5] <= (i_data[11:10] == 0) ? 0 : gx[7] * $signed(i_data[11:10]);
        (* use_dsp = "yes" *)gx_mul[6] <= (i_data[13:12] == 0) ? 0 : gx[2] * $signed(i_data[13:12]);
        (* use_dsp = "yes" *)gx_mul[7] <= (i_data[15:14] == 0) ? 0 : gx[5] * $signed(i_data[15:14]);
        (* use_dsp = "yes" *)gx_mul[8] <= (i_data[17:16] == 0) ? 0 : gx[8] * $signed(i_data[17:16]);
    end

    // pipelining  : adder tree
    always @(posedge clk ) begin
        if(!resetn) begin
            sum1[0] <= 0; sum1[1] <= 0; sum1[2] <= 0; sum1[3] <= 0;
        end else begin
            sum1[0] <= gx_mul[0] + gx_mul[1];
            sum1[1] <= gx_mul[2] + gx_mul[3];
            sum1[2] <= gx_mul[4] + gx_mul[5];
            sum1[3] <= gx_mul[6] + gx_mul[7];
            sum1[4] <= gx_mul[8] + 0;
        end
    end
    always @(posedge clk ) begin
        if(!resetn) begin
            sum2[0] <= 0; sum2[1] <= 0;
        end else begin
            sum2[0] <= sum1[0] + sum1[1];
            sum2[1] <= sum1[2] + sum1[3];
            sum2[2] <= sum1[4] + 0;
        end
    end
    
    always @(posedge clk ) begin
        if(!resetn) begin
            sum3[0] <= 0; sum3[1] <= 0;
        end else begin
          
            sum3[0] <= sum2[0] + sum2[2];
            sum3[1] <= sum2[1] + 0;
        end
    end

    always @(posedge clk) begin
        if(!resetn) begin
            sum <= 0;
        end else begin
            sum <= sum3[0] + sum3[1];
        end
    end
    
    assign o_data = sum;
endmodule
