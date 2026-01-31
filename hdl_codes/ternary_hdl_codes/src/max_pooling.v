`timescale 1ns / 1ps

module max_pooling(
    input wire clk,
    input wire resetn,
    input wire signed [1:0] i_d,
    input wire i_v,
    output reg signed [1:0] o_d,
    output reg o_v
);
    reg [4:0] cnt;
    reg x_1;
    reg signed [1:0] buffer1[0:23];
    reg signed [1:0] buffer2, buffer3;
    reg o_v_temp;
    

    always @(posedge clk) begin
        if(!resetn) begin x_1<=0; buffer2<=0; buffer3<=0; cnt<=0;end
        else begin
            if(i_v==1 && x_1 == 0)begin // 1 line read 
                buffer1[cnt]<=i_d;
                cnt <= (cnt<5'd23) ? cnt+1 : 0;
                x_1 <= (cnt<5'd23) ? 0 : 1'b1;
            end else if (i_v==1 && x_1 == 1) begin // next line read & compare
                if (cnt[0]==0)begin
                    buffer2 <= (i_d >= buffer1[cnt]) ? i_d : buffer1[cnt];
                    cnt<=cnt+1;
                end else if (cnt[0]==1)begin
                    buffer3 <= (i_d >= buffer1[cnt]) ? i_d : buffer1[cnt];
                    cnt <= (cnt<5'd23) ? cnt+1:0;
                    x_1 <= (cnt<5'd23) ? 1 : 0;
                end
            end
        end
    end

    always @(posedge clk) begin
        if(!resetn) o_v_temp <= 0;
        else o_v_temp <= (x_1==1 && cnt[0]==1); // delay 1 cycle
    end
    always @(posedge clk) begin
        if(!resetn) o_v <= 0;
        else o_v <= o_v_temp; // delay 2 cycle
    end

    always @(posedge clk) begin
        if(!resetn) o_d <= 0;
        else begin
            if(cnt[0] == 0) o_d <= (buffer2 > buffer3) ? buffer2 : buffer3;
            else o_d <= o_d;
        end 
    end
endmodule