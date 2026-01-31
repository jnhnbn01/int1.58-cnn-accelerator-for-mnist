module fc_layer (
    input wire clk,
    input wire resetn,
    input wire [1:0] i_data,
    input wire [1:0] i_weight,
    input wire i_valid,
    output wire [7:0] o_data,
    output reg o_data_real
);

    reg o_valid;
    reg [11:0] counter, counter_d;
    reg signed [13:0] sum;
    reg signed [1:0] mul;

    always @(posedge clk) begin
        if (!resetn) mul <= 0;
        else begin
            if (i_valid) (* use_dsp = "yes" *) mul <= (i_data == 0) ? 0 : $signed(i_data) * $signed(i_weight);
        end
    end

    always @(posedge clk) begin
        if(!resetn) counter <= 0;
        else begin
            if(i_valid) begin
                if(counter==0) counter <= counter +1;
                else if(counter < 12'd2304 && counter > 0) counter <= counter + 1;
            end else if(counter == 12'd2304) counter <= 0;
        end
    end

    always @(posedge clk) begin
        if(!resetn) counter_d <= 0;
        else counter_d <= counter; // counter delayed
    end

    always @(posedge clk) begin
        if(!resetn) begin o_valid<=0; sum <=0; end
        else begin
            if(i_valid) begin
                if(counter_d == 0) begin
                    sum <= 0;
                    o_valid <=0;
                end else if(counter_d < 12'd2304 && counter_d > 0) sum <= sum + mul;
            end
            else if(counter_d == 12'd2304) begin
                sum <= sum + mul;
                o_valid <= 1;
            end
        end
    end

    reg signed [7:0] o_data_temp;
    reg o_data_real_r;
    wire signed [12:0] sum_shifted = sum >> 1; // 1bit shift
    
    // satruation
    always @(posedge clk) begin
        if(!resetn) begin
            o_data_real_r<=0;
            o_data_temp<=0;
        end else begin
            if (o_valid) begin
                o_data_real_r <=1;
                if (sum_shifted > 127) o_data_temp <= 8'b01111111;
                else if (sum_shifted < -128) o_data_temp <= 8'b10000000;
                else o_data_temp <= sum_shifted;
            end
            else begin o_data_real_r <= 0;end
        end
    end

    assign o_data= o_data_temp;

    
    always @(posedge clk) begin
        if(!resetn) o_data_real <= 0;
        else o_data_real <= o_data_real_r; // o_data_real delayed
    end

endmodule
