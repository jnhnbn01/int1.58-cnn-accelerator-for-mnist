`timescale 1ns / 1ps

module top_cnn_accelerator(
    input wire clk,
    input wire resetn_d,
    input  wire start,
    output wire done,
    input wire s1_clka,
    input wire s1_ena,
    input wire [1:0] s1_wea,
    input wire [16:0] s1_addra,
    input wire [7:0] s1_dina,
    output wire [7:0] s1_douta,

    input wire s6_clka,
    input wire s6_ena,
    input wire [1:0] s6_wea,
    input wire [10:0] s6_addra,
    input wire [7:0] s6_dina,
    output wire [7:0] s6_douta,

    input wire s8_ena,
    input wire [3:0] s8_wea,
    input wire [9:0] s8_addra,
    input wire [7:0] s8_dina,
    output wire [7:0] s8_douta,

    input wire s7_clka_1,s7_clka_2,s7_clka_3,s7_clka_4,s7_clka_5,s7_clka_6,s7_clka_7,s7_clka_8,s7_clka_9,s7_clka_10,s8_clka,
    input wire s7_ena_1,s7_ena_2,s7_ena_3,s7_ena_4,s7_ena_5,s7_ena_6,s7_ena_7,s7_ena_8,s7_ena_9,s7_ena_10,
    input wire [1:0] s7_wea_1,s7_wea_2,s7_wea_3,s7_wea_4,s7_wea_5,s7_wea_6,s7_wea_7,s7_wea_8,s7_wea_9,s7_wea_10,
    input wire [11:0] s7_addra_1,s7_addra_2,s7_addra_3,s7_addra_4,s7_addra_5,s7_addra_6,s7_addra_7,s7_addra_8,s7_addra_9,s7_addra_10,
    input wire [7:0] s7_dina_1,s7_dina_2,s7_dina_3,s7_dina_4,s7_dina_5,s7_dina_6,s7_dina_7,s7_dina_8,s7_dina_9,s7_dina_10,
    output wire [7:0] s7_douta_1,s7_douta_2,s7_douta_3,s7_douta_4,s7_douta_5,s7_douta_6,s7_douta_7,s7_douta_8,s7_douta_9,s7_douta_10
);
    // async to sync reset
    reg resetn_s;
    always @(posedge clk) begin
        resetn_s <= resetn_d;
    end


    wire conv_done[7:0]; // done signals for each PEs
    reg resetn_pe[7:0]; // negative reset signals for each PEs
    assign done_led = done;

    // Input Fmap BRAM
    wire s1_en, s1_we;
    wire [1:0] s1_web;
    wire [16:0] s1_addr;
    wire [7:0] s1_din, s1_dout;
    assign s1_we = 1'b0;
    assign s1_web = {2{s1_we}};

    // Conv1 Result Fmap BRAM 
    wire s2_1_ena, s2_2_ena, s2_3_ena, s2_4_ena, s2_5_ena, s2_6_ena, s2_7_ena, s2_8_ena; // Port A
    wire s2_1_wea, s2_2_wea, s2_3_wea, s2_4_wea, s2_5_wea, s2_6_wea, s2_7_wea, s2_8_wea;
    wire [9:0] s2_1_addra, s2_2_addra, s2_3_addra, s2_4_addra, s2_5_addra, s2_6_addra, s2_7_addra, s2_8_addra;
    wire [7:0] s2_1_dina, s2_2_dina, s2_3_dina, s2_4_dina, s2_5_dina, s2_6_dina, s2_7_dina, s2_8_dina;
    wire [7:0] s2_1_douta, s2_2_douta, s2_3_douta, s2_4_douta, s2_5_douta, s2_6_douta, s2_7_douta, s2_8_douta;
    assign {s2_1_wea, s2_2_wea, s2_3_wea, s2_4_wea, s2_5_wea, s2_6_wea, s2_7_wea, s2_8_wea} = 8'b00000000;

    wire s2_1_enb, s2_2_enb, s2_3_enb, s2_4_enb, s2_5_enb, s2_6_enb, s2_7_enb, s2_8_enb; // Port B
    wire s2_1_web, s2_2_web, s2_3_web, s2_4_web, s2_5_web, s2_6_web, s2_7_web, s2_8_web;
    wire [9:0] s2_1_addrb, s2_2_addrb, s2_3_addrb, s2_4_addrb, s2_5_addrb, s2_6_addrb, s2_7_addrb, s2_8_addrb;
    wire [7:0] s2_1_dinb, s2_2_dinb, s2_3_dinb, s2_4_dinb, s2_5_dinb, s2_6_dinb, s2_7_dinb, s2_8_dinb;
    wire [7:0] s2_1_doutb, s2_2_doutb, s2_3_doutb, s2_4_doutb, s2_5_doutb, s2_6_doutb, s2_7_doutb, s2_8_doutb;
    assign s2_1_web = s2_1_enb;
    assign s2_2_web = s2_2_enb;
    assign s2_3_web = s2_3_enb;
    assign s2_4_web = s2_4_enb;
    assign s2_5_web = s2_5_enb;
    assign s2_6_web = s2_6_enb;
    assign s2_7_web = s2_7_enb;
    assign s2_8_web = s2_8_enb;

    // Conv1, Conv2 Weight BRAM
    wire s6_en, s6_we;
    wire [10:0] s6_addr;
    wire [7:0] s6_din, s6_dout;
    reg [8:0] conv_w_channel;
    reg [4:0] conv_w_index;
    wire [7:0] w_data[7:0];
    reg w_req[7:0];

    wire resetn;
    assign resetn = (resetn_d == 1'b0 || state == DONE || state == DONE1) ? 1'b0 : 1'b1;

    assign s6_en = 1'b1;
    assign s6_we = 1'b0;
    assign s6_web = {2{s6_we}};
    assign s6_addr = conv_w_channel * 9 + conv_w_index;

    assign w_data[0] = (w_req[0]) ? s6_dout : 0;
    assign w_data[1] = (w_req[1]) ? s6_dout : 0;
    assign w_data[2] = (w_req[2]) ? s6_dout : 0;
    assign w_data[3] = (w_req[3]) ? s6_dout : 0;
    assign w_data[4] = (w_req[4]) ? s6_dout : 0;
    assign w_data[5] = (w_req[5]) ? s6_dout : 0;
    assign w_data[6] = (w_req[6]) ? s6_dout : 0;
    assign w_data[7] = (w_req[7]) ? s6_dout : 0;

    // FSM
    localparam [2:0] IDLE = 3'b000, CONV1 = 3'b001, CONV1_DONE = 3'b010, CONV2 = 3'b011, CONV2_DONE = 3'b100, POOL = 3'b101, DONE = 3'b110, DONE1=3'b111;
    assign conv1 = (state == CONV1);
    assign conv1done = (state == CONV1_DONE);
    assign conv2 = (state == CONV2);
    assign conv2done = (state == CONV2_DONE);
    assign pool = (state == POOL);
    assign done1 = (state == DONE1);
    assign o_valid = o_v_1;

    reg [2:0] state;
    reg [4:0] conv2_counter;

    always @(posedge clk) begin
        if(!resetn_s) state <= IDLE;
        else begin
            case(state)
                IDLE: state <= (start) ? CONV1 : IDLE;
                CONV1 : state <= (conv_done[0] & conv_done[1] & conv_done[2] & conv_done[3] & conv_done[4] & conv_done[5] & conv_done[6] & conv_done[7]) ? CONV1_DONE : CONV1;
                CONV1_DONE : state <= (!conv_done[0] & !conv_done[1] & !conv_done[2] & !conv_done[3] & !conv_done[4] & !conv_done[5] & !conv_done[6] & !conv_done[7]) ? CONV2 : CONV1_DONE;
                CONV2 : state <= (conv_done[0] & conv_done[1] & conv_done[2] & conv_done[3] & conv_done[4] & conv_done[5] & conv_done[6] & conv_done[7]) ? CONV2_DONE : CONV2;
                CONV2_DONE : state <= (conv2_counter >= 16) ? POOL : (!conv_done[0] & !conv_done[1] & !conv_done[2] & !conv_done[3] & !conv_done[4] & !conv_done[5] & !conv_done[6] & !conv_done[7]) ? CONV2 : CONV2_DONE;
                POOL : state <= (x==10) ? DONE1 : POOL;
                DONE1 : state <= (batch==99) ? DONE : CONV1;
                DONE : state <= (start)? CONV1: DONE;
                default: state <= IDLE;
            endcase
        end
    end

    reg [6:0] batch;
    reg [16:0] addr_offset;
    reg [9:0] output_offset;
    always @(posedge clk) begin
        if(!resetn_s) begin batch <=0; addr_offset <= 0; output_offset = 0; end
        else begin
            if( state==DONE1)begin
                if(batch < 7'd100) begin
                    batch <= batch+1;
                    addr_offset <= addr_offset +  784;
                    output_offset <= output_offset + 10;
                end else if(batch ==7'd100) begin
                    batch <= batch;
                    addr_offset <= addr_offset ;
                    output_offset <= output_offset + 10;
                end
            end else if(state == DONE) begin batch <=0; addr_offset <= 0; output_offset = 0; end
        end
    end

    assign done = (state==DONE);
        reg done_neg; wire done_d; //deboucing done
    always @(posedge clk) begin
        if(!resetn_s) done_neg <= 1;
        else done_neg <= ~done;
    end
    
    assign done_d = done_neg & done;
    
    reg start_flag;

    always @(posedge clk) begin
        if(!resetn_s) start_flag <=0;
        else begin
            if(start) start_flag<= 1;
            else begin
                if(done_d) start_flag  <= 0;
                else start_flag <= start_flag;
            end 
            
        end
    end


    // resetn PEs & start PEs & conv2 iteration 16_times
    always @(posedge clk) begin
        if(!resetn) begin {resetn_pe[0], resetn_pe[1], resetn_pe[2], resetn_pe[3], resetn_pe[4], resetn_pe[5], resetn_pe[6], resetn_pe[7]} <= 8'b11111111;
        end
        else begin
            if(state == CONV1_DONE) begin
                {resetn_pe[0], resetn_pe[1], resetn_pe[2], resetn_pe[3], resetn_pe[4], resetn_pe[5], resetn_pe[6], resetn_pe[7]} <= 8'b00000000;
            end else if(state == CONV2) begin
                {resetn_pe[0], resetn_pe[1], resetn_pe[2], resetn_pe[3], resetn_pe[4], resetn_pe[5], resetn_pe[6], resetn_pe[7]} <= 8'b11111111;
            end else if(state == CONV2_DONE) begin
                {resetn_pe[0], resetn_pe[1], resetn_pe[2], resetn_pe[3], resetn_pe[4], resetn_pe[5], resetn_pe[6], resetn_pe[7]} <= 8'b00000000;
            end else begin
                {resetn_pe[0], resetn_pe[1], resetn_pe[2], resetn_pe[3], resetn_pe[4], resetn_pe[5], resetn_pe[6], resetn_pe[7]} <= 8'b11111111;
            end
        end
    end

    reg conv2_flag;
    always @(posedge clk) begin
        if(state == CONV2) begin
            if(y2 == input_depth - 2 && x2 == 0) begin
                conv2_counter <= (conv2_counter < 16 && conv2_flag) ? conv2_counter + 1 : conv2_counter;
                conv2_flag <= ~conv2_flag;
            end
            else conv2_counter <= conv2_counter;
        end else if(state == CONV2_DONE || state == POOL) conv2_counter <= conv2_counter;
        else begin conv2_flag <= 0; conv2_counter <= 0; end
    end

    // Conv1, 2 Weight BRAM Addressing
    always @(posedge clk) begin
        if(!resetn) begin
            {w_req[0], w_req[1], w_req[2], w_req[3], w_req[4], w_req[5], w_req[6], w_req[7]} <= 8'b00000000;
            conv_w_channel <= 0; conv_w_index <= 0;
        end else begin
            if(state == CONV1) begin
                if(conv_w_index < 8) begin
                    conv_w_index <= conv_w_index + 1;
                    if(conv_w_index == 0) begin
                        w_req[conv_w_channel] <= 1;
                        if(conv_w_channel > 0) w_req[conv_w_channel - 1] <= 0;
                    end else begin
                        w_req[conv_w_channel] <= w_req[conv_w_channel];
                    end
                end else begin
                    conv_w_channel <= (conv_w_channel < 8) ? conv_w_channel + 1 : conv_w_channel;
                    conv_w_index <= (conv_w_channel < 8) ? 0 : conv_w_index;
                end
            end else if(state == CONV1_DONE) begin
                {w_req[0], w_req[1], w_req[2], w_req[3], w_req[4], w_req[5], w_req[6], w_req[7]} <= 8'b00000000;
                conv_w_index <= 0;
            end else if(state == CONV2) begin
                if(conv_w_index < 8) begin
                    conv_w_index <= conv_w_index + 1;
                    if(conv_w_index == 0) begin
                        w_req[conv_w_channel - 8*(conv2_counter+1)] <= 1;
                        if(conv_w_channel- 8*(conv2_counter+1) > 0) w_req[conv_w_channel - 8*(conv2_counter+1) - 1] <= 0;
                    end else begin
                        w_req[conv_w_channel - 8*(conv2_counter + 1)] <= w_req[conv_w_channel - 8*(conv2_counter+1)];
                    end
                end else begin
                    conv_w_channel <= (conv_w_channel < 8 + 8*(conv2_counter+1)  ) ? conv_w_channel + 1 : conv_w_channel;
                    conv_w_index <= (conv_w_channel < 8 + 8*(conv2_counter+1) ) ? 0 : conv_w_index;
                end
            end else if(state == CONV2_DONE) begin
                {w_req[0], w_req[1], w_req[2], w_req[3], w_req[4], w_req[5], w_req[6], w_req[7]} <= 8'b00000000;
                conv_w_index <= 0;
            end else begin
                conv_w_index <= 0;
                conv_w_channel <= 0;
            end
        end
    end

    // Conv1, 2 data depth selection    
    wire [4:0] input_depth;
    assign input_depth = (state == CONV1) ? 5'd28 :
    (state == CONV2) ? 5'd26 : 5'd10;

    // Conv1, 2 BRAM data seletion
    wire signed [7:0] PE_din[7:0];
    wire signed [19:0] PE_dout[7:0];
    wire PE_enin[7:0];
    wire PE_enout[7:0];

    // Quantizer
    reg signed [9:0] dout_conv1_clipped [7:0];
    reg signed [7:0] dout_conv1 [7:0];
    reg signed [13:0] dout_conv2_clipped;
    reg signed [7:0] dout_conv2;

    always @(posedge clk) begin
        if(!resetn) begin
            dout_conv1_clipped[0] <= 0; dout_conv1_clipped[1] <= 0; dout_conv1_clipped[2] <= 0; dout_conv1_clipped[3] <= 0;
            dout_conv1_clipped[4] <= 0; dout_conv1_clipped[5] <= 0; dout_conv1_clipped[6] <= 0; dout_conv1_clipped[7] <= 0;
            dout_conv2_clipped <= 0;
        end else begin
            if(state == CONV1) begin // clipping
                dout_conv1_clipped[0] <= PE_dout[0] >>> 10;
                dout_conv1_clipped[1] <= PE_dout[1] >>> 10;
                dout_conv1_clipped[2] <= PE_dout[2] >>> 10;
                dout_conv1_clipped[3] <= PE_dout[3] >>> 10;
                dout_conv1_clipped[4] <= PE_dout[4] >>> 10;
                dout_conv1_clipped[5] <= PE_dout[5] >>> 10;
                dout_conv1_clipped[6] <= PE_dout[6] >>> 10;
                dout_conv1_clipped[7] <= PE_dout[7] >>> 10;
            end else if(state == CONV2) begin // sum & clipping
                dout_conv2_clipped <= (PE_dout[0] + PE_dout[1] + PE_dout[2] + PE_dout[3] + PE_dout[4] + PE_dout[5] + PE_dout[6] + PE_dout[7]) >>> 10;
            end
        end
    end
    always @(posedge clk) begin
        if(!resetn) begin
            dout_conv1[0] <= 0; dout_conv1[1] <= 0; dout_conv1[2] <= 0; dout_conv1[3] <= 0;
            dout_conv1[4] <= 0; dout_conv1[5] <= 0; dout_conv1[6] <= 0; dout_conv1[7] <= 0;
            dout_conv2 <= 0;
        end else begin
            if(state == CONV1) begin // saturation + relu
                dout_conv1[0] <= (dout_conv1_clipped[0] > 8'sd127) ? 8'sd127 : (dout_conv1_clipped[0] < 0) ? 0 : dout_conv1_clipped[0];
                dout_conv1[1] <= (dout_conv1_clipped[1] > 8'sd127) ? 8'sd127 : (dout_conv1_clipped[1] < 0) ? 0 : dout_conv1_clipped[1];
                dout_conv1[2] <= (dout_conv1_clipped[2] > 8'sd127) ? 8'sd127 : (dout_conv1_clipped[2] < 0) ? 0 : dout_conv1_clipped[2];
                dout_conv1[3] <= (dout_conv1_clipped[3] > 8'sd127) ? 8'sd127 : (dout_conv1_clipped[3] < 0) ? 0 : dout_conv1_clipped[3];
                dout_conv1[4] <= (dout_conv1_clipped[4] > 8'sd127) ? 8'sd127 : (dout_conv1_clipped[4] < 0) ? 0 : dout_conv1_clipped[4];
                dout_conv1[5] <= (dout_conv1_clipped[5] > 8'sd127) ? 8'sd127 : (dout_conv1_clipped[5] < 0) ? 0 : dout_conv1_clipped[5];
                dout_conv1[6] <= (dout_conv1_clipped[6] > 8'sd127) ? 8'sd127 : (dout_conv1_clipped[6] < 0) ? 0 : dout_conv1_clipped[6];
                dout_conv1[7] <= (dout_conv1_clipped[7] > 8'sd127) ? 8'sd127 : (dout_conv1_clipped[7] < 0) ? 0 : dout_conv1_clipped[7];
            end else if(state == CONV2) begin // saturated + relu
                dout_conv2 <= (dout_conv2_clipped > 8'sd127) ? 8'sd127 : (dout_conv2_clipped  < 8'sb0) ? 0 : dout_conv2_clipped;
            end else begin
                dout_conv1[0] <= 0; dout_conv1[1] <= 0; dout_conv1[2] <= 0; dout_conv1[3] <= 0;
                dout_conv1[4] <= 0; dout_conv1[5] <= 0; dout_conv1[6] <= 0; dout_conv1[7] <= 0;
                dout_conv2 <= 0;
            end
        end
    end

    assign PE_din[0] = (state == CONV1) ? s1_dout : (state == CONV2) ? s2_1_douta : 0;
    assign PE_din[1] = (state == CONV1) ? s1_dout : (state == CONV2) ? s2_2_douta : 0;
    assign PE_din[2] = (state == CONV1) ? s1_dout : (state == CONV2) ? s2_3_douta : 0;
    assign PE_din[3] = (state == CONV1) ? s1_dout : (state == CONV2) ? s2_4_douta : 0;
    assign PE_din[4] = (state == CONV1) ? s1_dout : (state == CONV2) ? s2_5_douta : 0;
    assign PE_din[5] = (state == CONV1) ? s1_dout : (state == CONV2) ? s2_6_douta : 0;
    assign PE_din[6] = (state == CONV1) ? s1_dout : (state == CONV2) ? s2_7_douta : 0;
    assign PE_din[7] = (state == CONV1) ? s1_dout : (state == CONV2) ? s2_8_douta : 0;

    assign s2_1_dinb = (state == CONV1) ? dout_conv1[0] : 0;
    assign s2_2_dinb = (state == CONV1) ? dout_conv1[1] : 0;
    assign s2_3_dinb = (state == CONV1) ? dout_conv1[2] : 0;
    assign s2_4_dinb = (state == CONV1) ? dout_conv1[3] : 0;
    assign s2_5_dinb = (state == CONV1) ? dout_conv1[4] : 0;
    assign s2_6_dinb = (state == CONV1) ? dout_conv1[5] : 0;
    assign s2_7_dinb = (state == CONV1) ? dout_conv1[6] : 0;
    assign s2_8_dinb = (state == CONV1) ? dout_conv1[7] : 0;


    reg [4:0] x1, y1, x2, y2;
    reg [9:0] ifmap_rd_offset;
    // Conv1, 2 BRAM Read Addressing
    always @(posedge clk) begin
        if(!resetn) begin
            x1 <= 0; y1 <= 0; ifmap_rd_offset = 0;
        end else begin
            if (PE_enin[0] || PE_enout[0]) begin
                x1 <= (x1 < input_depth - 1) ? x1 + 1 : 0;
                y1 <= (y1 < input_depth - 1 && x1 == input_depth - 1) ? y1 + 1 : y1;
                ifmap_rd_offset <= (y1 < input_depth - 1 && x1 == input_depth - 1) ? ifmap_rd_offset + input_depth : ifmap_rd_offset;
            end else begin x1 <= 0; y1 <= 0; ifmap_rd_offset <= 0;  end
        end
    end

    // Conv1, 2 BRAM Write  Addressing
    always @(posedge clk) begin
        if(!resetn) begin x2 <= 0; y2 <= 0; end
        else begin
            if(PE_enout[0] ) begin
                if(y1 == 3 && x1 >= 11) x2 <= (x2 < input_depth - 3 && y2 < input_depth - 2) ? x2 + 1 : 0;
                else if (y1 > 3 && (x1 >= 11 || x1 <= 8)) x2 <= (x2 < input_depth - 3 && y2 < input_depth - 2) ? x2 + 1 : 0;
                else x2 <= x2;
                y2 <= (x2 == input_depth - 3 && y2 < input_depth - 2) ? y2 + 1 : y2;
            end else begin x2 <= 0; y2 <= 0; end
        end
    end

    wire [9:0] read_addr, write_addr;
    wire [16:0] read_addr_1;
    wire pooling_ready;
    assign pooling_ready = (y1 > 3) ? (x1 >= 11 || x1 <= 8) ? 1 : 0 : (y1 == 3) ? (x1 >= 11) ? 1 : 0 : 0;
    assign read_addr = x1 + ifmap_rd_offset;
    assign read_addr_1=  read_addr + addr_offset;
    assign write_addr = x2 + (input_depth-2)*y2;

    // write address
    assign s2_1_addrb = (state == CONV1) ? write_addr : 0;
    assign s2_2_addrb = (state == CONV1) ? write_addr : 0;
    assign s2_3_addrb = (state == CONV1) ? write_addr : 0;
    assign s2_4_addrb = (state == CONV1) ? write_addr : 0;
    assign s2_5_addrb = (state == CONV1) ? write_addr : 0;
    assign s2_6_addrb = (state == CONV1) ? write_addr : 0;
    assign s2_7_addrb = (state == CONV1) ? write_addr : 0;
    assign s2_8_addrb = (state == CONV1) ? write_addr : 0;

    // read address
    assign s1_addr = (state == CONV1) ? read_addr_1 : 0;
    assign s2_1_addra = (state == CONV2) ? read_addr : 0;
    assign s2_2_addra = (state == CONV2) ? read_addr : 0;
    assign s2_3_addra = (state == CONV2) ? read_addr : 0;
    assign s2_4_addra = (state == CONV2) ? read_addr : 0;
    assign s2_5_addra = (state == CONV2) ? read_addr : 0;
    assign s2_6_addra = (state == CONV2) ? read_addr : 0;
    assign s2_7_addra = (state == CONV2) ? read_addr : 0;
    assign s2_8_addra = (state == CONV2) ? read_addr : 0;

    // input bram enable
    assign s1_en = (state == CONV1) ? PE_enin[0] : 0;
    assign s2_1_ena = (state == CONV2) ? PE_enin[0] : 0;
    assign s2_2_ena = (state == CONV2) ? PE_enin[1] : 0;
    assign s2_3_ena = (state == CONV2) ? PE_enin[2] : 0;
    assign s2_4_ena = (state == CONV2) ? PE_enin[3] : 0;
    assign s2_5_ena = (state == CONV2) ? PE_enin[4] : 0;
    assign s2_6_ena = (state == CONV2) ? PE_enin[5] : 0;
    assign s2_7_ena = (state == CONV2) ? PE_enin[6] : 0;
    assign s2_8_ena = (state == CONV2) ? PE_enin[7] : 0;

    // output bram enable
    assign s2_1_enb = (state == CONV1) ? PE_enout[0] : 0;
    assign s2_2_enb = (state == CONV1) ? PE_enout[1] : 0;
    assign s2_3_enb = (state == CONV1) ? PE_enout[2] : 0;
    assign s2_4_enb = (state == CONV1) ? PE_enout[3] : 0;
    assign s2_5_enb = (state == CONV1) ? PE_enout[4] : 0;
    assign s2_6_enb = (state == CONV1) ? PE_enout[5] : 0;
    assign s2_7_enb = (state == CONV1) ? PE_enout[6] : 0;
    assign s2_8_enb = (state == CONV1) ? PE_enout[7] : 0;

    top_convolution_core#(.DATA_WIDTH(8))
    PE0 (.clk(clk), .input_depth(input_depth), .resetn(resetn_pe[0] & resetn), .start(start_flag), .d_in(PE_din[0]), .w_in(w_data[0]), .w_req(w_req[0]), .conv_out(PE_dout[0]), .x1(x1), .y1(y1), .s1_en(PE_enin[0]), .s2_en(PE_enout[0]), .done(conv_done[0]));

    top_convolution_core#(.DATA_WIDTH(8))
    PE1 (.clk(clk), .input_depth(input_depth), .resetn(resetn_pe[1] & resetn), .start(start_flag), .d_in(PE_din[1]), .w_in(w_data[1]), .w_req(w_req[1]), .conv_out(PE_dout[1]), .x1(x1), .y1(y1), .s1_en(PE_enin[1]), .s2_en(PE_enout[1]), .done(conv_done[1]));

    top_convolution_core#(.DATA_WIDTH(8))
    PE2 (.clk(clk), .input_depth(input_depth), .resetn(resetn_pe[2] & resetn), .start(start_flag), .d_in(PE_din[2]), .w_in(w_data[2]), .w_req(w_req[2]), .conv_out(PE_dout[2]), .x1(x1), .y1(y1), .s1_en(PE_enin[2]), .s2_en(PE_enout[2]), .done(conv_done[2]));

    top_convolution_core#(.DATA_WIDTH(8))
    PE3 (.clk(clk), .input_depth(input_depth), .resetn(resetn_pe[3] & resetn), .start(start_flag), .d_in(PE_din[3]), .w_in(w_data[3]), .w_req(w_req[3]), .conv_out(PE_dout[3]), .x1(x1), .y1(y1), .s1_en(PE_enin[3]), .s2_en(PE_enout[3]), .done(conv_done[3]));

    top_convolution_core#(.DATA_WIDTH(8))
    PE4 (.clk(clk), .input_depth(input_depth), .resetn(resetn_pe[4] & resetn), .start(start_flag), .d_in(PE_din[4]), .w_in(w_data[4]), .w_req(w_req[4]), .conv_out(PE_dout[4]), .x1(x1), .y1(y1), .s1_en(PE_enin[4]), .s2_en(PE_enout[4]), .done(conv_done[4]));

    top_convolution_core#(.DATA_WIDTH(8))
    PE5 (.clk(clk), .input_depth(input_depth), .resetn(resetn_pe[5] & resetn), .start(start_flag), .d_in(PE_din[5]), .w_in(w_data[5]), .w_req(w_req[5]), .conv_out(PE_dout[5]), .x1(x1), .y1(y1), .s1_en(PE_enin[5]), .s2_en(PE_enout[5]), .done(conv_done[5]));

    top_convolution_core#(.DATA_WIDTH(8))
    PE6 (.clk(clk), .input_depth(input_depth), .resetn(resetn_pe[6] & resetn), .start(start_flag), .d_in(PE_din[6]), .w_in(w_data[6]), .w_req(w_req[6]), .conv_out(PE_dout[6]), .x1(x1), .y1(y1), .s1_en(PE_enin[6]), .s2_en(PE_enout[6]), .done(conv_done[6]));

    top_convolution_core#(.DATA_WIDTH(8))
    PE7 (.clk(clk), .input_depth(input_depth), .resetn(resetn_pe[7] & resetn), .start(start_flag), .d_in(PE_din[7]), .w_in(w_data[7]), .w_req(w_req[7]), .conv_out(PE_dout[7]), .x1(x1), .y1(y1), .s1_en(PE_enin[7]), .s2_en(PE_enout[7]), .done(conv_done[7]));

    BRAM1 input_fmap (
        .clka(s1_clka), .ena(s1_ena), .wea(s1_wea), .addra(s1_addra), .dina(s1_dina), .douta(s1_douta), // A Port
        .clkb(clk), .enb(s1_en), .web(s1_web), .addrb(s1_addr), .dinb(s1_din), .doutb(s1_dout) // B Port
    );

    BRAM2_1 fmap1_1 (
        .clka(clk), .ena(s2_1_ena),  .wea(s2_1_wea), .addra(s2_1_addra), .dina(s2_1_dina), .douta(s2_1_douta), // A Port
        .clkb(clk), .enb(s2_1_enb), .web(s2_1_web), .addrb(s2_1_addrb), .dinb(s2_1_dinb), .doutb(s2_1_doutb) // B Port
    );

    BRAM2_2 fmap1_2 (
        .clka(clk), .ena(s2_2_ena), .wea(s2_2_wea), .addra(s2_2_addra), .dina(s2_2_dina), .douta(s2_2_douta), // A Port
        .clkb(clk), .enb(s2_2_enb), .web(s2_2_web), .addrb(s2_2_addrb),.dinb(s2_2_dinb), .doutb(s2_2_doutb) // B Port
    );

    BRAM2_3 fmap1_3 (
        .clka(clk), .ena(s2_3_ena), .wea(s2_3_wea), .addra(s2_3_addra), .dina(s2_3_dina), .douta(s2_3_douta), // A Port
        .clkb(clk), .enb(s2_3_enb), .web(s2_3_web), .addrb(s2_3_addrb),.dinb(s2_3_dinb), .doutb(s2_3_doutb) // B Port
    );

    BRAM2_4 fmap1_4 (
        .clka(clk), .ena(s2_4_ena), .wea(s2_4_wea), .addra(s2_4_addra), .dina(s2_4_dina), .douta(s2_4_douta), // A Port
        .clkb(clk), .enb(s2_4_enb), .web(s2_4_web), .addrb(s2_4_addrb),.dinb(s2_4_dinb), .doutb(s2_4_doutb) // B Port
    );

    BRAM2_5 fmap1_5 (
        .clka(clk), .ena(s2_5_ena), .wea(s2_5_wea), .addra(s2_5_addra), .dina(s2_5_dina), .douta(s2_5_douta), // A Port
        .clkb(clk), .enb(s2_5_enb), .web(s2_5_web), .addrb(s2_5_addrb),.dinb(s2_5_dinb), .doutb(s2_5_doutb) // B Port
    );

    BRAM2_6 fmap1_6 (
        .clka(clk), .ena(s2_6_ena), .wea(s2_6_wea), .addra(s2_6_addra), .dina(s2_6_dina), .douta(s2_6_douta), // A Port
        .clkb(clk), .enb(s2_6_enb), .web(s2_6_web), .addrb(s2_6_addrb),.dinb(s2_6_dinb), .doutb(s2_6_doutb) // B Port
    );

    BRAM2_7 fmap1_7 (
        .clka(clk), .ena(s2_7_ena), .wea(s2_7_wea), .addra(s2_7_addra), .dina(s2_7_dina), .douta(s2_7_douta), // A Port
        .clkb(clk), .enb(s2_7_enb), .web(s2_7_web), .addrb(s2_7_addrb),.dinb(s2_7_dinb), .doutb(s2_7_doutb) // B Port
    );

    BRAM2_8 fmap1_8 (
        .clka(clk), .ena(s2_8_ena), .wea(s2_8_wea), .addra(s2_8_addra), .dina(s2_8_dina), .douta(s2_8_douta), // A Port
        .clkb(clk), .enb(s2_8_enb), .web(s2_8_web), .addrb(s2_8_addrb),.dinb(s2_8_dinb), .doutb(s2_8_doutb) // B Port
    );

    BRAM6 conv_weight (
        .clka(s6_clka), .ena(s6_ena), .wea(s6_wea), .addra(s6_addra), .dina(s6_dina), .douta(s6_douta), // A Port
        .clkb(clk), .enb(s6_en), .web(s6_we), .addrb(s6_addr), .dinb(s6_din), .doutb(s6_dout) // B Port
    );

    wire [7:0] o_data_0;
    wire o_valid_1;

    max_pooling max(
        .clk(clk), .resetn(resetn), .i_d(dout_conv2), .i_v(pooling_ready && state == CONV2), .o_d(o_data_0), .o_v(o_valid_1)
    );


    reg [11:0] counter_x; //0~2303
    reg flag;
    always @(posedge clk) begin
        if(!resetn)begin counter_x<=0; flag=0;end
        else begin
            if (state==CONV2 && o_valid_1==1) begin
                if(counter_x < 12'd2303) counter_x <= counter_x +1;
                else if (counter_x == 12'd2303 && flag==0) flag=1;
                else if (counter_x == 12'd2303 && flag==1) begin
                    counter_x <= 12'd2304;
                    flag=0;
                end
            end
        end
    end

    wire s8_en;
    wire s8_we;
    wire [3:0] s8_web;
    reg [7:0] s8_din;
    wire [9:0] s8_addr;
    assign s8_en=(o_v_1==1 && x<=9);
    assign s8_we=(o_v_1==1 && x<=9);
    assign s8_web={4{s8_we}};

    reg [3:0] x;
    assign s8_addr=x + output_offset ;

    always @(posedge clk) begin
        if (!resetn) x <= 4'd0;
        else begin
            if (o_v_1==1) x <= (x < 10) ? x + 1 : 10;
            else x <= 0;
        end
    end

    wire [7:0] o_data_1,o_data_2,o_data_3,o_data_4,o_data_5,o_data_6,o_data_7,o_data_8,o_data_9,o_data_10;

    always @(*) begin
        case (x)
            4'd0:  s8_din = o_data_1;
            4'd1:  s8_din = o_data_2;
            4'd2:  s8_din = o_data_3;
            4'd3:  s8_din = o_data_4;
            4'd4:  s8_din = o_data_5;
            4'd5:  s8_din = o_data_6;
            4'd6:  s8_din = o_data_7;
            4'd7:  s8_din = o_data_8;
            4'd8:  s8_din = o_data_9;
            4'd9:  s8_din = o_data_10;
            default: s8_din = 1'b0;
        endcase
    end

    // Weight for FC Layer BRAM Signals
    wire s7_en_1,s7_en_2,s7_en_3,s7_en_4,s7_en_5,s7_en_6,s7_en_7,s7_en_8,s7_en_9,s7_en_10;
    wire s7_we_1,s7_we_2,s7_we_3,s7_we_4,s7_we_5,s7_we_6,s7_we_7,s7_we_8,s7_we_9,s7_we_10;
    wire [1:0] s7_web_1,s7_web_2,s7_web_3,s7_web_4,s7_web_5,s7_web_6,s7_web_7,s7_web_8,s7_web_9,s7_web_10;
    wire [14:0] s7_addr_1,s7_addr_2,s7_addr_3,s7_addr_4,s7_addr_5,s7_addr_6,s7_addr_7,s7_addr_8,s7_addr_9,s7_addr_10;
    wire [7:0] s7_din_1,s7_din_2,s7_din_3,s7_din_4,s7_din_5,s7_din_6,s7_din_7,s7_din_8,s7_din_9,s7_din_10;
    wire [7:0] s7_dout_1,s7_dout_2,s7_dout_3,s7_dout_4,s7_dout_5,s7_dout_6,s7_dout_7,s7_dout_8,s7_dout_9,s7_dout_10;

    assign {s7_en_1,s7_en_2,s7_en_3,s7_en_4,s7_en_5,s7_en_6,s7_en_7,s7_en_8,s7_en_9,s7_en_10}={10{(state==CONV2)}};
    assign {s7_we_1,s7_we_2,s7_we_3,s7_we_4,s7_we_5,s7_we_6,s7_we_7,s7_we_8,s7_we_9,s7_we_10}=10'b00_0000_0000;

    assign s7_web_1 = {2{s7_we_1}};
    assign s7_web_2 = {2{s7_we_2}};
    assign s7_web_3 = {2{s7_we_3}};
    assign s7_web_4 = {2{s7_we_4}};
    assign s7_web_5 = {2{s7_we_5}};
    assign s7_web_6 = {2{s7_we_6}};
    assign s7_web_7 = {2{s7_we_7}};
    assign s7_web_8 = {2{s7_we_8}};
    assign s7_web_9 = {2{s7_we_9}};
    assign s7_web_10 = {2{s7_we_10}};

    assign s7_addr_1=(counter_x <= 12'd2303) ? counter_x: 12'd2303;
    assign s7_addr_2=(counter_x <= 12'd2303) ? counter_x: 12'd2303;
    assign s7_addr_3=(counter_x <= 12'd2303) ? counter_x: 12'd2303;
    assign s7_addr_4=(counter_x <= 12'd2303) ? counter_x: 12'd2303;
    assign s7_addr_5=(counter_x <= 12'd2303) ? counter_x: 12'd2303;
    assign s7_addr_6=(counter_x <= 12'd2303) ? counter_x: 12'd2303;
    assign s7_addr_7=(counter_x <= 12'd2303) ? counter_x: 12'd2303;
    assign s7_addr_8=(counter_x <= 12'd2303) ? counter_x: 12'd2303;
    assign s7_addr_9=(counter_x <= 12'd2303) ? counter_x: 12'd2303;
    assign s7_addr_10=(counter_x <= 12'd2303) ? counter_x: 12'd2303;

    wire o_v_1,o_v_2,o_v_3,o_v_4,o_v_5,o_v_6,o_v_7,o_v_8,o_v_9,o_v_10;
    //assign {i_v_1,i_v_2,i_v_3,i_v_4,i_v_5,i_v_6,i_v_7,i_v_8,i_v_9,i_v_10}={10{(state==CONV2)}};

    // FC Layer Core Instantiation
    fc_layer fc_1(
        .clk(clk), .resetn(resetn), .i_data(o_data_0), .i_weight(s7_dout_1), .i_valid(o_valid_1), .o_data(o_data_1), .o_data_real(o_v_1)
    );

    fc_layer fc_2(
        .clk(clk), .resetn(resetn), .i_data(o_data_0), .i_weight(s7_dout_2), .i_valid(o_valid_1), .o_data(o_data_2), .o_data_real(o_v_2)
    );

    fc_layer fc_3 (
        .clk(clk), .resetn(resetn), .i_data(o_data_0), .i_weight(s7_dout_3), .i_valid(o_valid_1), .o_data(o_data_3), .o_data_real(o_v_3)
    );

    fc_layer fc_4 (
        .clk(clk), .resetn(resetn), .i_data(o_data_0), .i_weight(s7_dout_4), .i_valid(o_valid_1), .o_data(o_data_4), .o_data_real(o_v_4)
    );

    fc_layer fc_5 (
        .clk(clk), .resetn(resetn), .i_data(o_data_0), .i_weight(s7_dout_5), .i_valid(o_valid_1), .o_data(o_data_5), .o_data_real(o_v_5)
    );

    fc_layer fc_6 (
        .clk(clk), .resetn(resetn), .i_data(o_data_0), .i_weight(s7_dout_6), .i_valid(o_valid_1), .o_data(o_data_6), .o_data_real(o_v_6)
    );

    fc_layer fc_7 (
        .clk(clk), .resetn(resetn), .i_data(o_data_0), .i_weight(s7_dout_7), .i_valid(o_valid_1), .o_data(o_data_7), .o_data_real(o_v_7)
    );

    fc_layer fc_8 (
        .clk(clk), .resetn(resetn), .i_data(o_data_0), .i_weight(s7_dout_8), .i_valid(o_valid_1), .o_data(o_data_8), .o_data_real(o_v_8)
    );

    fc_layer fc_9 (
        .clk(clk), .resetn(resetn), .i_data(o_data_0), .i_weight(s7_dout_9), .i_valid(o_valid_1), .o_data(o_data_9), .o_data_real(o_v_9)
    );

    fc_layer fc_10 (
        .clk(clk), .resetn(resetn), .i_data(o_data_0), .i_weight(s7_dout_10), .i_valid(o_valid_1), .o_data(o_data_10), .o_data_real(o_v_10)
    );



    // Weight for FC Layer BRAM Instantiation
    BRAM7_1 FC_layer_weight_1 (
        .clka(s7_clka_1), .ena(s7_ena_1), .wea(s7_wea_1), .addra(s7_addra_1), .dina(s7_dina_1), .douta(s7_douta_1), // Port A
        .clkb(clk), .enb(s7_en_1), .web(s7_web_1), .addrb(s7_addr_1), .dinb(s7_din_1), .doutb(s7_dout_1) // Port B
    );

    BRAM7_2 FC_layer_weight_2 (
        .clka(s7_clka_2), .ena(s7_ena_2), .wea(s7_wea_2), .addra(s7_addra_2), .dina(s7_dina_2), .douta(s7_douta_2), // Port A
        .clkb(clk), .enb(s7_en_2), .web(s7_web_2), .addrb(s7_addr_2), .dinb(s7_din_2), .doutb(s7_dout_2) // Port B
    );

    BRAM7_3 FC_layer_weight_3 (
        .clka(s7_clka_3), .ena(s7_ena_3), .wea(s7_wea_3), .addra(s7_addra_3), .dina(s7_dina_3), .douta(s7_douta_3), // Port A
        .clkb(clk), .enb(s7_en_3), .web(s7_web_3), .addrb(s7_addr_3), .dinb(s7_din_3), .doutb  (s7_dout_3) // Port B
    );

    BRAM7_4 FC_layer_weight_4 (
        .clka(s7_clka_4), .ena(s7_ena_4), .wea(s7_wea_4), .addra(s7_addra_4), .dina(s7_dina_4), .douta(s7_douta_4), // Port A
        .clkb(clk), .enb(s7_en_4), .web(s7_web_4), .addrb(s7_addr_4), .dinb(s7_din_4), .doutb(s7_dout_4) // Port B
    );

    BRAM7_5 FC_layer_weight_5 (
        .clka(s7_clka_5), .ena(s7_ena_5), .wea(s7_wea_5), .addra(s7_addra_5), .dina(s7_dina_5), .douta(s7_douta_5), // Port A
        .clkb(clk), .enb(s7_en_5), .web(s7_web_5), .addrb(s7_addr_5), .dinb(s7_din_5), .doutb(s7_dout_5) // Port B
    );

    BRAM7_6 FC_layer_weight_6 (
        .clka(s7_clka_6), .ena(s7_ena_6), .wea(s7_wea_6), .addra(s7_addra_6), .dina(s7_dina_6), .douta(s7_douta_6), // Port A
        .clkb(clk), .enb(s7_en_6), .web(s7_web_6), .addrb(s7_addr_6), .dinb(s7_din_6), .doutb(s7_dout_6) // Port B
    );

    BRAM7_7 FC_layer_weight_7 (
        .clka(s7_clka_7), .ena(s7_ena_7), .wea(s7_wea_7), .addra(s7_addra_7), .dina(s7_dina_7), .douta(s7_douta_7), // Port A
        .clkb(clk), .enb(s7_en_7), .web(s7_web_7), .addrb(s7_addr_7), .dinb(s7_din_7), .doutb(s7_dout_7) // Port B
    );

    BRAM7_8 FC_layer_weight_8 (
        .clka(s7_clka_8), .ena(s7_ena_8), .wea(s7_wea_8), .addra(s7_addra_8), .dina(s7_dina_8), .douta(s7_douta_8), // Port A
        .clkb(clk), .enb(s7_en_8), .web(s7_web_8), .addrb(s7_addr_8), .dinb(s7_din_8), .doutb(s7_dout_8) // Port B
    );

    BRAM7_9 FC_layer_weight_9 (
        .clka(s7_clka_9), .ena(s7_ena_9), .wea(s7_wea_9), .addra(s7_addra_9), .dina(s7_dina_9), .douta(s7_douta_9), // Port A
        .clkb(clk), .enb(s7_en_9), .web(s7_web_9), .addrb(s7_addr_9), .dinb(s7_din_9), .doutb(s7_dout_9) // Port B
    );

    BRAM7_10 FC_layer_weight_10 (
        .clka(s7_clka_10), .ena(s7_ena_10), .wea(s7_wea_10), .addra(s7_addra_10), .dina(s7_dina_10), .douta(s7_douta_10), // Port A
        .clkb(clk), .enb(s7_en_10), .web(s7_web_10), .addrb(s7_addr_10), .dinb(s7_din_10), .doutb(s7_dout_10) // Port B
    );

    // BRAM for Storing Final Result
    BRAM8 FC_layer_result(
        .clka(s8_clka), .ena(s8_ena), .wea(s8_wea), .addra(s8_addra), .dina(s8_dina), .douta(s8_douta), // Port A
        .clkb(clk), .enb(s8_en), .web(s8_web), .addrb(s8_addr), .dinb(s8_din), .doutb(s8_dout) // Port B
    );
endmodule