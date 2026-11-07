module accelerator_top #(
    parameter integer ROWS=8, COLS=8, A_WORDS=2048, B_WORDS=2048, LANES=4
)(
    input logic clk, rst, start,
    input logic [15:0] dim_m, dim_n, dim_k,
    output wire busy, done,
    output wire a_req, b_req,
    output wire [31:0] a_addr, b_addr,
    output wire [15:0] a_count, b_count,
    input logic a_ready, b_ready,
    input logic [16*LANES-1:0] a_data, b_data,
    output wire c_valid,
    input logic c_ready,
    output wire [31:0] c_addr,
    output wire signed [63:0] c_data,
    // Observability ports: checked against actual numerical work by the harness.
    output wire pe_enable,
    output wire [15:0] active_rows, active_cols
);
    localparam integer A_DEPTH=A_WORDS/ROWS, B_DEPTH=B_WORDS/COLS;
    localparam integer K_DEPTH=(A_DEPTH<B_DEPTH) ? A_DEPTH : B_DEPTH;
    wire [15:0] a_bank, b_bank, a_pos, b_pos, rd_pos, out_row, out_col;
    wire a_write, b_write, pe_clear;
    wire [16*ROWS-1:0] a_values;
    wire [16*COLS-1:0] b_values;
    wire [64*ROWS*COLS-1:0] accumulators;
    controller #(.ROWS(ROWS), .COLS(COLS), .K_DEPTH(K_DEPTH), .LANES(LANES)) control(
        .clk(clk), .rst(rst), .start(start), .dim_m(dim_m), .dim_n(dim_n), .dim_k(dim_k),
        .busy(busy), .done(done), .a_req(a_req), .b_req(b_req), .a_addr(a_addr), .b_addr(b_addr),
        .a_count(a_count), .b_count(b_count), .a_ready(a_ready), .b_ready(b_ready),
        .a_bank(a_bank), .b_bank(b_bank), .a_pos(a_pos), .b_pos(b_pos),
        .a_write(a_write), .b_write(b_write), .pe_clear(pe_clear), .pe_enable(pe_enable),
        .active_rows(active_rows), .active_cols(active_cols), .rd_pos(rd_pos),
        .c_valid(c_valid), .c_ready(c_ready), .c_addr(c_addr), .out_row(out_row), .out_col(out_col));
    tile_buffer #(.BANKS(ROWS), .DEPTH(A_DEPTH), .LANES(LANES)) a_buffer(
        .clk(clk), .wr_valid(a_write), .wr_bank(a_bank), .wr_pos(a_pos), .wr_count(a_count),
        .wr_data(a_data), .rd_pos(rd_pos), .rd_data(a_values));
    tile_buffer #(.BANKS(COLS), .DEPTH(B_DEPTH), .LANES(LANES)) b_buffer(
        .clk(clk), .wr_valid(b_write), .wr_bank(b_bank), .wr_pos(b_pos), .wr_count(b_count),
        .wr_data(b_data), .rd_pos(rd_pos), .rd_data(b_values));
    pe_array #(.ROWS(ROWS), .COLS(COLS)) array(
        .clk(clk), .rst(rst), .clear(pe_clear), .enable(pe_enable),
        .active_rows(active_rows), .active_cols(active_cols), .a(a_values), .b(b_values), .acc(accumulators));
    assign c_data = accumulators[64*(32'(out_row)*COLS+32'(out_col)) +: 64];
endmodule
