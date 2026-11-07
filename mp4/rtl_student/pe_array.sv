// Output-stationary broadcast array: each row receives A, each column receives B.
module pe_array #(parameter integer ROWS=8, COLS=8)(
    input logic clk, rst, clear, enable,
    input logic [15:0] active_rows, active_cols,
    input logic [16*ROWS-1:0] a,
    input logic [16*COLS-1:0] b,
    output wire [64*ROWS*COLS-1:0] acc
);
    for (genvar r=0; r<ROWS; r++) begin: row
        for (genvar c=0; c<COLS; c++) begin: col
            pe mac(.clk(clk), .rst(rst), .clear(clear),
                .enable(enable && r<active_rows && c<active_cols),
                .a(a[16*r +: 16]), .b(b[16*c +: 16]), .acc(acc[64*(r*COLS+c) +: 64]));
        end
    end
endmodule
