// One bank per PE row/column; LANES contiguous words may be written per beat.
// Read port broadcasts one word per bank per MAC cycle. No reset of SRAM data.
module tile_buffer #(parameter integer BANKS=8, DEPTH=256, LANES=4)(
    input logic clk, wr_valid,
    input logic [15:0] wr_bank, wr_pos, wr_count, rd_pos,
    input logic [16*LANES-1:0] wr_data,
    output wire [16*BANKS-1:0] rd_data
);
    for (genvar bank=0; bank<BANKS; bank++) begin: banks
        logic signed [15:0] words [DEPTH];
        always_ff @(posedge clk) begin
            for (integer lane=0; lane<LANES; lane++) begin
                if (wr_valid && wr_bank==bank && lane<wr_count && wr_pos+lane<DEPTH)
                    words[wr_pos+lane] <= wr_data[16*lane +: 16];
            end
        end
        assign rd_data[16*bank +: 16] = words[rd_pos];
    end
endmodule
