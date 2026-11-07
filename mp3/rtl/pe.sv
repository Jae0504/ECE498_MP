// Signed INT16 MAC, exact for K <= 65535 with a signed 64-bit accumulator.
module pe(input logic clk, rst, clear, enable,
          input logic signed [15:0] a, b,
          output logic signed [63:0] acc);
    logic signed [31:0] product;
    assign product = a * b;
    always_ff @(posedge clk) begin
        if (rst || clear) acc <= '0;
        else if (enable) acc <= acc + {{32{product[31]}}, product};
    end
endmodule
