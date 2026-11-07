// Student architecture work belongs here as well as in the PE/buffer modules.
// Baseline: N tiles -> M tiles -> K panels; reload both operands for every tile.
module controller #(parameter integer ROWS=8, COLS=8, K_DEPTH=256, LANES=4)(
    input logic clk, rst, start,
    input logic [15:0] dim_m, dim_n, dim_k,
    output logic busy, done,
    output logic a_req, b_req,
    output logic [31:0] a_addr, b_addr,
    output logic [15:0] a_count, b_count,
    input logic a_ready, b_ready,
    output logic [15:0] a_bank, b_bank, a_pos, b_pos,
    output logic a_write, b_write, pe_clear, pe_enable,
    output logic [15:0] active_rows, active_cols, rd_pos,
    output logic c_valid,
    input logic c_ready,
    output logic [31:0] c_addr,
    output logic [15:0] out_row, out_col
);
    typedef enum logic [2:0] {IDLE, SETUP, LOAD, COMPUTE, DRAIN} state_t;
    state_t state;
    logic [15:0] m_size, n_size, k_size;
    logic [15:0] m_base, n_base, k_base, k_len;
    logic a_loaded, b_loaded;
    function automatic [15:0] minimum(input integer x, y);
        minimum = 16'((x<y) ? x : y);
    endfunction
    assign active_rows = minimum(ROWS, m_size-m_base);
    assign active_cols = minimum(COLS, n_size-n_base);
    assign k_len = minimum(K_DEPTH, k_size-k_base);
    assign busy = state != IDLE;
    assign a_req = state==LOAD && !a_loaded;
    assign b_req = state==LOAD && !b_loaded;
    // B is stored column-major by the host. Addresses are INT16 word offsets.
    assign a_addr = (32'(m_base)+32'(a_bank))*32'(k_size)+32'(k_base)+32'(a_pos);
    assign b_addr = (32'(n_base)+32'(b_bank))*32'(k_size)+32'(k_base)+32'(b_pos);
    assign a_count = minimum(LANES, k_len-a_pos);
    assign b_count = minimum(LANES, k_len-b_pos);
    assign a_write = a_req && a_ready;
    assign b_write = b_req && b_ready;
    assign pe_clear = state==SETUP && k_base==0;
    assign pe_enable = state==COMPUTE;
    assign c_valid = state==DRAIN;
    assign c_addr = (32'(m_base)+32'(out_row))*32'(n_size)+32'(n_base)+32'(out_col);
    always_ff @(posedge clk) begin
        if (rst) begin
            state<=IDLE; done<=0;
            m_size<=0; n_size<=0; k_size<=0;
            m_base<=0; n_base<=0; k_base<=0;
            a_bank<=0; b_bank<=0; a_pos<=0; b_pos<=0;
            a_loaded<=0; b_loaded<=0; rd_pos<=0; out_row<=0; out_col<=0;
        end else begin
            done<=0;
            case (state)
                IDLE: if (start && dim_m!=0 && dim_n!=0 && dim_k!=0) begin
                    m_size<=dim_m; n_size<=dim_n; k_size<=dim_k;
                    m_base<=0; n_base<=0; k_base<=0; state<=SETUP;
                end
                SETUP: begin
                    a_bank<=0; b_bank<=0; a_pos<=0; b_pos<=0;
                    a_loaded<=0; b_loaded<=0; rd_pos<=0;
                    state<=LOAD;
                end
                LOAD: begin
                    if (a_write) begin
                        if (a_pos+a_count==k_len) begin
                            a_pos<=0;
                            if (a_bank+1==active_rows) a_loaded<=1;
                            else a_bank<=a_bank+1;
                        end else a_pos<=a_pos+a_count;
                    end
                    if (b_write) begin
                        if (b_pos+b_count==k_len) begin
                            b_pos<=0;
                            if (b_bank+1==active_cols) b_loaded<=1;
                            else b_bank<=b_bank+1;
                        end else b_pos<=b_pos+b_count;
                    end
                    if (a_loaded && b_loaded) state<=COMPUTE;
                end
                COMPUTE: begin
                    if (rd_pos+1==k_len) begin
                        rd_pos<=0;
                        if (k_base+k_len==k_size) begin
                            out_row<=0; out_col<=0; state<=DRAIN;
                        end else begin
                            k_base<=k_base+k_len; state<=SETUP;
                        end
                    end else rd_pos<=rd_pos+1;
                end
                DRAIN: if (c_ready) begin
                    if (out_col+1==active_cols) begin
                        out_col<=0;
                        if (out_row+1==active_rows) begin
                            out_row<=0; k_base<=0;
                            if (m_base+active_rows==m_size) begin
                                m_base<=0;
                                if (n_base+active_cols==n_size) begin
                                    state<=IDLE; done<=1;
                                end else begin
                                    n_base<=n_base+active_cols; state<=SETUP;
                                end
                            end else begin
                                m_base<=m_base+active_rows; state<=SETUP;
                            end
                        end else out_row<=out_row+1;
                    end else out_col<=out_col+1;
                end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
