# MP3/MP4 dependencies and distribution

The parent repository stores Git submodule entries, not copied vendor source
or compiled binaries. A normal clone is sufficient; `sh mp3/setup_rtl.sh`
initializes the required submodules at the commits recorded by the parent.
It verifies the Verilator/cocotb source revisions before installation and leaves
tracked upstream source unchanged.

| Root submodule | Purpose | Pinned commit |
| --- | --- | --- |
| `miniLLAMA/` | Model; a patched runtime copy is prepared under `mp3/.deps/` | `542b3ae89bca7de773e5b048768e0d06f1598e80` |
| `cocotb/` | Python RTL testbench and model bridge | `ae55ced614b3f4ba89d77da5bd635024546d0737` |
| `verilator/` | Compile and simulate student RTL | `770b7d009e59071311d4173f8c86b4ac6085f1e9` |

MP1's `dynamorio/` dependency is separate and is initialized by MP1 setup.
See the root `.gitmodules` for upstream URLs and each vendor repository for its
license. Synthesis uses the YoWASP Yosys Python package installed by setup.

## Required setup

Tested on Linux aarch64 with Python 3.12.3, PyTorch 2.6.0 CPU, NumPy 1.26.4,
and the pinned cocotb/Verilator sources above. The bootstrap accepts Python
3.9–3.12; use `MP3_PYTHON=python3.12` to select the tested interpreter.
Provide Git, a C++ compiler, make, autoconf, flex, bison, Python development
headers, and `FlexLexer.h`. Debian/Ubuntu missing headers can be downloaded
and extracted locally by setup. Setup does not install system packages.

```sh
sh mp3/setup_rtl.sh
# Verify a prepared environment without reinstalling:
sh mp3/setup_rtl.sh --skip-install
mp3/.venv/bin/python -m pytest -q mp3/tests
```

Initial setup needs access to the submodule repositories and Python package
indexes. Prepared experiments run offline. Setup installs the CPU PyTorch wheel,
`requirements_rtl.txt`, cocotb from its pinned checkout, and
`yowasp-yosys==0.70.0.0.post1259`. The native `yosys` on PATH takes precedence
during synthesis; the report records the tool actually used.

`patches/miniLLAMA.patch` adds causal cached execution, per-layer KV caches,
and attention observation/execution hooks. `vendor_versions.json` pins the
original source and source-preparation inputs. The original `miniLLAMA/`
checkout is never patched in place. Source preparation records the revision,
patch hash, and prepared-file hashes in `mp3/.deps/manifest.json` and checks
them on subsequent setup runs.

## Local artifacts

| Location | Contents | Distributed through Git? |
| --- | --- | --- |
| `mp3/.venv/` | Python environment and installed tools | No |
| `mp3/.deps/miniLLAMA/` | Prepared model source and its manifest | No; recreated by setup |
| `mp3/.deps/rtl_build/` | Compiled Verilator models, keyed by source/parameters/tools | No |
| `mp3/results/`, `mp4/results/` | Run reports, logs, waveforms, netlists | No |
| `mp3/sample_answer/`, `mp4/sample_answer/` | Private instructor solutions and demos | No; also export-ignored |
| `mp4/rtl_student/*.sv` | Unchanged editable RTL starter | Yes |

Keep source, configuration, patches, setup/validation code, documentation,
`.gitmodules`, and submodule Git entries in the student repository. Raw folder
copies or ZIP tools must explicitly exclude the private directories and local
artifacts; Git ignore rules do not control those tools.

See [PPA](rtl_ppa.md) for the structural comparison and technology-analysis limits.
