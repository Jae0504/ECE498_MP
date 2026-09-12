# TinyLlama configuration provenance

The benchmark dimensions were checked on 2026-09-04 against two official
TinyLlama-controlled sources:

- Official source repository: <https://github.com/jzhang38/TinyLlama>
- Official model configuration:
  <https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/main/config.json>

The configuration was cross-checked against the source repository at Git commit
`bf122247c486b6b897050e98cbb7bedae8eeba73`. The model page displayed config
revision `26d45f2` when inspected. TinyLlama source is not vendored or included
as a submodule because no benchmark build or runtime path uses it.

Both sources give 22 layers, 32 query heads, 4 KV groups, hidden size 2048,
intermediate size 5632, and maximum sequence length 2048. The per-head width is
therefore derived as `2048 / 32 = 64`.

No model weights are downloaded or needed by this prototype.
