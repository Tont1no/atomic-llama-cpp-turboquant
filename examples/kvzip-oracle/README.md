# Bonsai KVzap oracle pilot

This example follows the pinned `kvzap/data.py` training collector at NVIDIA/kvpress commit `7331c23da9e6f1510d89ea651d0dea77a57b3252`. It scores original user-context keys against **only** the assistant's copied-context queries in one full chat sequence. The native `kq_soft_max` callback supplies the model's full causal attention. There is no key subset or post-softmax renormalization. Each query's attention is divided by its post-`attn_norm` hidden-state norm. `score.py` multiplies by `norm(W_O V)` and takes the maximum over copied-context queries and the six query heads in each GQA group. Sink, recent, and protected-position rules affect only the final training eligibility mask.

`prepare_input.py` uses the verified local fast tokenizer's exact chat template and offset map. It renders `user: <context>\n\nRepeat the previous context exactly.` followed by `assistant: <context>`. It writes the complete token sequence plus explicit `[source_start, source_end)` and `[repeat_start, repeat_end)` boundaries to a new private directory. It refuses tokens that cross a text/template boundary. The repeated text can tokenize to a different number of tokens; there is no false one-to-one token-origin mapping.

Prompt, token, capture, and score files must stay under `repo/tmp/bonsai-training-20260924` or the exact marker-verified `Y:\Ai-Loader-training-20260924` root. The preparer accepts a UTF-8 text-only prompt file in the private area or one row from the private five-field candidate JSONL:

```text
python prepare_input.py --repo REPO --text PRIVATE_PROMPT --output NEW_PRIVATE_INPUT_DIR
python prepare_input.py --repo REPO --jsonl PRIVATE_CANDIDATES.jsonl --index 0 --output NEW_PRIVATE_INPUT_DIR
```

The collector takes `ids.txt`, `protected.txt` (absolute original-source token positions; empty for text-only prompts), and the four integer boundaries from `input-receipt.json`. It never prints token IDs or tensor values.

The command uses ordered flags:

```text
llama-kvzip-oracle --repo REPO --model GGUF --ids PRIVATE_IDS --protected PRIVATE_POSITIONS --output NEW_CAPTURE_NAME --source-start N --source-end N --repeat-start N --repeat-end N --gpu-layers N --sha256 TARGET_GGUF_SHA256 --binary-sha256 EXE_SHA256 --template-sha256 CHAT_TEMPLATE_SHA256
```

The root Ai-Loader build forces llama.cpp examples off. Configure this directory standalone with `-DAI_LOADER_BUILD_DIR=.../build-ada-blackwell-mixed`. The executable links the existing fork import libraries and needs the build's `Inference engine/Release` directory on `PATH` at runtime. The assigned operator owns GPU runs.

The collector writes FP32 post-`attn_norm` source features, FP32 source V, FP32 native-attention maxima, source/query positions, eligibility, eight W_O probe rows per layer, and a receipt. It uses F16 KV cache and disables FlashAttention so the real `kq_soft_max` callback is available. `query_origin.i32` is -1: copied text need not have one-to-one token alignment with the original. A complete `receipt.json` marks a successful capture; incomplete directories are invalid.

`score.py` converts this capture with the effective FP16 W_O matrices from `target_export_receipt.json`. It first compares `W_O @ attn_gated` against native `attn_output` at 8 source positions in every attention layer. Failed probes produce no final oracle output. A passing run emits per-layer NPZ files and `oracle_meta.json` with the trainer's KVzip+ contract:

```text
python score.py --repo REPO --capture PRIVATE_CAPTURE_DIR --export-receipt PRIVATE_EXPORT_RECEIPT --output NEW_PRIVATE_SCORE_DIR --device cuda
```

The scorer rounds Vcur to the F16 cache basis, computes `norm(W_O V)` for each query head, multiplies the native attention maxima, then takes the maximum over each six-head GQA group. The FP16 W_O approximation is accepted only if the native probe meets its relative-error and cosine limits. No training input is valid before the final `oracle_meta.json` exists.

For the 110-context stage, `prepare_batch.py` reads the private candidate/heldout JSONL once and writes exact eight-field `runs.json` manifests plus private token files under the marked `Y:\Ai-Loader-training-20260924\kvzap` root. New input receipts bind each row to the original JSONL path, its SHA-256, the original row index, and the token-file SHA-256. Existing pilot receipts remain untouched.

```text
python prepare_batch.py --repo REPO --jsonl PRIVATE_CANDIDATES.jsonl --name candidate64 --output-root Y:\Ai-Loader-training-20260924\kvzap
llama-kvzip-oracle --repo REPO --model GGUF --runs Y:\Ai-Loader-training-20260924\kvzap\candidate64-runs.json --gpu-layers 99 --sha256 TARGET_GGUF_SHA256 --binary-sha256 EXE_SHA256
python score_batch.py --repo REPO --runs Y:\Ai-Loader-training-20260924\kvzap\candidate64-runs.json --export-receipt PRIVATE_EXPORT_RECEIPT --device cuda
```

Use `--validate-only` on the batch collector command to check all manifest rows and actual token-file hashes without loading the model. The batch collector loads the verified GGUF once, then creates and frees a fresh `llama_context` for every prompt. It preserves the v2 oracle math and all source rows. The native v3 receipt records the SHA-256 of the bytes actually decoded. The scorer checks a native W_O probe for every context, while its process-local cache hashes and loads each effective W_O file once. Both batch commands resume only complete, matching outputs and refuse partial directories. Only the assigned GPU operator runs capture and CUDA scoring. The trainer's row limit and sampling policy need a separate decision before expanding beyond this stage.
