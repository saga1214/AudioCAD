# Qwen3-Omni: CAD reproduction

This subdirectory reproduces the **Qwen3-Omni-30B-A3B-Instruct** row of
Table 1 in *From Awareness to Adherence: Bridging the Context Gap in
Spoken Dialogue Systems via Context-Aware Decoding*.

## 1. Environment

```bash
conda create -n qwen python=3.12 -y
conda activate qwen
pip install uv

uv pip install "transformers==4.57.3" "datasets==3.6.0"
uv pip install accelerate soundfile
uv pip install -U qwen-omni-utils
uv pip install -U flash-attn --no-build-isolation
uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 torchcodec==0.9.1
```

The `Qwen3-Omni/` subtree contains the two patched files:
- `src/modeling_qwen3_omni_moe.py` — generate() wrapper that routes CAD
  arguments (`guidance_scale`, `intra`, `inter`, `layer_from`,
  `num_rounds`, `turn_indices`) to the thinker.
- `src/utils.py` — `_generate_helper` extended with the attention probe,
  per-token/turn/round aggregation, top-K masking, and contrastive
  logit combination (Eq. 3 in the paper).

## 2. Data & weights

```bash
export AUDIOMC_CACHE_DIR=/path/to/audiomc/cache
huggingface-cli download ScaleAI/audiomc --repo-type dataset --cache-dir "${AUDIOMC_CACHE_DIR}"

# Model weights ~60 GB. Pulled lazily on first run from HuggingFace
# (override with --model_path or QWEN3_OMNI_MODEL_PATH for a local snapshot).
```

## 3. Reproduce Table 1

Qwen3-Omni-30B-A3B-Instruct needs ~80 GB of GPU memory; we ran on 2 GPUs.

```bash
# Baseline (standard decoding, alpha = 1.0). 5 independent runs.
bash bash_scripts/run_table1_baseline.sh "0,1"

# Ours (CAD), paper-optimal config: alpha=2.5, intra=mean, beta=0.5,
# layer_from=last_4, K=1. 5 independent runs.
bash bash_scripts/run_table1_cad.sh "0,1"
```

Each iteration writes 173 `output_text_<sample_id>.txt` files to
`results/baseline/Qwen3Omni_gs1.0_iter{1..5}/` and
`results/cad/Qwen3Omni_gs2.5_mean_inter0.5_last_4_r1_iter{1..5}/`.

## 4. Evaluate with gpt-5-nano

```bash
export OPENAI_API_KEY=sk-...

# Baseline
python ../eval/evaluate_openai.py \
  --result_dirs results/baseline/Qwen3Omni_gs1.0_iter*

# CAD
python ../eval/evaluate_openai.py \
  --result_dirs results/cad/Qwen3Omni_gs2.5_mean_inter0.5_last_4_r1_iter*

# Single baseline-vs-CAD delta
python ../eval/compare_eval_stats.py \
  --baseline_dir results/baseline/Qwen3Omni_gs1.0_iter1 \
  --cad_dir     results/cad/Qwen3Omni_gs2.5_mean_inter0.5_last_4_r1_iter1
```

## 5. Expected numbers (paper Table 1)

5-iteration averages against gpt-5-nano-2025-08-07:

| Setting | Semantic Memory | Self Coherence | Average |
|---|---:|---:|---:|
| Qwen3-Omni-30B-A3B-Instruct       | 22.67 | 29.16 | 25.78 |
| + Ours (CAD)                      | **39.33** | **38.80** | **39.08** |

Pre-generated outputs and per-iteration stats for both runs are bundled
under `results/` so you can run only the evaluation step to verify.
