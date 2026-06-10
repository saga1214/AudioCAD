# MiMo-Audio: CAD reproduction

This subdirectory reproduces the **Mimo-Audio-7B-Instruct** row of Table 1
in *From Awareness to Adherence: Bridging the Context Gap in Spoken
Dialogue Systems via Context-Aware Decoding*, as well as the two
ablations (Fig. 2 and Fig. 3) which use MiMo-Audio as the base model.

## 1. Environment

```bash
conda create -n mimo python=3.12 -y
conda activate mimo
pip install uv

uv pip install "transformers==4.49.0" "datasets==3.6.0"
uv pip install accelerate soundfile librosa
uv pip install -U flash-attn --no-build-isolation
uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 torchcodec==0.9.1
```

The `MiMo-Audio/` subtree is the upstream MiMo-Audio repository with our
CAD patches; the main edits live in
`MiMo-Audio/src/mimo_audio/modeling_mimo_audio.py` (CAD generation, attention
probing, contrastive logits) and `MiMo-Audio/src/mimo_audio/mimo_audio.py`
(the `speech2text_dialogue_sft_multiturn` wrapper that exposes the CAD
arguments). Thinking mode (Sec. 3.1.2 of the paper) is enabled in
`inference.py`.

## 2. Data & weights

```bash
export AUDIOMC_CACHE_DIR=/path/to/audiomc/cache
huggingface-cli download ScaleAI/audiomc --repo-type dataset --cache-dir "${AUDIOMC_CACHE_DIR}"

# Model + audio tokenizer (pulled lazily on first run unless overridden via
# MIMO_AUDIO_MODEL_PATH / MIMO_AUDIO_TOKENIZER_PATH or --model_path / --tokenizer_path).
```

## 3. Reproduce Table 1

```bash
# Baseline (standard decoding, alpha = 1.0). 5 independent runs.
bash bash_scripts/run_table1_baseline.sh 0

# Ours (CAD), paper-optimal config:
# alpha=2.5, intra=mean, beta=0.5, layer_from=last_4, K=1. 5 runs.
bash bash_scripts/run_table1_cad.sh 0
```

## 4. Reproduce Fig. 2 (layer / intra / beta ablation)

```bash
# 7 unique configurations at alpha=3.0, K=1, 5 iterations each.
bash bash_scripts/run_ablation_fig2.sh 0
```

The No CAD reference (purple dashed line in Fig. 2, paper 26.01%) is the
same as the MiMo baseline from Sec. 3.

## 5. Reproduce Fig. 3 (penalty weight x context scope)

```bash
# alpha in {1.5, 2.0, 2.5, 3.0} x K in {1, 2} at (mean, 0.5, last_4).
# alpha=1.0 point in Fig. 3 reuses the run_table1_baseline.sh output.
bash bash_scripts/run_ablation_fig3.sh 0
```

## 6. Evaluate with gpt-5-nano

```bash
export OPENAI_API_KEY=sk-...

# Table 1 -- CAD (single eval over INFERENCE_MEMORY + SELF_COHERENCE)
python ../eval/evaluate_openai.py \
  --result_dirs results/cad/MiMo_gs2.5_mean_inter0.5_last_4_r1_iter*

# Table 1 -- Baseline (two separate evals because the existing outputs
# come from historic per-axis scripts: Semantic Memory and Self Coherence
# were generated and evaluated separately. Re-running these is a no-op
# because every dataset sample is already in the bundled jsonl, so no
# OpenAI calls are made; only the stats.json is rewritten in the new
# format.)
python ../eval/evaluate_openai.py \
  --axes INFERENCE_MEMORY --inference_memory_subset semantic \
  --result_dirs results/baseline_im/MiMo_gs1.0_iter*

python ../eval/evaluate_openai.py \
  --axes SELF_COHERENCE \
  --result_dirs results/baseline_sc/MiMo_gs1.0_iter*

# Ablations
python ../eval/evaluate_openai.py --result_dirs results/ablation_fig2/*
python ../eval/evaluate_openai.py --result_dirs results/ablation_fig3/*
```

> **Note on the baseline directories:** the bundled
> `results/baseline_im/MiMo_gs1.0_iter{1..5}/` and
> `results/baseline_sc/MiMo_gs1.0_iter{1..5}/` hold the historic per-axis
> generations: `baseline_im/` keeps the 90 Semantic-Memory samples (the
> 42 audio-cue ids are filtered out) and `baseline_sc/` keeps the 83
> Self-Coherence samples. Each directory ships with its own
> `*_evaluation_results.jsonl` and `*_stats.json` so the paper Table 1
> baseline scores can be read directly without any re-evaluation.

## 7. Expected numbers

**Table 1** (5-iteration averages, gpt-5-nano-2025-08-07):

| Setting | Semantic Memory | Self Coherence | Average |
|---|---:|---:|---:|
| Mimo-Audio-7B-Instruct           | 26.00 | 26.02 | 26.01 |
| + Ours (CAD)                     | **32.00** | **36.39** | **34.11** |

**Fig. 2** (alpha=3.0, K=1, 5-iteration averages from bundled stats):

| sub-plot | config | Average APR |
|---|---|---:|
| Layer Selection   | last_1 / **last_4** / all      | 29.76 / **33.10** / 30.85 |
| Token-to-Turn     | max / sum / **mean**           | 30.41 / 32.84 / **33.10** |
| Turn-to-Round (β) | 0.25 / **0.5** / 1.0           | 28.83 / **33.10** / 31.51 |

No CAD reference (purple dashed line in Fig. 2) = Table 1 baseline =
**26.01** (SemMem 26.00, SC 26.02).

**Fig. 3** (mean, 0.5, last_4, 5-iteration averages from bundled stats):

| alpha | K=1 (APR) | K=2 (APR) |
|---|---:|---:|
| 1.0  | 26.01            | n/a |
| 1.5  | 29.45            | 26.83 |
| 2.0  | 29.32            | 27.69 |
| 2.5  | **34.19 (best)** | 26.93 |
| 3.0  | 33.10            | 26.50 |

The optimum is alpha=2.5 at K=1; expanding to K=2 generally degrades
performance, sharply so at higher alpha values.

Pre-generated outputs and per-iteration stats for all five buckets
(`baseline_im/`, `baseline_sc/`, `cad/`, `ablation_fig2/`,
`ablation_fig3/`) are bundled under `results/` so you can run only the
evaluation step to verify, with the single exception described in
Section 6 above.
