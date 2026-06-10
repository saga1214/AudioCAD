# Kimi-Audio: CAD reproduction

This subdirectory reproduces the **Kimi-Audio-7B-Instruct** row of Table 1
in *From Awareness to Adherence: Bridging the Context Gap in Spoken
Dialogue Systems via Context-Aware Decoding*.

## 1. Environment

```bash
conda create -n kimi python=3.10 -y
conda activate kimi
pip install uv

conda install -y nvidia/label/cuda-12.8.1::cuda-nvcc
conda install -y nvidia/label/cuda-12.8.1::cuda-toolkit
# install ffmpeg from conda-forge to avoid a channel conflict with the pinned cuda-toolkit
conda install -y -c conda-forge --override-channels ffmpeg

uv pip install -r Kimi-Audio/requirements.txt
uv pip install "transformers==4.51.3" "datasets==3.6.0"
```

The `Kimi-Audio/` subtree is the upstream Kimi-Audio repository with our
CAD patches applied; the main edits live in
`Kimi-Audio/kimia_infer/api/kimia.py` (CAD generation, attention probing,
contrastive logits) and `Kimi-Audio/kimia_infer/api/prompt_manager.py`
(turn-index plumbing).

## 2. Data & weights

```bash
# AudioMultiChallenge benchmark (90 Semantic-Memory + 83 Self-Coherence
# samples after audio-cue exclusion)
export AUDIOMC_CACHE_DIR=/path/to/audiomc/cache
huggingface-cli download ScaleAI/audiomc --repo-type dataset --cache-dir "${AUDIOMC_CACHE_DIR}"

# Model weights — pulled lazily on first run from HuggingFace
# (override with --model_path or KIMI_AUDIO_MODEL_PATH if you already have
#  a local snapshot of moonshotai/Kimi-Audio-7B-Instruct)
```

## 3. Reproduce Table 1

```bash
# Baseline (standard decoding, alpha = 1.0). 5 independent runs.
bash bash_scripts/run_table1_baseline.sh 0     # GPU 0

# Ours (CAD), paper-optimal config: alpha=2.5, intra=mean, beta=0.5,
# layer_from=last_4, K=1. 5 independent runs.
bash bash_scripts/run_table1_cad.sh 0
```

Each iteration writes 173 `output_text_<sample_id>.txt` files to
`results/baseline/KimiAudio_gs1.0_iter{1..5}/` and
`results/cad/KimiAudio_gs2.5_mean_inter0.5_last_4_r1_iter{1..5}/`.

## 4. Evaluate with gpt-5-nano

The same evaluator is used for all three models, so it lives at the repo
root in `../eval/evaluate_openai.py`.

```bash
export OPENAI_API_KEY=sk-...

# Baseline
python ../eval/evaluate_openai.py \
  --result_dirs results/baseline/KimiAudio_gs1.0_iter*

# CAD
python ../eval/evaluate_openai.py \
  --result_dirs results/cad/KimiAudio_gs2.5_mean_inter0.5_last_4_r1_iter*

# Quick baseline-vs-CAD delta (uses one iter from each)
python ../eval/compare_eval_stats.py \
  --baseline_dir results/baseline/KimiAudio_gs1.0_iter1 \
  --cad_dir     results/cad/KimiAudio_gs2.5_mean_inter0.5_last_4_r1_iter1
```

## 5. Expected numbers (paper Table 1)

5-iteration averages against gpt-5-nano-2025-08-07:

| Setting | Semantic Memory | Self Coherence | Average |
|---|---|---|---|
| Kimi-Audio-7B-Instruct           | 13.56 | 19.04 | 16.19 |
| + Ours (CAD)                     | **23.11** | **22.65** | **22.89** |

Pre-generated outputs and per-iteration stats for both runs are bundled
under `results/` so you can run only the evaluation step to verify.
