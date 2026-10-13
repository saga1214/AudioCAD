# From Awareness to Adherence: Bridging the Context Gap in Spoken Dialogue Systems via Context-Aware Decoding

Official implementation of the Interspeech 2026 paper *From Awareness to
Adherence: Bridging the Context Gap in Spoken Dialogue Systems via
Context-Aware Decoding*.

We propose an audio-adapted Context-Aware Decoding (CAD) procedure that
amplifies contextual influence by contrasting output distributions with
and without a small key context (Sec. 2 of the paper). The key context
is identified at inference time from the model's own attention weights
(Sec. 2.3). The approach is training-free and works as a drop-in
modification to the generation loop of an existing spoken dialogue
system. We instantiate it on three state-of-the-art models:
**Kimi-Audio-7B-Instruct**, **Mimo-Audio-7B-Instruct** (with Thinking),
and **Qwen3-Omni-30B-A3B-Instruct**.

## Repository layout

```
AudioCAD/
├── README.md                       # this file
├── eval/
│   ├── evaluate_openai.py          # shared LLM-as-a-Judge (gpt-5-nano)
│   ├── compare_eval_stats.py       # baseline vs CAD delta report
│   └── audio-cue-im.txt            # 42 audio-cue IDs excluded from Semantic Memory
├── kimi/                           # Kimi-Audio (Sec. 3, Table 1)
│   ├── README.md
│   ├── inference.py
│   ├── Kimi-Audio/                 # CAD-patched upstream tree
│   ├── bash_scripts/               # run_table1_baseline.sh, run_table1_cad.sh
│   └── results/                    # bundled outputs + per-iter stats
├── mimo/                           # Mimo-Audio (Sec. 3, Table 1 + Fig. 2 + Fig. 3)
│   ├── README.md
│   ├── inference.py
│   ├── MiMo-Audio/
│   ├── bash_scripts/               # + run_ablation_fig2.sh, run_ablation_fig3.sh
│   └── results/
└── qwen/                           # Qwen3-Omni (Sec. 3, Table 1)
    ├── README.md
    ├── inference.py
    ├── Qwen3-Omni/
    ├── bash_scripts/
    └── results/
```

The CAD logic for each model is consistent across implementations:
attention probe -> layer selection -> token-to-turn aggregation
(`intra`) -> turn-to-round aggregation with cross-modal ratio `inter`
(beta) -> top-K rounds as key context (`num_rounds`) -> contrastive
decoding with penalty weight `guidance_scale` (alpha) following Eq. 3.

## Per-model environments

Each model uses a different upstream stack, so we keep their conda
environments separate. Full instructions live in
[`kimi/README.md`](kimi/README.md), [`mimo/README.md`](mimo/README.md),
[`qwen/README.md`](qwen/README.md). A short summary:

| Model | Python | transformers | Extras |
|---|---|---|---|
| Kimi-Audio | 3.10 | 4.51.3 | cuda-nvcc 12.8, ffmpeg, vendored requirements.txt |
| MiMo-Audio | 3.12 | 4.49.0 | accelerate, flash-attn, torch 2.9.1 + torchaudio |
| Qwen3-Omni | 3.12 | 4.57.3 | qwen-omni-utils, flash-attn, torch 2.9.1 |

## Shared setup

```bash
# Audio MultiChallenge benchmark
export AUDIOMC_CACHE_DIR=/path/to/audiomc/cache
huggingface-cli download ScaleAI/audiomc --repo-type dataset \
  --cache-dir "${AUDIOMC_CACHE_DIR}"

# OpenAI key for the LLM-as-a-Judge evaluation step
export OPENAI_API_KEY=sk-...
```

The inference scripts honour `--dataset_cache_dir` /
`--audio_cache_dir` CLI flags as well as the `AUDIOMC_CACHE_DIR` /
`AUDIOMC_WAV_DIR` env-vars; model weights are pulled lazily on first
run from HuggingFace unless `--model_path` or the per-model env-var
(e.g. `KIMI_AUDIO_MODEL_PATH`) points at a local snapshot.

## Reproducing Table 1

For every model, two scripts produce the baseline and CAD rows. Each
issues five independent generation runs over the
90 Semantic-Memory + 83 Self-Coherence samples (173 total) of the
benchmark, matching the paper's 5-iteration averaging.

```bash
# Kimi (single GPU)
( cd kimi && bash bash_scripts/run_table1_baseline.sh 0 \
            && bash bash_scripts/run_table1_cad.sh 0 )

# MiMo (single GPU)
( cd mimo && bash bash_scripts/run_table1_baseline.sh 0 \
            && bash bash_scripts/run_table1_cad.sh 0 )

# Qwen (two GPUs)
( cd qwen && bash bash_scripts/run_table1_baseline.sh "0,1" \
            && bash bash_scripts/run_table1_cad.sh "0,1" )
```

Then evaluate with `gpt-5-nano-2025-08-07`:

```bash
# example -- adapt to each model's result paths
python eval/evaluate_openai.py \
  --result_dirs kimi/results/baseline/KimiAudio_gs1.0_iter* \
                kimi/results/cad/KimiAudio_gs2.5_mean_inter0.5_last_4_r1_iter*
```

Pre-generated outputs and per-iteration `*_stats.json` for all rows are
shipped under each `results/` directory, so the evaluation step can be
skipped if you only want to read the bundled scores.

> **Note on reproducibility.** Some sources of non-determinism — BF16
> numerical noise, non-deterministic CUDA kernels, and MiMo's thinking
> mode in particular — mean that re-running inference may not yield
> outputs byte-identical to the bundled ones. To control for this, every
> number reported in the paper (and shown below) is the average of **5
> independent generation runs**, so the per-cell expectations remain
> stable in distribution even when individual generations diverge.

Expected 5-iteration averages on `gpt-5-nano-2025-08-07`:

| Model | Semantic Memory | Self Coherence | Average |
|---|---:|---:|---:|
| Mimo-Audio-7B-Instruct        | 26.00 | 26.02 | 26.01 |
| Mimo-Audio + Ours (CAD)       | **32.00** | **36.39** | **34.11** |
| Qwen3-Omni-30B-A3B-Instruct   | 22.67 | 29.16 | 25.78 |
| Qwen3-Omni + Ours (CAD)       | **39.33** | **38.80** | **39.08** |
| Kimi-Audio-7B-Instruct        | 13.56 | 19.04 | 16.19 |
| Kimi-Audio + Ours (CAD)       | **23.11** | **22.65** | **22.89** |

## Reproducing Fig. 2 and Fig. 3 (MiMo-Audio)

Both ablations use MiMo-Audio as the base model (Sec. 3.3).

```bash
# Fig. 2: ablation of layer / intra / beta at alpha=3.0, K=1
( cd mimo && bash bash_scripts/run_ablation_fig2.sh 0 )

# Fig. 3: penalty weight (alpha in {1.5, 2.0, 2.5, 3.0}) x K in {1, 2}
( cd mimo && bash bash_scripts/run_ablation_fig3.sh 0 )

# Both sets are bundled under mimo/results/ablation_fig2 and ablation_fig3;
# evaluation is a no-op replay against the bundled jsonls.
python eval/evaluate_openai.py --result_dirs mimo/results/ablation_fig2/*
python eval/evaluate_openai.py --result_dirs mimo/results/ablation_fig3/*
```

See [`mimo/README.md`](mimo/README.md) for the full set of expected
numbers per plot point.

## Citation

```bibtex
@inproceedings{lee26m_interspeech,
  title     = {{From Awareness to Adherence: Bridging the Context Gap in Spoken Dialogue Systems via Context-Aware Decoding}},
  author    = {Che Hyun Lee and Heeseung Kim and Sungroh Yoon},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {3421--3426},
  doi       = {10.21437/Interspeech.2026-1589},
  issn      = {2958-1796},
}
```

## Acknowledgements

We build directly on the open-source releases of
[Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio),
[MiMo-Audio](https://github.com/XiaomiMiMo/MiMo-Audio), and
[Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni), and on the
[Audio MultiChallenge benchmark](https://huggingface.co/datasets/ScaleAI/audiomc).
