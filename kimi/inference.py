"""
Run Kimi-Audio inference on the Audio MultiChallenge benchmark, optionally
with audio-adapted Context-Aware Decoding (CAD).

Set ``--guidance_scale 1.0`` to reproduce the standard-decoding baseline.
The CAD hyper-parameters used in the paper's Table 1 are:
``intra=mean inter=0.5 layer_from=last_4 num_rounds=1 guidance_scale=2.5``.
"""

import argparse
import os
import sys

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "Kimi-Audio")
if src_path not in sys.path:
    sys.path.append(src_path)

from kimia_infer.api.kimia import KimiAudio  # noqa: E402


def get_or_save_audio(sample_id, turn_idx, audio_data, cache_dir):
    if audio_data is None:
        return None
    filename = f"{sample_id}_turn_{turn_idx}.wav"
    file_path = os.path.join(cache_dir, filename)
    if not os.path.exists(file_path):
        sf.write(file_path, audio_data["array"], audio_data["sampling_rate"])
    return file_path


def load_exclude_ids(filepath: str) -> set:
    if not os.path.exists(filepath):
        print(f"[Warning] '{filepath}' not found. No IDs will be excluded.")
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        ids = {line.strip() for line in f if line.strip()}
    print(f"Loaded {len(ids)} IDs to exclude from {filepath}")
    return ids


def build_conversation(sample, sample_id, audio_cache_dir):
    last_turn_idx = 0
    for t in range(1, 9):
        if sample.get(f"user_turn_{t}_audio") is not None:
            last_turn_idx = t
        else:
            break

    conversation = []
    for t in range(1, last_turn_idx + 1):
        audio_path = get_or_save_audio(
            sample_id,
            t,
            sample.get(f"user_turn_{t}_audio"),
            cache_dir=audio_cache_dir,
        )
        if audio_path:
            conversation.append({"role": "user", "message_type": "audio", "content": audio_path})

        if t < last_turn_idx:
            a_text = sample.get(f"assistant_turn_{t}_transcript")
            if a_text:
                conversation.append({"role": "assistant", "message_type": "text", "content": a_text})
    return conversation


def main(args):
    os.makedirs(args.audio_cache_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    exclude_ids = load_exclude_ids(args.audio_cue_file)

    print("Loading Kimi-Audio model...")
    model = KimiAudio(model_path=args.model_path, load_detokenizer=True)

    dataset = load_dataset("ScaleAI/audiomc", split="test", cache_dir=args.dataset_cache_dir)
    dataset = dataset.filter(
        lambda x: x["axis"] in ["INFERENCE_MEMORY", "SELF_COHERENCE"] and x["id"] not in exclude_ids
    )
    print(f"Filtered task size (excluding audio-cue): {len(dataset)}")

    # Greedy decoding for reproducibility.
    sampling_params = {
        "text_temperature": 0.0,
        "text_top_k": 1,
        "text_repetition_penalty": 1.0,
    }

    for sample_idx, sample in enumerate(tqdm(dataset, desc="Processing"), start=1):
        if args.max_samples is not None and sample_idx > args.max_samples:
            break

        sample_id = sample["id"]

        txt_path = os.path.join(args.result_dir, f"output_text_{sample_id}.txt")
        if os.path.exists(txt_path):
            continue

        conversation = build_conversation(sample, sample_id, args.audio_cache_dir)

        try:
            _, generated_text = model.generate(
                conversation,
                output_type="text",
                **sampling_params,
                guidance_scale=args.guidance_scale,
                intra=args.intra,
                inter=args.inter,
                layer_from=args.layer_from,
                num_rounds=args.num_rounds,
            )
        except Exception as e:
            print(f"Generation error for {sample_id}: {e}")
            continue

        print(f"Generated Response: {generated_text}")
        print(f"Rubric: {sample['rubric']}")

        if generated_text is not None:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(generated_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="results/debug")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("KIMI_AUDIO_MODEL_PATH", "moonshotai/Kimi-Audio-7B-Instruct"),
        help="Local path or HuggingFace model id for Kimi-Audio-7B-Instruct.",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=os.environ.get("AUDIOMC_CACHE_DIR", "./data/audiomc"),
        help="Cache directory for the ScaleAI/audiomc dataset.",
    )
    parser.add_argument(
        "--audio_cache_dir",
        type=str,
        default=os.environ.get("AUDIOMC_WAV_DIR", "./data/audiomc/wav"),
        help="Where extracted .wav files for each user turn will be cached.",
    )
    parser.add_argument(
        "--audio_cue_file",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval", "audio-cue-im.txt"),
        help="List of INFERENCE_MEMORY sample IDs to drop (audio-cue subset).",
    )
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="CAD penalty weight alpha. 1.0 = standard decoding (baseline).")
    parser.add_argument("--intra", type=str, default="mean", choices=["sum", "mean", "max"],
                        help="Token-to-turn aggregation.")
    parser.add_argument("--inter", type=float, default=0.5,
                        help="Turn-to-round ratio beta = S(A) + beta * S(U).")
    parser.add_argument("--layer_from", type=str, default="last_4", choices=["last_1", "last_4", "all"],
                        help="Attention layers used for the latent-awareness probe.")
    parser.add_argument("--num_rounds", type=int, default=1, help="Top-K rounds kept as key context.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Stop after this many samples. Default = all 173.")
    args = parser.parse_args()

    main(args)
