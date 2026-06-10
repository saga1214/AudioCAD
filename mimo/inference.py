"""
Run MiMo-Audio inference on the Audio MultiChallenge benchmark, optionally
with audio-adapted Context-Aware Decoding (CAD).

Set ``--guidance_scale 1.0`` to reproduce the standard-decoding baseline.
The CAD hyper-parameters used in the paper's Table 1 are:
``intra=mean inter=0.5 layer_from=last_4 num_rounds=1 guidance_scale=2.5``.

Thinking mode is enabled to match the paper setting (Mimo-Audio with
the "Thinking" option, see paper Sec. 3.1.2).
"""

import argparse
import os
import sys
from collections import Counter

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "MiMo-Audio")
if src_path not in sys.path:
    sys.path.append(src_path)

from src.mimo_audio.mimo_audio import MimoAudio  # noqa: E402


def parse_axes(raw_axes: str) -> list:
    return [axis.strip() for axis in raw_axes.split(",") if axis.strip()]


def load_audio_cue_ids(filepath: str) -> set:
    if not os.path.exists(filepath):
        print(f"[Warning] '{filepath}' not found. Audio-cue filtering will be skipped.")
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def build_dataset(args):
    dataset = load_dataset("ScaleAI/audiomc", split="test", cache_dir=args.dataset_cache_dir)
    print(f"Original size: {len(dataset)}")

    axes = set(parse_axes(args.axes))
    audio_cue_ids = load_audio_cue_ids(args.audio_cue_file)

    def keep_sample(sample):
        if sample["axis"] not in axes:
            return False
        if sample["axis"] != "INFERENCE_MEMORY":
            return True
        if args.inference_memory_subset == "all":
            return True
        if args.inference_memory_subset == "audio_cue":
            return sample["id"] in audio_cue_ids
        if args.inference_memory_subset == "semantic":
            return sample["id"] not in audio_cue_ids
        return False

    filtered = dataset.filter(keep_sample)
    axis_counter = Counter(filtered["axis"]) if len(filtered) > 0 else Counter()
    print(
        f"Filtered size: {len(filtered)} | axes={sorted(axes)} "
        f"| inference_memory_subset={args.inference_memory_subset}"
    )
    print(f"Axis counts: {dict(axis_counter)}")
    return filtered


def get_or_save_audio(sample_id, turn_idx, audio_data, cache_dir):
    if audio_data is None:
        return None
    filename = f"{sample_id}_turn_{turn_idx}.wav"
    file_path = os.path.join(cache_dir, filename)
    if not os.path.exists(file_path):
        sf.write(file_path, audio_data["array"], audio_data["sampling_rate"])
    return file_path


def build_conversation(sample, sample_id, audio_cache_dir):
    last_turn_idx = 0
    for turn_idx in range(1, 9):
        if sample.get(f"user_turn_{turn_idx}_audio") is not None:
            last_turn_idx = turn_idx
        else:
            break

    conversation = []
    for turn_idx in range(1, last_turn_idx + 1):
        audio_path = get_or_save_audio(
            sample_id,
            turn_idx,
            sample.get(f"user_turn_{turn_idx}_audio"),
            cache_dir=audio_cache_dir,
        )
        if audio_path:
            conversation.append({"role": "user", "content": audio_path})

        if turn_idx < last_turn_idx:
            assistant_text = sample.get(f"assistant_turn_{turn_idx}_transcript")
            if assistant_text:
                conversation.append({"role": "assistant", "content": assistant_text})

    return conversation, last_turn_idx


def main(args):
    os.makedirs(args.audio_cache_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    print("Loading MiMo-Audio model...")
    model = MimoAudio(args.model_path, args.tokenizer_path)
    dataset = build_dataset(args)

    for sample_idx, sample in enumerate(tqdm(dataset, desc="Processing"), start=1):
        if args.max_samples is not None and sample_idx > args.max_samples:
            break

        sample_id = sample["id"]
        txt_path = os.path.join(args.result_dir, f"output_text_{sample_id}.txt")
        if os.path.exists(txt_path):
            continue

        conversation, last_turn_idx = build_conversation(sample, sample_id, args.audio_cache_dir)
        generated_text = model.speech2text_dialogue_sft_multiturn(
            conversation,
            guidance_scale=args.guidance_scale,
            intra=args.intra,
            inter=args.inter,
            layer_from=args.layer_from,
            num_rounds=args.num_rounds,
            thinking=True,
        )

        print(f"Generated Response: {generated_text}")
        print(f"Rubric: {sample['rubric']}")
        print(f"Reference: {sample.get(f'assistant_turn_{last_turn_idx}_transcript')}")

        if generated_text is not None:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(generated_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="results/debug")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("MIMO_AUDIO_MODEL_PATH", "XiaomiMiMo/MiMo-Audio-7B-Instruct"),
        help="Local path or HuggingFace model id for MiMo-Audio-7B-Instruct.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=os.environ.get("MIMO_AUDIO_TOKENIZER_PATH", "XiaomiMiMo/MiMo-Audio-Tokenizer"),
        help="Local path or HuggingFace model id for the MiMo-Audio audio tokenizer.",
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
    parser.add_argument("--axes", type=str, default="INFERENCE_MEMORY,SELF_COHERENCE")
    parser.add_argument(
        "--inference_memory_subset",
        type=str,
        default="semantic",
        choices=["all", "audio_cue", "semantic"],
        help="Which INFERENCE_MEMORY subset to keep. Paper uses 'semantic'.",
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
