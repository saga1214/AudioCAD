"""
LLM-as-a-Judge evaluator for the Audio MultiChallenge benchmark.

For each conversation in a result directory, the judge model (default
gpt-5-nano-2025-08-07) scores the assistant's final response against each
rubric item on a binary scale. A generation passes only if every rubric item
passes.

The OpenAI API key is read from the ``OPENAI_API_KEY`` environment variable
or the ``--api_key`` CLI argument.
"""

import argparse
import ast
import json
import os
import re
import time
from typing import Any, Dict, List

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

EXCLUDED_SAMPLE_IDS = {"plrw2n71ditatq03"}


SYSTEM_PROMPT_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation History
{conversation_history}

# Rubric item
{rubric_item}

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
• The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
• The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item.

# Example 1
For example, if the response is "For tonight's dinner, think of making a cheese and mushroom pizza" and the rubric item is "Avoids dairy product in the recipes", you should return a json like this:

```json
{{
"explanation": "The response includes a dairy product in the recipe, so the criteria was not met.",
"criteria_met": false
}}
```

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


def parse_axes(raw_axes: str) -> List[str]:
    return [axis.strip() for axis in raw_axes.split(",") if axis.strip()]


def load_audio_cue_ids(filepath: str) -> set:
    if not os.path.exists(filepath):
        print(f"[Warning] '{filepath}' not found. Audio-cue filtering will be skipped.")
        return set()

    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def build_dataset(args, audio_cue_ids):
    dataset = load_dataset(
        "ScaleAI/audiomc",
        split="test",
        cache_dir=args.dataset_cache_dir,
    )

    axes = set(parse_axes(args.axes))

    def keep_sample(sample):
        if sample["id"] in EXCLUDED_SAMPLE_IDS:
            return False

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
    print(
        f"Loaded {len(filtered)} evaluation samples | "
        f"axes={parse_axes(args.axes)} | "
        f"inference_memory_subset={args.inference_memory_subset}"
    )
    if EXCLUDED_SAMPLE_IDS:
        print(f"Excluded sample IDs: {sorted(EXCLUDED_SAMPLE_IDS)}")
    return filtered


def build_grading_conversation_history(row: Dict[str, str], model_response: str) -> str:
    last_user_turn = 0
    for turn_num in range(1, 9):
        user_transcript_col = f"user_turn_{turn_num}_transcript"
        if row[user_transcript_col]:
            last_user_turn = turn_num

    history_parts = []

    for turn_num in range(1, last_user_turn + 1):
        user_transcript_col = f"user_turn_{turn_num}_transcript"
        assistant_transcript_col = f"assistant_turn_{turn_num}_transcript"

        if user_transcript_col in row and row[user_transcript_col].strip():
            history_parts.append(f"User: {row[user_transcript_col].strip()}")

        if turn_num < last_user_turn:
            if assistant_transcript_col in row and row[assistant_transcript_col].strip():
                history_parts.append(f"Assistant: {row[assistant_transcript_col].strip()}")
        else:
            history_parts.append(f"Assistant: {model_response}")

    return "\n\n".join(history_parts)


def parse_rubric(rubric_data: Any) -> List[str]:
    if isinstance(rubric_data, str):
        try:
            parsed = ast.literal_eval(rubric_data)
            if isinstance(parsed, list):
                return parsed
            return [rubric_data]
        except (ValueError, SyntaxError):
            return [rubric_data]

    return [str(rubric_data)]


def parse_json_response(content: str):
    try:
        match = re.search(r"```json\s*(.*?)```", content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = content.strip()

        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"JSON Parsing Error. Raw content: {content}")
        return {"criteria_met": False, "explanation": "JSON Parsing Failed"}


def calc_stats(results):
    total = len(results)
    passed = sum(1 for result in results if result.get("criteria_met", False))
    score = (passed / total * 100) if total > 0 else 0.0
    return {"total": total, "passed": passed, "score": score}


def build_inference_memory_breakdown(results, audio_cue_ids):
    inference_memory_results = [result for result in results if result.get("axis") == "INFERENCE_MEMORY"]
    audio_cue_results = [result for result in inference_memory_results if result.get("sample_id") in audio_cue_ids]
    semantic_results = [result for result in inference_memory_results if result.get("sample_id") not in audio_cue_ids]
    return {
        "audio_cue": calc_stats(audio_cue_results),
        "semantic": calc_stats(semantic_results),
    }


def evaluate_result_dir(result_dir: str, dataset, client: OpenAI, args, audio_cue_ids):
    eval_output_file = os.path.join(result_dir, f"{args.judge_model}_evaluation_results.jsonl")
    results = []

    processed_ids = set()
    if os.path.exists(eval_output_file):
        with open(eval_output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("sample_id") in EXCLUDED_SAMPLE_IDS:
                        continue
                    processed_ids.add(data["sample_id"])
                    results.append(data)
                except Exception:
                    pass

    print(f"Already processed in {os.path.basename(result_dir)}: {len(processed_ids)}")

    with open(eval_output_file, "a", encoding="utf-8") as f_out:
        for sample in tqdm(dataset, desc=f"Evaluating {os.path.basename(result_dir)}"):
            sample_id = sample["id"]
            if sample_id in processed_ids:
                continue

            txt_path = os.path.join(result_dir, f"output_text_{sample_id}.txt")
            if not os.path.exists(txt_path):
                print(f"[Warning] File not found for {sample_id}. Marking as failed.")
                output_data = {
                    "sample_id": sample_id,
                    "axis": sample.get("axis", "UNKNOWN"),
                    "criteria_met": False,
                    "explanation": "Generation failed. No output file found.",
                    "detailed_results": [
                        {
                            "criteria_met": False,
                            "explanation": "Missing output file",
                            "rubric_item": "All",
                        }
                    ],
                    "original_rubric": sample["rubric"],
                    "model_response": "[NO OUTPUT]",
                }
                f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                f_out.flush()
                results.append(output_data)
                continue

            with open(txt_path, "r", encoding="utf-8") as f:
                model_response = f.read().split("</think>")[-1].strip()

            conversation_history = build_grading_conversation_history(sample, model_response)
            rubric_items = parse_rubric(sample["rubric"])
            item_results = []

            for rubric_item in rubric_items:
                final_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                    conversation_history=conversation_history,
                    rubric_item=rubric_item,
                )

                eval_result = None
                for attempt in range(args.max_retries):
                    try:
                        response = client.chat.completions.create(
                            model=args.judge_model,
                            messages=[{"role": "user", "content": final_prompt}],
                        )

                        llm_output = response.choices[0].message.content
                        parsed_output = parse_json_response(llm_output)
                        if parsed_output.get("explanation") == "JSON Parsing Failed":
                            print(
                                f"JSON Parsing Error on attempt {attempt + 1}/{args.max_retries}. Retrying..."
                            )
                            if attempt < args.max_retries - 1:
                                continue

                        eval_result = parsed_output
                        break
                    except Exception as e:
                        print(
                            f"API Error on attempt {attempt + 1}/{args.max_retries} "
                            f"- rubric: {rubric_item[:20]}... : {e}"
                        )
                        if attempt < args.max_retries - 1:
                            time.sleep(args.retry_sleep_sec)
                        else:
                            eval_result = {
                                "criteria_met": False,
                                "explanation": (
                                    f"API/Parsing Failed after {args.max_retries} attempts. Error: {str(e)}"
                                ),
                            }
                            if eval_result is None or eval_result.get("explanation") == "JSON Parsing Failed":
                                eval_result = {
                                    "criteria_met": False,
                                    "explanation": (
                                        f"Failed to parse JSON after {args.max_retries} attempts."
                                    ),
                                }

                eval_result["rubric_item"] = rubric_item
                item_results.append(eval_result)

            final_pass = all(item["criteria_met"] for item in item_results)
            combined_explanation = "\n---\n".join(
                [
                    f"[Rubric: {item['rubric_item']}]\n"
                    f"Result: {item['criteria_met']}\n"
                    f"Explanation: {item['explanation']}"
                    for item in item_results
                ]
            )

            output_data = {
                "sample_id": sample_id,
                "axis": sample["axis"],
                "criteria_met": final_pass,
                "explanation": combined_explanation,
                "detailed_results": item_results,
                "original_rubric": sample["rubric"],
                "model_response": model_response,
            }

            f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            f_out.flush()
            results.append(output_data)

    overall_stats = calc_stats(results)
    axis_stats = {}
    for axis in parse_axes(args.axes):
        axis_stats[axis] = calc_stats([result for result in results if result.get("axis") == axis])

    inference_memory_breakdown = None
    if "INFERENCE_MEMORY" in parse_axes(args.axes):
        inference_memory_breakdown = build_inference_memory_breakdown(results, audio_cue_ids)

    stats_path = os.path.join(result_dir, f"{args.judge_model}_stats.json")
    stats_data = {
        "model": args.judge_model,
        "axes": parse_axes(args.axes),
        "inference_memory_subset": args.inference_memory_subset,
        "overall": overall_stats,
        "by_axis": axis_stats,
        "inference_memory_breakdown": inference_memory_breakdown,
        "eval_file": eval_output_file,
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Evaluation Complete for {os.path.basename(result_dir)}")
    print("-" * 60)
    print(
        f"Overall: {overall_stats['score']:5.2f}% "
        f"({overall_stats['passed']}/{overall_stats['total']})"
    )
    for axis, axis_stat in axis_stats.items():
        print(
            f"{axis:20}: {axis_stat['score']:5.2f}% "
            f"({axis_stat['passed']}/{axis_stat['total']})"
        )
    if inference_memory_breakdown is not None:
        for subset_name in ["audio_cue", "semantic"]:
            subset_stat = inference_memory_breakdown[subset_name]
            print(
                f"INFERENCE_MEMORY/{subset_name:9}: {subset_stat['score']:5.2f}% "
                f"({subset_stat['passed']}/{subset_stat['total']})"
            )
    print("=" * 60)
    print(f"Stats saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dirs", nargs="+", required=True)
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=os.environ.get("AUDIOMC_CACHE_DIR", "./data/audiomc"),
        help="Directory used by datasets.load_dataset for the ScaleAI/audiomc cache.",
    )
    parser.add_argument(
        "--audio_cue_file",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio-cue-im.txt"),
        help="Path to the list of INFERENCE_MEMORY sample IDs that belong to the audio-cue subset.",
    )
    parser.add_argument("--axes", type=str, default="INFERENCE_MEMORY,SELF_COHERENCE")
    parser.add_argument(
        "--inference_memory_subset",
        type=str,
        default="semantic",
        choices=["all", "audio_cue", "semantic"],
    )
    parser.add_argument("--judge_model", type=str, default="gpt-5-nano-2025-08-07")
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key. Falls back to the OPENAI_API_KEY environment variable.",
    )
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep_sec", type=float, default=2.0)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "OpenAI API key not provided. Set OPENAI_API_KEY or pass --api_key."
        )

    client = OpenAI(api_key=args.api_key)
    audio_cue_ids = load_audio_cue_ids(args.audio_cue_file)
    dataset = build_dataset(args, audio_cue_ids)

    for result_dir in args.result_dirs:
        if not os.path.exists(result_dir):
            print(f"Directory not found, skipping: {result_dir}")
            continue
        evaluate_result_dir(result_dir, dataset, client, args, audio_cue_ids)


if __name__ == "__main__":
    main()
