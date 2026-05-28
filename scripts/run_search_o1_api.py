# run_search_o1_api.py
"""Search-o1 inference via DeepSeek V4 API (flash / pro)."""
import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from bing_search import (
    bing_web_search,
    extract_relevant_info,
    extract_snippet_with_context,
    fetch_page_content,
)
from config import (
    BING_ENDPOINT,
    BING_SUBSCRIPTION_KEY,
    DEEPSEEK_MODEL,
    JINA_API_KEY,
    get_deepseek_model,
    require_deepseek_api_key,
)
from deepseek_client import deepseek_chat, message_to_text
from evaluate import extract_answer, run_evaluation
from prompts import (
    get_code_search_o1_instruction,
    get_gpqa_search_o1_instruction,
    get_math_search_o1_instruction,
    get_multiqa_search_o1_instruction,
    get_singleqa_search_o1_instruction,
    get_task_instruction_code,
    get_task_instruction_math,
    get_task_instruction_multi_choice,
    get_task_instruction_openqa,
    get_webpage_to_reasonchain_instruction,
)

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"


@dataclass
class GenerationOutput:
    text: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Search-o1 with DeepSeek V4 API (flash or pro)."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=[
            "gpqa",
            "math500",
            "aime",
            "amc",
            "livecode",
            "nq",
            "triviaqa",
            "hotpotqa",
            "2wiki",
            "musique",
            "bamboogle",
        ],
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["test", "diamond", "main", "extended"],
    )
    parser.add_argument("--subset_num", type=int, default=-1)
    parser.add_argument("--max_search_limit", type=int, default=10)
    parser.add_argument("--max_turn", type=int, default=15)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_doc_len", type=int, default=3000)
    parser.add_argument("--use_jina", type=bool, default=True)
    parser.add_argument(
        "--model_variant",
        type=str,
        default=None,
        choices=["flash", "pro"],
        help="DeepSeek V4 variant. Defaults to DEEPSEEK_MODEL in .env.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        choices=["high", "max"],
        help="Thinking effort for main reasoning. Defaults to env.",
    )
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument(
        "--bing_subscription_key",
        type=str,
        default=None,
        help="Bing key. Falls back to BING_SUBSCRIPTION_KEY in .env.",
    )
    parser.add_argument(
        "--bing_endpoint",
        type=str,
        default=None,
        help="Bing endpoint. Falls back to BING_ENDPOINT in .env.",
    )
    parser.add_argument(
        "--jina_api_key",
        type=str,
        default=None,
        help="Jina key. Falls back to JINA_API_KEY in .env.",
    )
    return parser.parse_args()


def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def replace_recent_steps(origin_str: str, replace_str: str) -> str:
    step_pattern = re.compile(r"Step\s+(\d+):\s*")

    def parse_steps(text):
        steps = {}
        current_step_num = None
        current_content = []
        for line in text.splitlines():
            step_match = step_pattern.match(line)
            if step_match:
                if current_step_num is not None:
                    steps[current_step_num] = "\n".join(current_content).strip()
                current_step_num = int(step_match.group(1))
                content = line[step_match.end() :].strip()
                current_content = [content] if content else []
            elif current_step_num is not None:
                current_content.append(line)
        if current_step_num is not None:
            steps[current_step_num] = "\n".join(current_content).strip()
        return steps

    origin_steps = parse_steps(origin_str)
    replace_steps = parse_steps(replace_str)
    for step_num, content in replace_steps.items():
        if "DELETE THIS STEP" in content:
            origin_steps.pop(step_num, None)
        else:
            origin_steps[step_num] = content
    sorted_steps = sorted(origin_steps.items())
    return "\n\n".join([content for _, content in sorted_steps])


def build_instruction_and_prompt(dataset_name: str, item: dict, max_search_limit: int):
    question = item["Question"]
    if dataset_name in ["nq", "triviaqa", "hotpotqa", "musique", "bamboogle", "2wiki"]:
        if dataset_name in ["nq", "triviaqa"]:
            instruction = get_singleqa_search_o1_instruction(max_search_limit)
        else:
            instruction = get_multiqa_search_o1_instruction(max_search_limit)
        user_prompt = get_task_instruction_openqa(question, model_name="qwq")
    elif dataset_name in ["math500", "aime", "amc"]:
        instruction = get_math_search_o1_instruction(max_search_limit)
        user_prompt = get_task_instruction_math(question, model_name="qwq")
    elif dataset_name == "gpqa":
        instruction = get_gpqa_search_o1_instruction(max_search_limit)
        user_prompt = get_task_instruction_multi_choice(question, model_name="qwq")
    elif dataset_name == "livecode":
        instruction = get_code_search_o1_instruction(max_search_limit)
        question_title = item.get("question_title", "")
        user_prompt = get_task_instruction_code(
            question, question_title=question_title, model_name="qwq"
        )
    else:
        instruction = ""
        user_prompt = ""
    return instruction + user_prompt


def truncate_prev_reasoning(output: str) -> str:
    all_reasoning_steps = output.replace("\n\n", "\n").split("\n")
    truncated_prev_reasoning = ""
    for i, step in enumerate(all_reasoning_steps):
        truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

    prev_steps = truncated_prev_reasoning.split("\n\n")
    if len(prev_steps) <= 5:
        return "\n\n".join(prev_steps)
    truncated = ""
    for i, step in enumerate(prev_steps):
        if (
            i == 0
            or i >= len(prev_steps) - 4
            or BEGIN_SEARCH_QUERY in step
            or BEGIN_SEARCH_RESULT in step
        ):
            truncated += step + "\n\n"
        elif truncated[-len("\n\n...\n\n") :] != "\n\n...\n\n":
            truncated += "...\n\n"
    return truncated.strip("\n")


def run_generation_api(
    sequences: List[Dict],
    *,
    model_variant: str,
    max_tokens: int,
    reasoning_effort: Optional[str],
) -> List[GenerationOutput]:
    outputs = []
    for seq in sequences:
        response = deepseek_chat(
            [{"role": "user", "content": seq["prompt"]}],
            variant=model_variant,
            max_tokens=max_tokens,
            stop=[END_SEARCH_QUERY],
            reasoning_effort=reasoning_effort,
        )
        text = message_to_text(response.choices[0].message)
        if extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY) and not text.rstrip().endswith(
            END_SEARCH_QUERY
        ):
            text += END_SEARCH_QUERY
        outputs.append(GenerationOutput(text=text))
    return outputs


def generate_webpage_to_reasonchain_batch_api(
    prev_reasonings: List[str],
    search_queries: List[str],
    documents: List[str],
    *,
    model_variant: str,
    batch_output_records: List[Dict],
    max_tokens: int,
) -> List[str]:
    extracted_infos = []
    for prev_reasoning, search_query, document in zip(
        prev_reasonings, search_queries, documents
    ):
        user_prompt = get_webpage_to_reasonchain_instruction(
            prev_reasoning, search_query, document
        )
        response = deepseek_chat(
            [{"role": "user", "content": user_prompt}],
            variant=model_variant,
            max_tokens=max_tokens,
            thinking=False,
        )
        raw_output = message_to_text(
            response.choices[0].message, include_reasoning=False
        )
        extracted_info = extract_answer(raw_output, mode="infogen")
        batch_output_records.append(
            {
                "prompt": user_prompt,
                "raw_output": raw_output,
                "extracted_info": extracted_info,
            }
        )
        extracted_infos.append(extracted_info)
    return extracted_infos


def resolve_max_tokens(dataset_name: str, max_tokens: Optional[int]) -> int:
    if max_tokens is not None:
        return max_tokens
    if dataset_name in ["aime", "amc", "livecode"]:
        return 32768
    return 20480


def main():
    args = parse_args()
    require_deepseek_api_key()

    dataset_name = args.dataset_name
    split = args.split
    subset_num = args.subset_num
    max_search_limit = args.max_search_limit
    max_turn = args.max_turn
    top_k = args.top_k
    max_doc_len = args.max_doc_len
    model_variant = (args.model_variant or DEEPSEEK_MODEL).strip().lower()
    if model_variant not in ("flash", "pro"):
        model_variant = "flash" if "flash" in model_variant else "pro"
    model_id = get_deepseek_model(model_variant)
    reasoning_effort = args.reasoning_effort
    max_tokens = resolve_max_tokens(dataset_name, args.max_tokens)

    bing_subscription_key = args.bing_subscription_key or BING_SUBSCRIPTION_KEY
    bing_endpoint = args.bing_endpoint or BING_ENDPOINT
    jina_api_key = args.jina_api_key or JINA_API_KEY
    if args.jina_api_key == "None":
        jina_api_key = None

    if not bing_subscription_key:
        raise ValueError(
            "Bing subscription key required. Set BING_SUBSCRIPTION_KEY in .env "
            "or pass --bing_subscription_key."
        )

    if dataset_name in ["nq", "triviaqa", "hotpotqa", "musique", "bamboogle", "2wiki"]:
        max_search_limit = 5
        if dataset_name in ["hotpotqa", "musique", "bamboogle", "2wiki"]:
            max_search_limit = 10
            max_turn = 15
        top_k = 10
        max_doc_len = 3000

    if dataset_name == "livecode":
        data_path = f"./data/LiveCodeBench/{split}.json"
    elif dataset_name in ["math500", "gpqa", "aime", "amc"]:
        data_path = f"./data/{dataset_name.upper()}/{split}.json"
    else:
        data_path = f"./data/QA_Datasets/{dataset_name}.json"

    print("-----------------------")
    print(f"Using {dataset_name} {split} set.")
    print(f"DeepSeek model: {model_id} ({model_variant})")
    print("-----------------------")

    cache_dir = "./cache"
    search_cache_path = os.path.join(cache_dir, "search_cache.json")
    url_cache_path = os.path.join(cache_dir, "url_cache.json")
    os.makedirs(cache_dir, exist_ok=True)

    search_cache = {}
    url_cache = {}
    if os.path.exists(search_cache_path):
        with open(search_cache_path, "r", encoding="utf-8") as f:
            search_cache = json.load(f)
    if os.path.exists(url_cache_path):
        with open(url_cache_path, "r", encoding="utf-8") as f:
            url_cache = json.load(f)

    def save_caches():
        with open(search_cache_path, "w", encoding="utf-8") as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)
        with open(url_cache_path, "w", encoding="utf-8") as f:
            json.dump(url_cache, f, ensure_ascii=False, indent=2)

    output_dir = f"./outputs/runs.api/{dataset_name}.{model_variant}.search_o1"
    os.makedirs(output_dir, exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as json_file:
        filtered_data = json.load(json_file)

    input_list = [
        build_instruction_and_prompt(dataset_name, item, max_search_limit)
        for item in filtered_data
    ]
    if subset_num != -1:
        input_list = input_list[:subset_num]
        filtered_data = filtered_data[:subset_num]

    active_sequences = [
        {
            "item": item,
            "prompt": prompt,
            "output": "",
            "finished": False,
            "history": [],
            "search_count": 0,
            "executed_search_queries": set(),
        }
        for item, prompt in zip(filtered_data, input_list)
    ]

    batch_output_records = []
    start_time = time.time()
    turn = 0

    while True:
        sequences_needing_generation = [seq for seq in active_sequences if not seq["finished"]]
        if not sequences_needing_generation:
            break

        turn += 1
        print(f"\n-------------- Turn {turn} --------------")
        print(f"We have {len(sequences_needing_generation)} sequences needing generation...")
        outputs = run_generation_api(
            sequences_needing_generation,
            model_variant=model_variant,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        print("Generation completed, processing outputs...")

        batch_relevant_info = []
        batch_prev_reasonings = []
        batch_search_queries = []
        batch_documents = []
        batch_sequences = []
        all_urls_to_fetch = set()

        for seq, out in zip(sequences_needing_generation, outputs):
            text = out.text
            seq["history"].append(text)
            seq["prompt"] += text
            seq["output"] += text
            search_query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

            if search_query and seq["output"].rstrip().endswith(END_SEARCH_QUERY):
                if (
                    seq["search_count"] < max_search_limit
                    and search_query not in seq["executed_search_queries"]
                ):
                    if search_query in search_cache:
                        results = search_cache[search_query]
                        print(f'Using cached search results for query: "{search_query}"')
                    else:
                        try:
                            results = bing_web_search(
                                search_query,
                                bing_subscription_key,
                                bing_endpoint,
                                market="en-US",
                                language="en",
                            )
                            search_cache[search_query] = results
                            print(f'Executed and cached search for query: "{search_query}"')
                        except Exception as e:
                            print(f"Error during search query '{search_query}': {e}")
                            search_cache[search_query] = {}
                            results = {}

                    relevant_info = extract_relevant_info(results)[:top_k]
                    seq["relevant_info"] = relevant_info
                    urls_to_fetch = [it["url"] for it in relevant_info]
                    for url in urls_to_fetch:
                        if url not in url_cache:
                            all_urls_to_fetch.add(url)

                    batch_relevant_info.append(relevant_info)
                    batch_prev_reasonings.append(truncate_prev_reasoning(seq["output"]))
                    batch_search_queries.append(search_query)
                    batch_sequences.append(seq)
                    seq["search_count"] += 1
                    seq["executed_search_queries"].add(search_query)
                elif seq["search_count"] >= max_search_limit:
                    limit_message = (
                        f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. "
                        f"You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                    )
                    seq["prompt"] += limit_message
                    seq["output"] += limit_message
                    seq["history"].append(limit_message)
                elif search_query in seq["executed_search_queries"]:
                    limit_message = (
                        f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. "
                        f"Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                    )
                    seq["prompt"] += limit_message
                    seq["output"] += limit_message
                    seq["history"].append(limit_message)
            else:
                seq["finished"] = True
                print("Sequence marked as complete.")

        if all_urls_to_fetch:
            print(f"Fetching {len(all_urls_to_fetch)} URLs...")
            try:
                fetched_contents = fetch_page_content(
                    list(all_urls_to_fetch),
                    use_jina=args.use_jina,
                    jina_api_key=jina_api_key,
                )
            except Exception as e:
                print(f"Error during batch URL fetching: {e}")
                fetched_contents = {url: f"Error fetching URL: {e}" for url in all_urls_to_fetch}
            url_cache.update(fetched_contents)

        for relevant_info in batch_relevant_info:
            formatted_documents = ""
            for i, doc_info in enumerate(relevant_info):
                url = doc_info["url"]
                raw_context = url_cache.get(url, "")
                doc_info["snippet"] = doc_info["snippet"].replace("<b>", "").replace("</b>", "")
                success, filtered_context = extract_snippet_with_context(
                    raw_context, doc_info["snippet"], context_chars=max_doc_len
                )
                context = filtered_context if success else raw_context[: max_doc_len * 2]
                doc_info["context"] = context
                formatted_documents += f"**Web Page {i + 1}:**\n"
                formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
            batch_documents.append(formatted_documents)

        if batch_sequences:
            print(
                f"Batch processing {len(batch_sequences)} sequences with Reason-in-Documents..."
            )
            webpage_analyses = generate_webpage_to_reasonchain_batch_api(
                batch_prev_reasonings,
                batch_search_queries,
                batch_documents,
                model_variant=model_variant,
                batch_output_records=batch_output_records,
                max_tokens=max_tokens,
            )
            for seq, analysis in zip(batch_sequences, webpage_analyses):
                if isinstance(analysis, str):
                    append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                else:
                    append_text = replace_recent_steps(seq["output"], analysis)
                seq["prompt"] += append_text
                seq["output"] += append_text
                seq["history"].append(append_text)

        if turn >= max_turn:
            print(f"Maximum number of turns ({max_turn}) reached, stopping.")
            break

    total_time = time.time() - start_time
    t = time.localtime()
    batch_output_file = os.path.join(
        output_dir,
        f"{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.info_extract.json",
    )
    with open(batch_output_file, "w", encoding="utf-8") as f:
        json.dump(batch_output_records, f, ensure_ascii=False, indent=2)
    print(f"Batch outputs saved to {batch_output_file}")

    output_list = [seq["output"] for seq in active_sequences]
    run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split)
    save_caches()
    print("Process completed.")


if __name__ == "__main__":
    main()
