#!/usr/bin/env python3
"""Analyze whether search results were injected into reasoning chains."""
import argparse
import json
import re
from pathlib import Path

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"


def analyze_output_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for row in data:
        output = row.get("Output", "")
        queries = re.findall(
            re.escape(BEGIN_SEARCH_QUERY) + r"(.*?)" + re.escape(END_SEARCH_QUERY),
            output,
            flags=re.DOTALL,
        )
        blocks = re.findall(
            re.escape(BEGIN_SEARCH_RESULT) + r"(.*?)" + re.escape(END_SEARCH_RESULT),
            output,
            flags=re.DOTALL,
        )
        items.append(
            {
                "id": row.get("id"),
                "num_queries": len(queries),
                "num_result_blocks": len(blocks),
                "injection_ok": len(queries) == 0 or len(blocks) >= len(queries),
                "correct": row.get("Metrics", {}).get("em") == 1,
            }
        )
    searched = [i for i in items if i["num_queries"] > 0]
    return {
        "file": str(path),
        "total": len(items),
        "used_search": len(searched),
        "injection_ok": sum(1 for i in searched if i["injection_ok"]),
        "items": items,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    args = parser.parse_args()
    report = analyze_output_json(args.result_json)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
