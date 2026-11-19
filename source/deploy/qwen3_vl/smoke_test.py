#!/usr/bin/env python3

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from urllib.request import Request, urlopen


SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["confirm", "reject", "uncertain"]},
        "defect_class_id": {
            "type": "string",
            "enum": ["NONE", "L2-G", "L2-P", "L3-06", "L3-07", "L3-MIS"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
        "needs_human_review": {"type": "boolean"},
    },
    "required": [
        "decision",
        "defect_class_id",
        "confidence",
        "evidence",
        "needs_human_review",
    ],
    "additionalProperties": False,
}


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL OpenAI-compatible smoke test")
    parser.add_argument("--image", type=Path, help="Optional candidate crop")
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8001/v1"))
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path(os.getenv("LLM_API_KEY_FILE", str(Path.home() / ".spaceqa-qwen3vl-api-key"))),
    )
    args = parser.parse_args()

    content = [
        {
            "type": "text",
            "text": (
                "Review this OPC space-science annotation candidate. Return only the requested "
                "JSON. A crop without enough context must be marked uncertain. Do not calculate "
                "dataset scores or pollution ratios."
            ),
        }
    ]
    if args.image:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(args.image)}})

    payload = {
        "model": "Qwen3-VL-8B-Instruct",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 256,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "opc_defect_review", "strict": True, "schema": SCHEMA},
        },
    }
    request = Request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {args.api_key_file.read_text().strip()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        result = json.load(response)
    output = json.loads(result["choices"][0]["message"]["content"])
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
