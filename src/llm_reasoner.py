import json
import os
from pathlib import Path

import requests
from pydantic import BaseModel, Field


URL = "https://api.sarvam.ai/v1/chat/completions"
MODEL = os.getenv("SARVAM_MODEL", "sarvam-105b")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "experiments" / "juice_bottle_res256_20260320_214217" / "anomaly_registry.json"
BATCH_SIZE = int(os.getenv("SIGNALFLOW_BATCH_SIZE", "12"))

QUESTIONS = [
    "Which images are anomalous?",
    "Which images are normal?",
    "How many normal images are there?",
]


class InspectionResult(BaseModel):
    image_id: str = Field(description="Use image_ref from the provided data as the identifier.")
    predicted_status: str = Field(description='Either "normal" or "anomaly".')
    ground_truth_status: str = Field(description='Either "normal" or "anomaly".')
    is_correct: bool
    error_type: str = Field(description='One of "correct", "false_positive", or "false_negative".')
    short_explanation: str = Field(description="Brief explanation under 12 words.")


def load_registry_data(registry_path: Path, limit: int | None = None) -> list[dict]:
    with open(registry_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data["entries"]
    if limit is not None:
        entries = entries[:limit]

    extracted_data = []
    for entry in entries:
        image_path = Path(entry["image_path"])
        extracted_data.append(
            {
                "image_id": entry["image_id"],
                "image_ref": f"{image_path.parent.name}/{image_path.stem}",
                "pred_label_calibrated": entry["pred_label_calibrated"],
                "gt_label": entry["gt_label"],
                "rank": entry["rank"],
            }
        )
    return extracted_data


def build_prompt(question: str, extracted_data: list[dict]) -> str:
    return f"""
Return valid JSON only.
Return a JSON array only.
No reasoning text.
No markdown.
No code fences.
No prose before or after JSON.

Rules:
- pred_label_calibrated = 1 means predicted_status = "anomaly"
- pred_label_calibrated = 0 means predicted_status = "normal"
- gt_label = 1 means ground_truth_status = "anomaly"
- gt_label = 0 means ground_truth_status = "normal"
- if predicted_status == ground_truth_status then is_correct = true and error_type = "correct"
- if predicted_status == "anomaly" and ground_truth_status == "normal" then is_correct = false and error_type = "false_positive"
- if predicted_status == "normal" and ground_truth_status == "anomaly" then is_correct = false and error_type = "false_negative"
- use image_ref from the data as the image identifier
- short_explanation must be under 12 words
- sort output by rank ascending

Required JSON schema for every array item:
{{
  "image_id": "string",
  "predicted_status": "normal_or_anomaly",
  "ground_truth_status": "normal_or_anomaly",
  "is_correct": true,
  "error_type": "correct_or_false_positive_or_false_negative",
  "short_explanation": "string"
}}

Question mapping:
- "Which images are anomalous?" -> return only rows where predicted_status = "anomaly"
- "Which images are normal?" -> return only rows where predicted_status = "normal"
- "How many normal images are there?" -> return only rows where predicted_status = "normal"; the caller will count the array length

Data:
{json.dumps(extracted_data, indent=2)}

Question:
{question}
"""


def ask_question(question: str, extracted_data: list[dict]) -> requests.Response:
    api_key = os.environ["SARVAM_API_KEY"]
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": build_prompt(question, extracted_data)}],
        "temperature": 0,
    }
    return requests.post(URL, headers=headers, json=payload, timeout=120)


def extract_message_content(response: requests.Response) -> tuple[str, str | None]:
    data = response.json()
    choice = data["choices"][0]
    finish_reason = choice.get("finish_reason")
    message = choice.get("message", {})
    content = message.get("content")

    if content is None:
        content = "[]"

    return content, finish_reason


def chunk_entries(entries: list[dict], batch_size: int) -> list[list[dict]]:
    return [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]


def run_single_batch(question: str, batch: list[dict]) -> tuple[list[InspectionResult], list[str]]:
    response = ask_question(question, batch)
    content, finish_reason = extract_message_content(response)
    finish_reason_value = finish_reason or "unknown"

    try:
        parsed = json.loads(content)
        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON array from the model.")
        validated = [InspectionResult.model_validate(item) for item in parsed]
        return validated, [finish_reason_value]
    except (json.JSONDecodeError, ValueError):
        if len(batch) == 1:
            raise

        mid = len(batch) // 2
        left_results, left_reasons = run_single_batch(question, batch[:mid])
        right_results, right_reasons = run_single_batch(question, batch[mid:])
        return left_results + right_results, [finish_reason_value] + left_reasons + right_reasons


def run_question_in_batches(question: str, extracted_data: list[dict]) -> tuple[list[InspectionResult], list[str]]:
    combined_results: list[InspectionResult] = []
    finish_reasons: list[str] = []

    for batch in chunk_entries(extracted_data, BATCH_SIZE):
        batch_results, batch_reasons = run_single_batch(question, batch)
        combined_results.extend(batch_results)
        finish_reasons.extend(batch_reasons)

    return combined_results, finish_reasons


def main(registry_path: Path = REGISTRY_PATH) -> None:
    extracted_data = load_registry_data(registry_path)
    print(f"Loaded {len(extracted_data)} entries for prompting.")
    print(f"Using batch size: {BATCH_SIZE}")

    for question in QUESTIONS:
        print(f"\nQuestion: {question}")
        results, _finish_reasons = run_question_in_batches(question, extracted_data)
        print(json.dumps([item.model_dump() for item in results], indent=2))
        print(f"Count: {len(results)}")
