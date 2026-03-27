import json
import os
from pathlib import Path

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


REGISTRY_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = REGISTRY_DIR / "juice_bottle_res256_20260320_214217" / "anomaly_registry.json"
BATCH_SIZE = int(os.getenv("SIGNALFLOW_BATCH_SIZE", "8"))
BACKEND = os.getenv("SIGNALFLOW_LLM_BACKEND", "ollama")

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


class InspectionBatchResponse(BaseModel):
    results: list[InspectionResult]


def load_registry_data(registry_path: Path) -> list[dict]:
    with open(registry_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    extracted_data = []
    for entry in data["entries"]:
        image_path = Path(entry["image_path"])
        extracted_data.append(
            {
                "image_ref": f"{image_path.parent.name}/{image_path.stem}",
                "pred_label_calibrated": entry["pred_label_calibrated"],
                "gt_label": entry["gt_label"],
                "rank": entry["rank"],
            }
        )
    return extracted_data


def get_llm():
    if BACKEND == "ollama":
        model = os.getenv("SIGNALFLOW_OLLAMA_MODEL", "llama3.1:8b")
        return ChatOllama(model=model, temperature=0)

    if BACKEND == "openai_compatible":
        api_key = os.environ["SIGNALFLOW_API_KEY"]
        base_url = os.environ["SIGNALFLOW_BASE_URL"]
        model = os.getenv("SIGNALFLOW_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)

    raise ValueError(f"Unsupported SIGNALFLOW_LLM_BACKEND: {BACKEND}")


def build_chain():
    parser = PydanticOutputParser(pydantic_object=InspectionBatchResponse)
    prompt = PromptTemplate(
        template="""
You are the SignalFlow reasoning layer.
Reason only from the provided rows and rules.

Rules:
- pred_label_calibrated = 1 means predicted_status = "anomaly"
- pred_label_calibrated = 0 means predicted_status = "normal"
- gt_label = 1 means ground_truth_status = "anomaly"
- gt_label = 0 means ground_truth_status = "normal"
- if predicted_status == ground_truth_status then is_correct = true and error_type = "correct"
- if predicted_status == "anomaly" and ground_truth_status == "normal" then is_correct = false and error_type = "false_positive"
- if predicted_status == "normal" and ground_truth_status == "anomaly" then is_correct = false and error_type = "false_negative"
- use image_ref as image_id in the output
- sort results by rank ascending
- short_explanation must be under 12 words

Question mapping:
- "Which images are anomalous?" -> include only rows where predicted_status = "anomaly"
- "Which images are normal?" -> include only rows where predicted_status = "normal"
- "How many normal images are there?" -> include only rows where predicted_status = "normal"; the caller will count len(results)

{format_instructions}

Data:
{data}

Question:
{question}
""".strip(),
        input_variables=["data", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | get_llm() | parser


def chunk_entries(entries: list[dict], batch_size: int) -> list[list[dict]]:
    return [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]


def run_single_batch(chain, question: str, batch: list[dict]) -> list[InspectionResult]:
    try:
        response = chain.invoke({"data": json.dumps(batch, indent=2), "question": question})
        return response.results
    except Exception:
        if len(batch) == 1:
            raise

        mid = len(batch) // 2
        left = run_single_batch(chain, question, batch[:mid])
        right = run_single_batch(chain, question, batch[mid:])
        return left + right


def run_question_in_batches(chain, question: str, extracted_data: list[dict]) -> list[InspectionResult]:
    combined_results: list[InspectionResult] = []
    for batch in chunk_entries(extracted_data, BATCH_SIZE):
        combined_results.extend(run_single_batch(chain, question, batch))
    return combined_results


def main() -> None:
    chain = build_chain()
    extracted_data = load_registry_data(REGISTRY_PATH)

    print(f"Loaded {len(extracted_data)} entries.")
    print(f"Backend: {BACKEND}")
    print(f"Batch size: {BATCH_SIZE}")

    for question in QUESTIONS:
        print(f"\nQuestion: {question}")
        results = run_question_in_batches(chain, question, extracted_data)
        print(json.dumps([item.model_dump() for item in results], indent=2))
        print(f"Count: {len(results)}")


if __name__ == "__main__":
    main()
