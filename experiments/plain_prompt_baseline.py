from pathlib import Path
import json

from langchain_ollama import ChatOllama


llm = ChatOllama(model="llama3.1:8b")
REGISTRY_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = REGISTRY_DIR / "juice_bottle_res256_20260320_214217" / "anomaly_registry.json"


def explain_registry(registry_path: Path) -> None:
    with open(registry_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data["entries"]

    extracted_data = []

    for entry in entries:
        extracted_data.append({
            "image_id": entry["image_id"],
            "image_path": entry["image_path"],
            "pred_score": entry["pred_score"],
            "pred_label_calibrated": entry["pred_label_calibrated"],
            "gt_label": entry["gt_label"],
            "threshold_used": entry["threshold_used"],
            "rank": entry["rank"]
        })



    for _ in range(20):

        user_query = input("\nAsk a question about the inspection results: ")

        if user_query.lower() in ["exit", "quit"]:
            break

        prompt = f"""
Return valid JSON only.

Context:
- System: SignalFlow AI Quality Inspection
- Dataset: MVTec LOCO
- Domain: juice bottle inspection
- Labels: 0 = normal, 1 = anomaly
- pred_score: higher means more anomalous
- threshold_used: anomaly decision threshold
- pred_label_calibrated: final predicted label
- gt_label: ground truth
- rank: anomaly rank, 1 = highest anomaly score

Rules:
- If pred_score >= threshold_used, predicted_status = "anomaly"
- Else predicted_status = "normal"
- If gt_label == 1, ground_truth_status = "anomaly"
- Else ground_truth_status = "normal"
- If predicted_status == ground_truth_status, is_correct = true and error_type = "correct"
- If predicted_status == "anomaly" and ground_truth_status == "normal", error_type = "false_positive"
- If predicted_status == "normal" and ground_truth_status == "anomaly", error_type = "false_negative"
- short_explanation must be under 20 words
- Do not output anything except JSON

Required JSON schema:

  "image_id": "string",
  "image_path": "string",
  "predicted_status": "normal_or_anomaly",
  "ground_truth_status": "normal_or_anomaly",
  "is_correct": true,
  "error_type": "correct_or_false_positive_or_false_negative",
  "short_explanation": "string"


Data:
{extracted_data}

User question:
{user_query}

Answer clearly using the data.
"""

        response = llm.invoke(prompt)

        print("\nLLM Analysis:\n")
        print(response.content)


def main() -> None:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry file not found: {REGISTRY_PATH}")

    explain_registry(REGISTRY_PATH)


if __name__ == "__main__":
    main()
