import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List


def run_eval(hf_model_id: str, quant_mode: str, output_dir: str, subjects: str, torch_dtype: str, device_map: str, max_new_tokens: int) -> None:
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "evaluate_from_api.py"),
        "--model_name", "hf-local",
        "--hf_model_id", hf_model_id,
        "--hf_quantization", quant_mode,
        "--torch_dtype", torch_dtype,
        "--device_map", device_map,
        "--max_new_tokens", str(max_new_tokens),
        "--assigned_subjects", subjects,
        "--output_dir", output_dir,
    ]
    subprocess.run(cmd, check=True)


def collect_results(output_dir: str) -> Dict[str, float]:
    metrics = {}
    for file_name in os.listdir(output_dir):
        if file_name.endswith("_summary.json"):
            path = os.path.join(output_dir, file_name)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if "total" in data and "acc" in data["total"]:
                    metrics[file_name.replace("_summary.json", "")] = data["total"]["acc"]
            except Exception:
                continue
    return metrics


def aggregate_total_accuracy(output_dir: str) -> float:
    totals = []
    for file_name in os.listdir(output_dir):
        if file_name.endswith("_summary.json"):
            path = os.path.join(output_dir, file_name)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if "total" in data and data["total"]["corr"] + data["total"]["wrong"] > 0:
                    corr = data["total"]["corr"]
                    wrong = data["total"]["wrong"]
                    totals.append((corr, wrong))
            except Exception:
                continue
    if not totals:
        return 0.0
    sum_corr = sum(c for c, _ in totals)
    sum_wrong = sum(w for _, w in totals)
    return float(sum_corr) / float(sum_corr + sum_wrong) if (sum_corr + sum_wrong) > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_id", required=True, type=str, help="HF model id or local path")
    parser.add_argument("--output_root", type=str, default="quant_eval_results", help="Root directory for outputs")
    parser.add_argument("--assigned_subjects", type=str, default="all", help="Comma separated subjects or 'all'")
    parser.add_argument("--quant_modes", type=str, default="none,8bit,4bit", help="Comma separated: none,8bit,4bit")
    parser.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--report_file", type=str, default="quant_report.json")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    quant_list: List[str] = [q.strip() for q in args.quant_modes.split(",") if q.strip()]
    summary: Dict[str, Dict[str, float]] = {}

    for quant in quant_list:
        out_dir = os.path.join(args.output_root, f"{quant}")
        os.makedirs(out_dir, exist_ok=True)
        run_eval(
            hf_model_id=args.hf_model_id,
            quant_mode=quant,
            output_dir=out_dir,
            subjects=args.assigned_subjects,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            max_new_tokens=args.max_new_tokens,
        )
        total_acc = aggregate_total_accuracy(out_dir)
        per_subject = collect_results(out_dir)
        summary[quant] = {"total_acc": total_acc, **{f"subject::{k}": v for k, v in per_subject.items()}}

    report_path = os.path.join(args.output_root, args.report_file)
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()



