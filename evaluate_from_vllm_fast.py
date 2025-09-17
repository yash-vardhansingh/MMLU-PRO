#!/usr/bin/env python3
# evaluate_from_vllm_fast.py
import os
import json
import re
import time
import random
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from datasets import load_dataset

random.seed(12345)

# --------- Utils ---------
CHOICE_MAP = "ABCDEFGHIJ"

# replace VLLMClient with this:
class UniversalClient:
    def __init__(self, base_url, model, max_tokens=1024, temperature=0.0,
                 timeout=300, backend="auto"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.session = requests.Session()

        # detect backend
        if backend == "auto":
            if self.base_url.endswith("/v1") or "/v1" in self.base_url:
                self.backend = "openai"
            elif "/api/" in self.base_url:
                self.backend = "ollama"
            else:
                # default to openai-style
                self.backend = "openai"
        else:
            self.backend = backend  # "openai" or "ollama"

    def chat(self, prompt):
        if self.backend == "openai":
            # OpenAI/vLLM style: POST /v1/chat/completions
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:600]}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        elif self.backend == "ollama":
            # Ollama native: prefer /api/chat (messages) if base_url doesn't already include a path
            if self.base_url.endswith("/api/chat"):
                url = self.base_url
                use_chat = True
            elif self.base_url.endswith("/api/generate"):
                url = self.base_url
                use_chat = False
            else:
                # no path given -> use /api/chat
                url = f"{self.base_url}/api/chat"
                use_chat = True

            if use_chat:
                payload = {
                    "model": self.model,              # e.g., "llama3:8b" or your local tag
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                }
                resp = self.session.post(url, json=payload, timeout=self.timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:600]}")
                data = resp.json()
                # chat response shape: {"message": {"content": "..."} , ...}
                return data["message"]["content"]
            else:
                # /api/generate form (single prompt)
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                }
                resp = self.session.post(url, json=payload, timeout=self.timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:600]}")
                data = resp.json()
                # generate response shape: {"response": "..."}
                return data["response"]
        else:
            raise ValueError(f"Unknown backend: {self.backend}")


def format_example(question, options, cot_content=""):
    if not cot_content:
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = f"Question: {question}\nOptions: "
    for i, opt in enumerate(options):
        example += f"{CHOICE_MAP[i]}. {opt}\n"
    example += "Answer: " + cot_content + "\n\n"
    return example

def format_query_only(question, options):
    example = f"Question: {question}\nOptions: "
    for i, opt in enumerate(options):
        example += f"{CHOICE_MAP[i]}. {opt}\n"
    example += "Answer: "
    return example

ANSWER_PATTERNS = [
    re.compile(r"answer is\s*\(?([A-J])\)?", re.IGNORECASE),
    re.compile(r".*[aA]nswer:\s*([A-J])"),
    re.compile(r"\b([A-J])\b(?!.*\b[A-J]\b)", re.DOTALL),
]

def extract_answer(text):
    if not text:
        return None
    for pat in ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    return None

def preprocess_split(hf_split):
    """Remove N/A options and regroup by category."""
    cleaned = []
    for row in hf_split:
        opts = [o for o in row["options"] if o != "N/A"]
        row = dict(row)
        row["options"] = opts
        cleaned.append(row)
    by_cat = defaultdict(list)
    for row in cleaned:
        by_cat[row["category"]].append(row)
    return dict(by_cat)

def load_mmlu_pro():
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df = preprocess_split(ds["test"])
    val_df  = preprocess_split(ds["validation"])
    return test_df, val_df

# --------- vLLM client ---------
class VLLMClient:
    def __init__(self, base_url, model, max_tokens=1024, temperature=0.0, timeout=300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.session = requests.Session()

    def chat(self, prompt):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        url = f"{self.base_url}/chat/completions"
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

# --------- Evaluation ---------
def build_prompt(category, cot_examples, q, max_cot_examples):
    """Few-shot CoT + the test question. Keep it short for speed."""
    prompt = (
        f"The following are multiple choice questions (with answers) about {category}. "
        f"Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n"
    )
    for ex in cot_examples[:max_cot_examples]:
        prompt += format_example(ex["question"], ex["options"], ex.get("cot_content", ""))
    prompt += format_query_only(q["question"], q["options"])
    return prompt

def worker_one_question(client, category, q, cot_examples, max_cot_examples, retries=3, backoff=1.5):
    """Call vLLM with retries; return (question_id, pred, raw_response)."""
    prompt = build_prompt(category, cot_examples, q, max_cot_examples)
    last_err = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = client.chat(prompt)
            # minor cleanup
            resp = resp.replace("**", "")
            pred = extract_answer(resp)
            return q["question_id"], pred, resp, (time.time() - t0)
        except Exception as e:
            last_err = e
            time.sleep(backoff ** attempt)
    # on failure
    return q["question_id"], None, f"[ERROR] {last_err}", None

def save_res(res_list, path):
    # dedupe by question_id keeping last
    seen = {}
    for item in res_list:
        seen[item["question_id"]] = item
    with open(path, "w") as f:
        json.dump(list(seen.values()), f)

def save_summary(category_record, path):
    total_corr, total_wrong = 0.0, 0.0
    out = {}
    for k, v in category_record.items():
        corr, wrong = v["corr"], v["wrong"]
        acc = corr / (corr + wrong) if (corr + wrong) > 0 else 0.0
        out[k] = {"corr": corr, "wrong": wrong, "acc": acc}
        total_corr += corr
        total_wrong += wrong
    total_acc = total_corr / (total_corr + total_wrong) if (total_corr + total_wrong) > 0 else 0.0
    out["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": total_acc}
    with open(path, "w") as f:
        json.dump(out, f)

def evaluate_all(args):
    # Client
    client = UniversalClient(
        base_url=args.vllm_base_url,
        model=args.model_alias,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        backend=args.backend
    )

    # Data
    test_by_cat, val_by_cat = load_mmlu_pro()
    subjects = list(test_by_cat.keys()) if args.assigned_subjects == ["all"] else args.assigned_subjects

    os.makedirs(args.output_dir, exist_ok=True)

    # State
    results_all = []  # list of dict rows (for JSON)
    category_record = defaultdict(lambda: {"corr": 0.0, "wrong": 0.0})
    progress_total = 0

    for subject in subjects:
        test_data = test_by_cat[subject]
        cot_examples = val_by_cat.get(subject, [])  # dev set few-shot
        out_res_path = os.path.join(args.output_dir, f"{subject}_result.json")
        out_sum_path = os.path.join(args.output_dir, f"{subject}_summary.json")

        # Load existing (resume)
        if os.path.exists(out_res_path):
            try:
                with open(out_res_path, "r") as f:
                    prev = json.load(f)
                # incorporate existing into results_all and stats
                for row in prev:
                    results_all.append(row)
                    if "pred" in row and row["pred"] is not None:
                        if row["pred"] == row["answer"]:
                            category_record[subject]["corr"] += 1
                        else:
                            category_record[subject]["wrong"] += 1
                # filter out already answered questions from run list
                answered_ids = {r["question_id"] for r in prev}
                test_data = [q for q in test_data if q["question_id"] not in answered_ids]
            except Exception:
                pass

        if not test_data:
            # nothing new to do for this subject
            save_summary(category_record, out_sum_path)
            continue

        # Parallel loop
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = []
            for q in test_data:
                fut = ex.submit(
                    worker_one_question,
                    client,
                    subject,
                    q,
                    cot_examples,
                    args.max_cot_examples,
                    args.retries,
                    args.backoff,
                )
                futs.append((fut, q))

            for i, (fut, q) in enumerate(tqdm(futs, desc=f"{subject}", unit="q")):
                qid, pred, raw, dt = fut.result()
                row = dict(q)
                row["pred"] = pred
                row["model_outputs"] = raw
                results_all.append(row)

                # update stats
                if pred is not None:
                    if pred == q["answer"]:
                        category_record[subject]["corr"] += 1
                    else:
                        category_record[subject]["wrong"] += 1
                else:
                    category_record[subject]["wrong"] += 1

                progress_total += 1
                # checkpoint occasionally
                if progress_total % args.save_every == 0:
                    save_res(results_all, out_res_path)
                    save_summary(category_record, out_sum_path)

        # final save for this subject
        save_res(results_all, out_res_path)
        save_summary(category_record, out_sum_path)

    # final global summary (optional)
    # We already saved per-subject summaries; nothing else needed.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", "-o", type=str, default="eval_results")
    p.add_argument("--assigned_subjects", "-a", type=str, default="all",
                   help='Comma-separated subjects or "all"')
    # vLLM connection
    p.add_argument("--vllm_base_url", type=str, default="http://0.0.0.0:8000/v1",
                   help="Base URL of your vLLM server (with /v1)")
    p.add_argument("--model_alias", type=str, default="openai/gpt-oss-120b",
                   help="The model name that vLLM was launched with")
    # performance knobs
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max_cot_examples", type=int, default=1,
                   help="Few-shot CoT examples per subject (lower = faster)")
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--backoff", type=float, default=1.5)
    p.add_argument("--backend", type=str, default="auto",
               choices=["auto","openai","ollama"],
               help="API style: vLLM/OpenAI or Ollama native")
    args = p.parse_args()

    if args.assigned_subjects.lower() == "all":
        args.assigned_subjects = ["all"]
    else:
        args.assigned_subjects = [s.strip() for s in args.assigned_subjects.split(",") if s.strip()]
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate_all(args)
