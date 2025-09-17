#!/usr/bin/env python3
import os, json, re, time, random, argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import requests
from tqdm import tqdm
from datasets import load_dataset

random.seed(12345)

CHOICE_MAP = "ABCDEFGHIJ"

# --------- Universal Client ---------
class UniversalClient:
    def __init__(self, base_url, model, max_tokens=1024, temperature=0.0,
                 timeout=300, backend="auto"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.session = requests.Session()

        if backend == "auto":
            if self.base_url.endswith("/v1") or "/v1" in self.base_url:
                self.backend = "openai"
            elif "/api/" in self.base_url:
                self.backend = "ollama"
            else:
                self.backend = "openai"
        else:
            self.backend = backend

        print(f"[INFO] Client initialized | backend={self.backend}, url={self.base_url}, model={self.model}")

    def chat(self, prompt, qid=None, category=None):
        print(f"[DEBUG] Sending request | subj={category}, qid={qid}, prompt_len={len(prompt)}")

        if self.backend == "openai":
            # Try chat/completions first (OpenAI-style). If the server rejects
            # chat because of the missing chat-template (Transformers >=4.44),
            # automatically fall back to the completions endpoint which works
            # with your model files.
            chat_url = f"{self.base_url}/chat/completions"
            chat_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

            try:
                resp = self.session.post(chat_url, json=chat_payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    # Chat-style response (if available)
                    content = data["choices"][0]["message"]["content"]
                else:
                    # Non-200 — raise to go to fallback handler below
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:600]}")
            except Exception as e:
                # Detect the specific "no chat template" error message (from your curl output).
                err_text = str(e)
                # Also try to inspect resp.text if available (for requests exceptions this may not exist)
                try:
                    resp_text = resp.text if 'resp' in locals() and resp is not None else ""
                except Exception:
                    resp_text = ""

                # If server complains about chat template (Transformers >=4.44), or any other
                # failure, fall back to completions endpoint which you confirmed works.
                if "chat template" in resp_text.lower() or "chat template" in err_text.lower() or True:
                    # Use completions endpoint as fallback
                    comp_url = f"{self.base_url}/completions"
                    comp_payload = {
                        "model": self.model,
                        "prompt": prompt,
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                        # Stopping so model doesn't generate repeated QA turns
                        "stop": ["\nUser:", "\nAssistant:"],
                    }
                    creq = self.session.post(comp_url, json=comp_payload, timeout=self.timeout)
                    if creq.status_code != 200:
                        print(f"[ERROR] Fallback completions failed {creq.status_code}: {creq.text[:400]}")
                        raise RuntimeError(f"HTTP {creq.status_code}: {creq.text[:400]}")
                    cdata = creq.json()
                    # vLLM completions returns choices[0].text
                    content = cdata["choices"][0].get("text") or cdata["choices"][0].get("message", {}).get("content", "")
                else:
                    # Re-raise original error if it wasn't the chat-template case (should be rare)
                    raise



        # if self.backend == "openai":
        #     # vLLM in your setup requires using the completions endpoint (no chat template).
        #     url = f"{self.base_url}/completions"
        #     payload = {
        #         "model": self.model,
        #         "prompt": prompt,
        #         "max_tokens": self.max_tokens,
        #         "temperature": self.temperature,
        #         # stop at next User/Assistant label so model doesn't keep continuing turns
        #         "stop": ["\nUser:", "\nAssistant:"],
        #     }
        #     resp = self.session.post(url, json=payload, timeout=self.timeout)

        #     if resp.status_code != 200:
        #         print(f"[ERROR] OpenAI-style completions failed {resp.status_code}: {resp.text[:400]}")
        #         raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
        #     data = resp.json()
        #     content = data["choices"][0].get("text") or data["choices"][0].get("message", {}).get("content", "")

        # if self.backend == "openai":
        #     url = f"{self.base_url}/chat/completions"
        #     payload = {
        #         "model": self.model,
        #         "messages": [{"role": "user", "content": prompt}],
        #         "max_tokens": self.max_tokens,
        #         "temperature": self.temperature,
        #     }
        #     resp = self.session.post(url, json=payload, timeout=self.timeout)

        #     if resp.status_code != 200:
        #         print(f"[ERROR] OpenAI backend failed {resp.status_code}: {resp.text[:300]}")
        #         raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        #     data = resp.json()
        #     content = data["choices"][0]["message"]["content"]

        elif self.backend == "ollama":
            # prefer /api/chat
            if self.base_url.endswith("/api/chat"):
                url = self.base_url
                use_chat = True
            elif self.base_url.endswith("/api/generate"):
                url = self.base_url
                use_chat = False
            else:
                url = f"{self.base_url}/api/chat"
                use_chat = True

            if use_chat:
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": self.temperature, "num_predict": self.max_tokens}
                }
                resp = self.session.post(url, json=payload, timeout=self.timeout)
                if resp.status_code != 200:
                    print(f"[ERROR] Ollama chat failed {resp.status_code}: {resp.text[:300]}")
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                content = data["message"]["content"]
            else:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.temperature, "num_predict": self.max_tokens}
                }
                resp = self.session.post(url, json=payload, timeout=self.timeout)
                if resp.status_code != 200:
                    print(f"[ERROR] Ollama generate failed {resp.status_code}: {resp.text[:300]}")
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                content = data["response"]

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        print(f"[DEBUG] Got response | subj={category}, qid={qid}, resp_len={len(content)}")
        return content

# --------- Helpers ---------
def format_example(question, options, cot_content=""):
    if not cot_content: cot_content = "Let's think step by step."
    if cot_content.startswith("A: "): cot_content = cot_content[3:]
    example = f"Question: {question}\nOptions: "
    for i, opt in enumerate(options):
        example += f"{CHOICE_MAP[i]}. {opt}\n"
    return example + "Answer: " + cot_content + "\n\n"

def format_query_only(question, options):
    example = f"Question: {question}\nOptions: "
    for i, opt in enumerate(options): example += f"{CHOICE_MAP[i]}. {opt}\n"
    return example + "Answer: "

ANSWER_PATTERNS = [
    re.compile(r"answer is\s*\(?([A-J])\)?", re.IGNORECASE),
    re.compile(r".*[aA]nswer:\s*([A-J])"),
    re.compile(r"\b([A-J])\b(?!.*\b[A-J]\b)", re.DOTALL),
]
def extract_answer(text):
    if not text: return None
    for pat in ANSWER_PATTERNS:
        m = pat.search(text)
        if m: return m.group(1).upper()
    return None

def preprocess_split(hf_split):
    cleaned = []
    for row in hf_split:
        opts = [o for o in row["options"] if o != "N/A"]
        row = dict(row); row["options"] = opts
        cleaned.append(row)
    by_cat = defaultdict(list)
    for row in cleaned: by_cat[row["category"]].append(row)
    return dict(by_cat)

def load_mmlu_pro():
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    return preprocess_split(ds["test"]), preprocess_split(ds["validation"])

def build_prompt(category, cot_examples, q, max_cot_examples):
    prompt = (f"The following are multiple choice questions (with answers) about {category}. "
              f"Think step by step and then output the answer in the format of \"The answer is (X)\".\n\n")
    for ex in cot_examples[:max_cot_examples]:
        prompt += format_example(ex["question"], ex["options"], ex.get("cot_content", ""))
    prompt += format_query_only(q["question"], q["options"])
    return prompt

# --------- Worker ---------
def worker_one_question(client, category, q, cot_examples, max_cot_examples, retries=3, backoff=1.5):
    prompt = build_prompt(category, cot_examples, q, max_cot_examples)
    last_err = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = client.chat(prompt, qid=q["question_id"], category=category)
            resp = resp.replace("**", "")
            pred = extract_answer(resp)
            print(f"[INFO] Finished qid={q['question_id']} | subj={category} | pred={pred} | time={time.time()-t0:.2f}s")
            return q["question_id"], pred, resp, (time.time() - t0)
        except Exception as e:
            last_err = e
            print(f"[WARN] Retry {attempt+1}/{retries} for qid={q['question_id']} | subj={category} | error={e}")
            time.sleep(backoff ** attempt)
    return q["question_id"], None, f"[ERROR] {last_err}", None

# --------- Save ---------
def save_res(res_list, path):
    seen = {item["question_id"]: item for item in res_list}
    with open(path, "w") as f: json.dump(list(seen.values()), f)
    print(f"[INFO] Saved results -> {path} ({len(seen)} records)")

def save_summary(category_record, path):
    total_corr, total_wrong = 0.0, 0.0
    out = {}
    for k, v in category_record.items():
        corr, wrong = v["corr"], v["wrong"]
        acc = corr / (corr + wrong) if (corr + wrong) > 0 else 0.0
        out[k] = {"corr": corr, "wrong": wrong, "acc": acc}
        total_corr += corr; total_wrong += wrong
    total_acc = total_corr / (total_corr + total_wrong) if (total_corr + total_wrong) > 0 else 0.0
    out["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": total_acc}
    with open(path, "w") as f: json.dump(out, f)
    print(f"[INFO] Saved summary -> {path}")

# --------- Evaluation ---------
def evaluate_all(args):
    client = UniversalClient(args.vllm_base_url, args.model_alias,
                             args.max_tokens, args.temperature,
                             args.timeout, args.backend)
    test_by_cat, val_by_cat = load_mmlu_pro()
    subjects = list(test_by_cat.keys()) if args.assigned_subjects == ["all"] else args.assigned_subjects
    os.makedirs(args.output_dir, exist_ok=True)

    results_all = []
    category_record = defaultdict(lambda: {"corr": 0.0, "wrong": 0.0})
    progress_total = 0

    for subject in subjects:
        test_data = test_by_cat[subject]
        cot_examples = val_by_cat.get(subject, [])
        out_res_path = os.path.join(args.output_dir, f"{subject}_result.json")
        out_sum_path = os.path.join(args.output_dir, f"{subject}_summary.json")

        print(f"[INFO] Starting subject {subject} | #Q={len(test_data)}")

        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(worker_one_question, client, subject, q, cot_examples, args.max_cot_examples,
                              args.retries, args.backoff) for q in test_data]

            for fut, q in zip(tqdm(futs, desc=f"{subject}", unit="q"), test_data):
                qid, pred, raw, dt = fut.result()
                row = dict(q); row["pred"] = pred; row["model_outputs"] = raw
                results_all.append(row)
                if pred == q["answer"]: category_record[subject]["corr"] += 1
                else: category_record[subject]["wrong"] += 1
                progress_total += 1

                if progress_total % args.save_every == 0:
                    save_res(results_all, out_res_path)
                    save_summary(category_record, out_sum_path)

        save_res(results_all, out_res_path)
        save_summary(category_record, out_sum_path)
        print(f"[INFO] Finished subject {subject}")

# --------- CLI ---------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir","-o",type=str,default="eval_results")
    p.add_argument("--assigned_subjects","-a",type=str,default="all")
    p.add_argument("--vllm_base_url",type=str,default="http://0.0.0.0:8000/v1")
    p.add_argument("--model_alias",type=str,default="llama3:8b")
    p.add_argument("--concurrency",type=int,default=8)
    p.add_argument("--max_cot_examples",type=int,default=1)
    p.add_argument("--save_every",type=int,default=100)
    p.add_argument("--max_tokens",type=int,default=512)
    p.add_argument("--temperature",type=float,default=0.0)
    p.add_argument("--timeout",type=int,default=300)
    p.add_argument("--retries",type=int,default=3)
    p.add_argument("--backoff",type=float,default=1.5)
    p.add_argument("--backend",type=str,default="auto",choices=["auto","openai","ollama"])
    args=p.parse_args()
    if args.assigned_subjects.lower()=="all": args.assigned_subjects=["all"]
    else: args.assigned_subjects=[s.strip() for s in args.assigned_subjects.split(",")]
    return args

if __name__=="__main__":
    args=parse_args()
    evaluate_all(args)
