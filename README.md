# MMLU‑Pro

|[**🤗 Dataset**](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | [**🏆 Leaderboard**](https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro) | [**📖 Paper**](https://arxiv.org/abs/2406.01574) |
|---|---|---|

This repo contains the evaluation code for the NeurIPS‑2024 paper **“MMLU‑Pro: A More Robust and Challenging Multi‑Task Language Understanding Benchmark”**.  

## Introduction

We introduce **MMLU‑Pro**, an enhanced benchmark designed to evaluate language‑understanding models across broader and more challenging tasks. Building on the Massive Multitask Language Understanding (MMLU) dataset, MMLU‑Pro integrates more reasoning‑focused questions and expands the answer choices per question from **4 → 10**, significantly raising the difficulty and reducing the chance of success through random guessing.

MMLU‑Pro comprises **≈12 k** rigorously curated questions from academic exams and textbooks, spanning **14** diverse domains (Biology, Business, Chemistry, …, Others).  

Our experiments show a **16 %–33 %** drop in accuracy compared to the original MMLU and a **≈2 %** reduction in prompt‑sensitivity (vs. 4‑5 % on MMLU). Moreover, Chain‑of‑Thought (CoT) reasoning now *helps* rather than hurts, confirming that the benchmark contains substantially more complex reasoning questions.

<img width="1432" alt="MMLU‑Pro overview" src="https://github.com/TIGER-AI-Lab/MMLU-Pro/assets/20929360/8e369fc2-5b6b-4bab-8a44-9e222e742027">

---

## Updates
- **Oct 10 2024** – Added the 24 tested prompt styles from the paper.
- **Apr 2025** – New **local inference script** for vLLM & Ollama (`vllm_ollama_eval.py`).  

---

## Dataset Creation

MMLU‑Pro was created to provide language models with a more challenging and robust benchmark, pushing the boundaries of expert‑level knowledge and reasoning. See the Hugging Face hub for details: https://huggingface.co/datasets/TIGER-Lab/MMLU‑Pro  

---

## Evaluation

### 1️⃣ Local inference with **vLLM** or **Ollama**

We now ship a ready‑to‑run Python script that can talk to any OpenAI‑compatible server (vLLM, vLLM‑served on `/v1`, or an Ollama instance).  
The script lives at the repository root:



#### What it does
* Loads the MMLU‑Pro test‑set (and the validation set for few‑shot CoT examples).  
* Builds prompts in the “think step‑by‑step → answer” style used in the paper.  
* Sends the prompts to the configured backend (vLLM or Ollama) using the appropriate endpoint (`/chat/completions`, `/completions`, or `/api/chat`).  
* Extracts the answer (A‑J) with a set of robust regexes, records latency, and writes **per‑subject JSON results** plus a **summary JSON**.  
* Supports multi‑processing via `ThreadPoolExecutor` (default 8 workers).

#### Quick start

```bash
# 1️⃣ Clone & cd (you already did this)
git clone https://github.com/yash-vardhansingh/MMLU-PRO.git
cd MMLU-PRO

# 2️⃣ (Optional) Create a clean Python env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # includes requests, tqdm, datasets, etc.

# 3️⃣ Run the script
python evaluate_master.py \
    --output_dir eval_results \
    --vllm_base_url http://127.0.0.1:8000/v1 \   # <-- your vLLM server
    --model_alias llama3:8b \                  # name known to the server
    --backend auto \                           # auto‑detect (vLLM vs Ollama)
    --max_cot_examples 1 \                     # how many CoT demos per subject
    --concurrency 8 \                          # parallel workers
    --temperature 0.0 \                        # deterministic
    --max_tokens 512
