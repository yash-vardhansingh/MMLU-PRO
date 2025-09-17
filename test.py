from datasets import load_dataset
from collections import defaultdict

def preprocess_split(hf_split):
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

ds = load_dataset("TIGER-Lab/MMLU-Pro")
test_by_cat = preprocess_split(ds["test"])

print("Available categories:")
for c in sorted(test_by_cat.keys()):
    print("-", c)