#!/usr/bin/env python3
"""
build_jsonl_v3.py – writes each example as a single JSON object with keys:

{
  "prompt":    [{"content": <instruction + candidates>, "role": "user"}],
  "completion":[{"content": <ground-truth sentence or "">, "role": "assistant"}]
}

Flip `EVAL_MODE`:
  • False → training/finetune file (completion = ground truth)
  • True  → eval/inference file   (completion = "")
"""

# ──────────────────────────────────────────────────────────────────
# 0) ONE-LINE CONFIG
# ──────────────────────────────────────────────────────────────────
EVAL_MODE = True     # ⇦ set True when you build the eval.jsonl
INCLUDE_PHONEMES = True 
DIR_OPTIONS = ["/home/ubuntu/data/model_transcriptions/txt_files/", "/home/ubuntu/data/model_transcriptions_finetune/txt_files/", 
               "/home/ubuntu/data/model_transcriptions_comp/text_files/"]
DIR = DIR_OPTIONS[0]
FILENAME = 'val_no_gt_with_phonemes'
OUT_JSONL = f"/home/ubuntu/transformers_with_dietcorp/src/neural_decoder/jsonl_files/{FILENAME}.jsonl"

# ──────────────────────────────────────────────────────────────────
# 1) Imports & paths
# ──────────────────────────────────────────────────────────────────
import json
from pathlib import Path

DIR = Path("/home/ubuntu/data/model_transcriptions/txt_files/")

GT_SENT_FILE = DIR / "ground_truth_sentences.txt"
PH_SENT_FILE = DIR / "ground_truth_phonemes.txt"

# 10 candidate transcription files (adjust if needed)
DEC_SENT_FILES = [DIR / f"transformer_short_training_fixed_seed_{i}_llm_outs.txt"
                  for i in range(10)]

# ──────────────────────────────────────────────────────────────────
# 2) Constant instruction (no phonemes)
# ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Your task is to perform automatic speech recognition error correction. "
    "Below are multiple candidate transcriptions of the same utterance. "
    "These candidates were decoded from neural activity and may contain errors. "
    "Based on the candidates, produce the single most accurate, coherent, and grammatical "
    "transcription of the original utterance. Focus on key differences between candidates "
    "that change meaning or correctness, and avoid repetitive or nonsensical phrases. "
    "Respond with only the final corrected transcription—no explanations or extra text."
)

# ──────────────────────────────────────────────────────────────────
# 3) Helpers
# ──────────────────────────────────────────────────────────────────
def load_lists(paths):
    return [Path(p).read_text(encoding="utf-8").splitlines() for p in paths]

dec_sent_lists = load_lists(DEC_SENT_FILES)
K = len(dec_sent_lists)                   # number of candidate files (expected 10)
N = len(dec_sent_lists[0])                # number of examples

# Sanity: all candidate files must have same length
assert all(len(lst) == N for lst in dec_sent_lists), "Decoded file lengths differ!"

# Load ground truth if training mode
if not EVAL_MODE:
    gt_sent_lines = Path(GT_SENT_FILE).read_text(encoding="utf-8").splitlines()
    assert len(gt_sent_lines) == N, "Ground-truth length mismatch!"

# ──────────────────────────────────────────────────────────────────
# 4) Write JSONL in prompt/completion format
# ──────────────────────────────────────────────────────────────────
with Path(OUT_JSONL).open("w", encoding="utf-8") as fout:
    for idx in range(N):
        # Build candidates text block
        cand_trans = "\n".join(
            f"{i+1}) {dec_sent_lists[i][idx]}"
            for i in range(K)
        )

        # Prompt content = instruction + candidates
        prompt_content = f"{SYSTEM_PROMPT}\n\nCandidates:\n{cand_trans}"

        # Completion content = ground-truth sentence ONLY (or empty in eval)
        completion_content = "" if EVAL_MODE else gt_sent_lines[idx].strip()

        obj = {
            "prompt": [
                {
                    "content": prompt_content,
                    "role": "user"
                }
            ],
            "completion": [
                {
                    "content": completion_content,
                    "role": "assistant"
                }
            ]
        }

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅  Wrote {N} examples to {Path(OUT_JSONL).resolve()}")
