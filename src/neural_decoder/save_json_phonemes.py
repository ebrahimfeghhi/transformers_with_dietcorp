#!/usr/bin/env python3
"""
Writes JSONL where each line is:
{"text": "<|start_header_id|>user<|end_header_id|>\n\n<SYSTEM_PROMPT>\n\nCandidates (phoneme representations):\n1) ...\n...\n\nCandidates (transcriptions):\n1) ...\n...\n\n<|start_header_id|>assistant<|end_header_id|>\n<GT phoneme>\n<GT transcription><EOS_TOKEN_if_set>"}

- Phoneme candidates come first in the prompt.
- Assistant block = GT phoneme (line 1) + GT transcription (line 2) when EVAL_MODE=False.
- When EVAL_MODE=True, assistant block is just the assistant header (model will fill in).
"""

# ──────────────────────────────────────────────────────────────────
# 0) CONFIG
# ──────────────────────────────────────────────────────────────────
EVAL_MODE = False  # ⇦ set True for eval/inference JSONL (no GT content)
INCLUDE_PHONEMES = True
EOS_TOKEN = ""     # e.g., "<|eot_id|>" or "<EOT>"; keep "" if you don't want one
DIR_OPTIONS = [
    "/home/ubuntu/data/model_transcriptions/txt_files/",
    "/home/ubuntu/data/model_transcriptions_finetune/txt_files/",
    "/home/ubuntu/data/model_transcriptions_comp/text_files/",
]
DIR = DIR_OPTIONS[1]
FILENAME = "train_with_phonemes"
OUT_JSONL = f"/home/ubuntu/transformers_with_dietcorp/src/neural_decoder/jsonl_files/{FILENAME}.jsonl"

# File patterns (tweak if your names differ)
SENT_PATTERN = "transformer_short_training_fixed_seed_*_llm_outs.txt"
PHON_PATTERN_CANDIDATES = "decoded_phonemes_seed_*.txt"

# ──────────────────────────────────────────────────────────────────
# 1) Imports & paths
# ──────────────────────────────────────────────────────────────────
import json
import re
from pathlib import Path

DIR = Path(DIR)

GT_SENT_FILE = DIR / "ground_truth_sentences.txt"
GT_PHON_FILE = DIR / "ground_truth_phonemes.txt"

DEC_SENT_FILES = sorted(DIR.glob(SENT_PATTERN))
DEC_PHON_FILES = sorted(DIR.glob(PHON_PATTERN_CANDIDATES))

def _seed_idx(path: Path):
    m = re.search(r"fixed_seed_(\d+)", path.name)
    return int(m.group(1)) if m else -1

DEC_SENT_FILES = sorted(DEC_SENT_FILES, key=_seed_idx)
DEC_PHON_FILES  = sorted(DEC_PHON_FILES,  key=_seed_idx)

assert DEC_SENT_FILES, "No candidate transcription files found."
if INCLUDE_PHONEMES:
    assert DEC_PHON_FILES, "INCLUDE_PHONEMES=True but no candidate phoneme files found."

# ──────────────────────────────────────────────────────────────────
# 2) SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Your task is to perform automatic speech recognition. Below are multiple candidate transcriptions together with their "
    "corresponding phoneme representations. The phonemes are taken from the CMU Pronouncing Dictionary. The special symbol SIL "
    "represents the start of the sentence, or the end of the sentence, or the space between two adjacent words. Based on the "
    "transcription candidates and their phoneme representations, come up with a transcription and its corresponding phoneme "
    "representation that are most accurate, ensuring the transcription is contextually and grammatically correct. Focus on key "
    "differences in the candidates that change the meaning or correctness. Avoid selections with repetitive or nonsensical phrases. "
    "In cases of ambiguity, select the option that is most coherent and contextually sound, taking clues from the phoneme representations. "
    "The candidate phoneme representations may not always be the correct representation of the corresponding candidate transcriptions. "
    "Some phonemes in the candidate phoneme sequences might have been incorrectly added, removed, or replaced. However, the candidate "
    "phonemes contain useful information that will help you come up with the correct transcription and phoneme representation. You should "
    "translate each subgroup of phonemes that is enclosed by two SIL symbols into one single word. You should remove SIL symbols at the "
    "start or the end of the phoneme sequence. Respond with your refined transcription and its corresponding phoneme representation only, "
    "without any introductory text."
)

# ──────────────────────────────────────────────────────────────────
# 3) Load data
# ──────────────────────────────────────────────────────────────────
def load_lists(paths):
    return [Path(p).read_text(encoding="utf-8").splitlines() for p in paths]

dec_sent_lists = load_lists(DEC_SENT_FILES)
K = len(dec_sent_lists)
N = len(dec_sent_lists[0])
assert all(len(lst) == N for lst in dec_sent_lists), "Decoded sentence file lengths differ!"

if INCLUDE_PHONEMES:
    assert len(DEC_PHON_FILES) == K, "Mismatch: #phoneme files != #sentence files"
    dec_phon_lists = load_lists(DEC_PHON_FILES)
    assert all(len(lst) == N for lst in dec_phon_lists), "Decoded phoneme file lengths differ!"

if not EVAL_MODE:
    gt_sent_lines = Path(GT_SENT_FILE).read_text(encoding="utf-8").splitlines()
    gt_phon_lines = Path(GT_PHON_FILE).read_text(encoding="utf-8").splitlines()
    assert len(gt_sent_lines) == N, "Ground-truth sentence length mismatch!"
    assert len(gt_phon_lines) == N, "Ground-truth phoneme length mismatch!"

# ──────────────────────────────────────────────────────────────────
# 4) Write JSONL (single 'text' field)
# ──────────────────────────────────────────────────────────────────
with Path(OUT_JSONL).open("w", encoding="utf-8") as fout:
    for idx in range(N):
        cand_phons = "\n".join(f"{i+1}) {dec_phon_lists[i][idx]}" for i in range(K)) if INCLUDE_PHONEMES else ""
        cand_trans = "\n".join(f"{i+1}) {dec_sent_lists[i][idx]}" for i in range(K))

        user_block = (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}\n\n"
            "Candidates (phoneme representations):\n"
            f"{cand_phons}\n\n"
            "Candidates (transcriptions):\n"
            f"{cand_trans}\n"
        )

        if EVAL_MODE:
            assistant_block = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            tail = ""  # no EOS for eval unless you really want it
        else:
            assistant_block = (
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{gt_phon_lines[idx].strip()}\n"
                f"{gt_sent_lines[idx].strip()}"
            )
            tail = EOS_TOKEN

        record_text = f"{user_block}\n{assistant_block}{tail}"
        fout.write(json.dumps({"text": record_text}, ensure_ascii=False) + "\n")

print(f"✅ Wrote {N} examples to {Path(OUT_JSONL).resolve()}")
