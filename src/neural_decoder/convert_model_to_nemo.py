from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel

lm = NGramGPULanguageModel.from_arpa(
    lm_path="/workspace/emg2qwerty/models/lm/wikitext-103-6gram-charlm.arpa",
    vocab_size=26, 
    normalize_unk=True
)

#lm.save_to("/workspace/transformers_with_dietcorp/lm/char_6gram_lm.nemo")

vocab = [chr(i) for i in range(ord('a'), ord('z') + 1)] 
V = len(vocab)

#from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig

#dec_cfg = CTCDecodingConfig(strategy="beam")
#dec_cfg.beam.beam_size = 16
#dec_cfg.beam.ngram_lm_model = "/workspace/transformers_with_dietcorp/lm/char_6gram_lm.nemo"   # or the .arpa
#dec_cfg.beam.ngram_lm_alpha = 0.25
#dec_cfg.beam.allow_cuda_graphs = True

#decoder = CTCDecoding(decoding_cfg=dec_cfg, vocabulary=vocab)


from nemo.collections.asr.parts.submodules.ctc_batched_beam_decoding import BatchedBeamCTCComputer


decoder = BatchedBeamCTCComputer(blank_index=26, beam_size=16, return_best_hypothesis=False, fusion_models=[lm], 
                                 fusion_models_alpha=[0.25])


import torch
import torch.nn.functional as F

def make_toy_ctc_batch(target_texts, vocab):
    """
    target_texts: list of strings, e.g. ["hello", "abc"]
    vocab: list of tokens (no CTC blank), e.g. ['a', ..., 'z']

    returns:
      log_probs: (B, T, V+1)  log-softmax over last dim
      log_probs_length: (B,)  valid timesteps for each item
      blank_idx: int (V)
    """
    B = len(target_texts)
    V = len(vocab)
    blank_idx = V

    # char -> id
    c2i = {c: i for i, c in enumerate(vocab)}

    # choose a common T big enough for all sequences: about 2*len + 1
    T = max(2 * len(t) + 1 for t in target_texts)
    # add a little pad slack
    T = T + 1

    # start with low logits everywhere
    logits = torch.full((B, T, V + 1), -6.0)
    lengths = []

    for b, text in enumerate(target_texts):
        t = 0

        # initial blank frame
        logits[b, t, blank_idx] = 8.0
        t += 1

        # for each char: emit char, then a blank (so repeats are legal in CTC)
        for ch in text:
            assert ch in c2i, f"char {ch!r} not in vocab"
            logits[b, t, c2i[ch]] = 8.0; t += 1
            logits[b, t, blank_idx] = 8.0; t += 1

        # record valid length
        lengths.append(t)

        # pad remaining frames with strong blank so they don't affect decoding
        if t < T:
            logits[b, t:, blank_idx] = 8.0

    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_length = torch.tensor(lengths, dtype=torch.int32)
    return log_probs, log_probs_length, blank_idx

# --- example usage ---
vocab = [chr(i) for i in range(ord('a'), ord('z') + 1)]  # 26 lowercase letters
texts = ["hello", "abc"]

log_probs, log_probs_length, blank_idx = make_toy_ctc_batch(texts, vocab)

print("log_probs shape:", tuple(log_probs.shape))         # (B, T, V+1)
print("log_probs_length:", log_probs_length.tolist())     # valid lengths
print("blank_idx:", blank_idx)

transcripts = decoder.batched_beam_search_torch(log_probs, log_probs_length)

breakpoint()
