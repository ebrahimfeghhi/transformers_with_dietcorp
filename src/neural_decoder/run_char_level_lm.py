from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.submodules.ctc_batched_beam_decoding import BatchedBeamCTCComputer
from inference_functions import load_bit_phoneme_model, evaluate_model
from dataset import getDatasetLoaders
import torch.nn.functional as F
import numpy as np
import torch

language_model_path = "/workspace/transformers_with_dietcorp/lm/char_6gram_lm.nemo"
vocab_size = 26
lm = NGramGPULanguageModel.from_nemo(
    lm_path=language_model_path,
    vocab_size=vocab_size
)

device = 'cuda'
bit_phoneme_filepath = "/data/models/time_masked_transfomer_characters_80ms_seed_0/"
model, args = load_bit_phoneme_model(bit_phoneme_filepath)
model = model.to(device)

data_file = '/data/neural_data/ptDecoder_ctc_both_char'
trainLoaders, testLoaders, loadedData = getDatasetLoaders(
        data_file, 8, None, 
        False
    )

outputs, cer, per_day_cer = evaluate_model(model, loadedData, args, partition='test', device='cuda')

num_classes = 30
logits = np.zeros((len(outputs['logits']), max(outputs['logitLengths']), num_classes))
for idx, l in enumerate(outputs['logits']):
    l_length = outputs['logitLengths'][idx]
    logits[idx, :l_length, :] = l
    
logits_torch = torch.from_numpy(logits)[:, :, :27]
log_probs = F.log_softmax(logits_torch, dim=-1)
# Build an index list for reordering the last dimension
# We want [1,2,...,26,0] instead of [0,1,...,26]
perm = torch.arange(1, log_probs.shape[-1], device=log_probs.device)   # [1,2,...,26]
perm = torch.cat([perm, torch.tensor([0], device=log_probs.device)])  # append 0 at end
log_probs = log_probs.index_select(-1, perm).contiguous()

log_probs_length = torch.from_numpy(np.array(outputs['logitLengths']))

decoder = BatchedBeamCTCComputer(blank_index=26, beam_size=16, return_best_hypothesis=False, fusion_models=[lm], 
                                 fusion_models_alpha=[0.25])

transcripts = decoder.batched_beam_search_torch(log_probs, log_probs_length)