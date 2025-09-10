from nemo.collections.asr.parts.submodules.wfst_decoder import RivaGpuWfstDecoder

from inference_funcs import load_bit_phoneme_model, evaluate_model
from dataset import getDatasetLoaders
import numpy as np
import torch
import torch.nn.functional as F

langugage_model_fst_path = "/data/code/nejm-brain-to-text/language_model/pretrained_language_models/openwebtext_1gram_lm_sil/TLG_with_symbols.fst"

decoder = RivaGpuWfstDecoder(lm_fst=langugage_model_fst_path, decoding_mode="nbest", 
                             beam_size=16, lm_weight=0.2, nbest_size=10)


device = 'cuda'

bit_phoneme_filepath = "/data/models/transformer_short_training_fixed_seed_0/"
model, args = load_bit_phoneme_model(bit_phoneme_filepath)
model = model.to(device)

data_file = '/data/neural_data/ptDecoder_ctc_both'
trainLoaders, testLoaders, loadedData = getDatasetLoaders(
        data_file, 8, None, 
        False
    )

outputs, cer, per_day_cer = evaluate_model(model, loadedData, args, partition='test', device='cuda', verbose=False)

num_classes = 41
logits = np.zeros((len(outputs['logits']), max(outputs['logitLengths']), num_classes))
for idx, l in enumerate(outputs['logits']):
    l_length = outputs['logitLengths'][idx]
    logits[idx, :l_length, :] = l
    
logits_torch = torch.from_numpy(logits)
log_probs = F.log_softmax(logits_torch, dim=-1).to(dtype=torch.float32, device=device)
log_probs_length = torch.from_numpy(np.array(outputs['logitLengths'])).to(dtype=torch.int64, device='cpu')
hypotheses = decoder._decode_nbest(log_probs, log_probs_length)

