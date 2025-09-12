from nemo.collections.asr.parts.submodules.wfst_decoder import RivaGpuWfstDecoder

from inference_funcs import load_bit_phoneme_model, evaluate_model
from dataset import getDatasetLoaders
import numpy as np
import torch
import torch.nn.functional as F

language_model_fst_path = "/data/code/nejm-brain-to-text/language_model/pretrained_language_models/openwebtext_1gram_lm_sil/TLG_with_symbols.fst"
language_model_path_3g = '/data/lm/TLG_opt_with_symbols.fst'

max_mem = 400000000
blank_penalty = 0.7
lm_weight = 1.25
beam_size = 18
nbest_size = 18

decoder = RivaGpuWfstDecoder(lm_fst=language_model_path_3g, decoding_mode="nbest", 
                             beam_size=beam_size, lm_weight=lm_weight,
                             nbest_size=nbest_size, max_mem=max_mem, blank_penalty=blank_penalty)


load_predictions = True

device = 'cuda'

bit_phoneme_filepath = "/data/models/transformer_short_training_fixed_seed_0/"

if load_predictions:
    
    log_probs_arranged = torch.load(f"{bit_phoneme_filepath}log_probs_arranged.pth").to(dtype=torch.float32, device=device)
    log_probs_length = torch.load(f"{bit_phoneme_filepath}log_probs_length.pth").to(dtype=torch.int64, device='cpu')

else: 
    
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
    
    log_probs_blank_last = torch.concat((log_probs[:, :, 1:], log_probs[:, :, 0:1]), dim=-1) # move blank to end
    log_probs_arranged = torch.concat((log_probs_blank_last[:, :, -1:], log_probs_blank_last[:, :, -2:-1], log_probs_blank_last[:, :, :-2]), dim=-1)
    
    torch.save(log_probs_arranged, f"{bit_phoneme_filepath}log_probs_arranged.pth")
    torch.save(log_probs_length, f"{bit_phoneme_filepath}log_probs_length.pth")
    
    
    
hypotheses = decoder.decode(log_probs_arranged, log_probs_length)
torch.save(hypotheses, f"{bit_phoneme_filepath}hypotheses.pth")

