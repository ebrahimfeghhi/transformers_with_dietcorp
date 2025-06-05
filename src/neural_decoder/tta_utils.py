import torch
import re
import numpy as np
import math
from torch.utils.data import Subset
from g2p_en import G2p
g2p = G2p()  # <- Global instance
def convert_sentence(s):
    
    s = s.lower()
    charMarks = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                 "'", ' ']
    ans = []
    for i in s:
        if(i in charMarks):
            ans.append(i)
    
    return ''.join(ans)

def compute_lambda(memo_loss: torch.Tensor, D: int, gamma: float = 1.0) -> torch.Tensor:
    max_entropy = math.log(D)
    norm_entropy = memo_loss / max_entropy
    lambda_val = (1.0 - norm_entropy).clamp(min=0.0, max=1.0)  # safety clamp
    return lambda_val ** gamma

def clean_transcription(text):
    
    """
    Cleans a transcription string by:
    1. Removing leading/trailing whitespace
    2. Removing all characters except letters, hyphens, spaces, and apostrophes
    3. Removing double hyphens
    4. Converting to lowercase
    """
    
    text = str(text).strip()
    text = re.sub(r"[^a-zA-Z\- ']", '', text)
    text = text.replace('--', '')
    return text.lower()

def get_phonemes(thisTranscription):
    
    phonemes = []
    
    for p in g2p(thisTranscription):
        
        if p == ' ':
            phonemes.append('SIL')
        p = re.sub(r'[0-9]', '', p)  # Remove stress
        if re.match(r'^[A-Z]+$', p):  # Only keep phonemes (uppercase only)
            phonemes.append(p)
    
    phonemes.append('SIL')  # Add trailing SIL
    
    PHONE_DEF = [
        'AA', 'AE', 'AH', 'AO', 'AW',
        'AY', 'B',  'CH', 'D', 'DH',
        'EH', 'ER', 'EY', 'F', 'G',
        'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW',
        'OY', 'P', 'R', 'S', 'SH',
        'T', 'TH', 'UH', 'UW', 'V',
        'W', 'Y', 'Z', 'ZH','SIL'
    ]
    
    PHONE_DEF_SIL = PHONE_DEF + ['SIL']

    phoneme_ids = [PHONE_DEF_SIL.index(p) + 1 for p in phonemes]

    return torch.tensor(phoneme_ids, dtype=torch.long), torch.tensor([len(phoneme_ids)], dtype=torch.long)

def get_data_file(path):
    
    suffix_map = {
        "data_log_both": "/data/willett_data/ptDecoder_ctc_both",
        "data": "/data/willett_data/ptDecoder_ctc",
        "data_log_both_held_out_days": "/data/willett_data/ptDecoder_ctc_both_held_out_days",
        "data_log_both_held_out_days_1": "/data/willett_data/ptDecoder_ctc_both_held_out_days_1",
        "data_log_both_held_out_days_2": "/data/willett_data/ptDecoder_ctc_both_held_out_days_2",
    }
    suffix = path.rsplit('/', 1)[-1]
    return suffix_map.get(suffix, path)

def reverse_dataset(dataset):
    return Subset(dataset, list(reversed(range(len(dataset)))))

def get_dataloader(dataset, batch_size=1):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                       shuffle=False, num_workers=0)

def decode_sequence(pred, adjusted_len):
    pred = torch.argmax(pred[:adjusted_len], dim=-1)
    pred = torch.unique_consecutive(pred)
    return np.array([i for i in pred.cpu().numpy() if i != 0])


