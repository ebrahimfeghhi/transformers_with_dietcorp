import re
from g2p_en import G2p
g2p = G2p()
import numpy as np

PHONE_DEF = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]
PHONE_DEF_SIL = PHONE_DEF + ['SIL']

def phoneToId(p):
    return PHONE_DEF_SIL.index(p)

def convert_to_phonemes(transcript):
    
    thisTranscription = transcript.strip()
    thisTranscription = re.sub(r'[^a-zA-Z\- \']', '', thisTranscription)
    thisTranscription = thisTranscription.replace('--', '').lower()
    addInterWordSymbol = True

    phonemes = []
    
    for p in g2p(thisTranscription):
        if addInterWordSymbol and p==' ':
            phonemes.append('SIL')
        p = re.sub(r'[0-9]', '', p)  # Remove stress
        if re.match(r'[A-Z]+', p):  # Only keep phonemes
            phonemes.append(p)

    #add one SIL symbol at the end so there's one at the end of each word
    if addInterWordSymbol:
        phonemes.append('SIL')
        
    seqLen = len(phonemes)
    maxSeqLen = 500
    seqClassIDs = np.zeros([maxSeqLen]).astype(np.int32)
    seqClassIDs[0:seqLen] = [phoneToId(p) + 1 for p in phonemes]
    return seqClassIDs, len(phonemes)
