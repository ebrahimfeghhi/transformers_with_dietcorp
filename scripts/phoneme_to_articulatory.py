import numpy as np
import torch


def get_phoneme_features(phoneme):
    
    '''
    This function takes as input a phoneme and outputs a 
    6D articulatory feature of the following form:
    
    [voiced, place of articulation, manner of articulation, height, backness, roundedness]
    
    Some phonemes, such as diphthongs, have dynamic properties. For these phonemes, the 
    value of certain articulatory features is provided as a list with the start and end
    position. Two articulatory feature vectors are returned for these phonemes,
    one for the start position and one for the end position. 
    '''
    
    if phoneme == 'SIL':
        return [0,0,0,0,0,0]
    
    # Define mappings from strings to integers
    place_mapping = {
        "bilabial": 1, "labiodental": 2, "dental": 3, "alveolar": 4, "postalveolar": 5, 
        "retroflex": 6, "palatal": 7, "velar": 8, "glottal": 9, 0: 0
    }
    
    manner_mapping = {
        "plosive": 1, "fricative": 2, "affricate": 3, "nasal": 4, 
        "liquid": 5, "approximant": 6, 0: 0
    }
    
    height_mapping = {
        "high": 1, "mid": 2, "low": 3, 0: 0
    }
    
    backness_mapping = {
        "front": 1, "central": 2, "back": 3, 0: 0
    }
    
    roundedness_mapping = {
        "unrounded": 1, "rounded": 2, 0: 0
    }

    # Voicing (1 = voiced, 0 = voiceless)
    voiced = {
        "P": 1, "B": 2, "T": 1, "D": 2, "K": 1, "G": 2, "F": 1, "V": 2,
        "TH": 1, "DH": 2, "S": 1, "Z": 2, "SH": 1, "ZH": 2, "CH": 1, "JH": 2,
        "M": 2, "N": 2, "NG": 2, "L": 2, "R": 2, "W": 2, "Y": 2, "HH": 1,
        "IY": 2, "IH": 2, "EH": 2, "AE": 2, "AA": 2, "UW": 2, "UH": 2, "AO": 2,
        "OW": 2, "ER": 2, "AH": 2, "AW": 2, "AY": 2, "EY": 2, "OY": 2, 0:0
    }

    # Function to convert lists to integer lists
    def convert_to_int(value, mapping):
        if isinstance(value, list):
            return [mapping[v] for v in value]
        return mapping[value]

    # Feature dictionaries
    place = {k: convert_to_int(v, place_mapping) for k, v in {
        "P": "bilabial", "B": "bilabial", "M": "bilabial",
        "F": "labiodental", "V": "labiodental",
        "TH": "dental", "DH": "dental",
        "T": "alveolar", "D": "alveolar", "S": "alveolar", "Z": "alveolar",
        "N": "alveolar", "L": "alveolar",
        "SH": "postalveolar", "ZH": "postalveolar", "CH": "postalveolar", "JH": "postalveolar",
        "R": "retroflex",
        "K": "velar", "G": "velar", "NG": "velar",
        "W": ["bilabial", "velar"], "Y": "palatal", "HH": "glottal"
    }.items()}

    manner = {k: convert_to_int(v, manner_mapping) for k, v in {
        "P": "plosive", "B": "plosive", "T": "plosive", "D": "plosive",
        "K": "plosive", "G": "plosive",
        "F": "fricative", "V": "fricative", "TH": "fricative", "DH": "fricative",
        "S": "fricative", "Z": "fricative", "SH": "fricative", "ZH": "fricative",
        "HH": "fricative",
        "CH": "affricate", "JH": "affricate",
        "M": "nasal", "N": "nasal", "NG": "nasal",
        "L": "approximant", "R": "approximant",
        "W": "approximant", "Y": "approximant"
    }.items()}

    height = {k: convert_to_int(v, height_mapping) for k, v in {
        "IY": "high", "IH": "high", "EH": "mid", "AE": "low",
        "AA": "low", "UW": "high", "UH": "high", "AO": "mid",
        "OW": "mid", "ER": "mid", "AH": "mid",
        "AW": ["low", "mid"], "AY": ["low", "high"], "EY": ["mid", "high"], "OY": ["mid", "high"]
    }.items()}

    backness = {k: convert_to_int(v, backness_mapping) for k, v in {
        "IY": "front", "IH": "front", "EH": "front", "AE": "front",
        "AA": "back", "UW": "back", "UH": "back", "AO": "back",
        "OW": "back", "ER": "central", "AH": "central",
        "AW": ["back", "front"], "AY": ["front", "front"], "EY": "front", "OY": ["back", "front"]
    }.items()}

    roundedness = {k: convert_to_int(v, roundedness_mapping) for k, v in {
        "IY": "unrounded", "IH": "unrounded", "EH": "unrounded", "AE": "unrounded",
        "AA": "unrounded", "UW": "rounded", "UH": "rounded", "AO": "rounded",
        "OW": "rounded", "ER": "unrounded", "AH": "unrounded",
        "AW": ["unrounded", "rounded"], "AY": "unrounded", "EY": "unrounded", "OY": ["rounded", "unrounded"]
    }.items()}

    # Construct vector(s)
    properties = [voiced, place, manner, height, backness, roundedness]
    values = [prop.get(phoneme, 0) for prop in properties]

    # If any value is a list, return two vectors
    if any(isinstance(v, list) for v in values):
        return [list(v[i] if isinstance(v, list) else v for v in values) for i in range(2)]
    else:
        return values
    
    
def convert_phonemes_to_articulatory(phoneme_list, maxSeqLen=500, to_pad = False):
    
    articulatory_feats_list = []
    
    for phoneme in phoneme_list:
        if(phoneme == 'EPS'):
            articulatory_feats_list.append([-1, -1, -1, -1, -1, -1])
            continue
        
        articulatory_feats = get_phoneme_features(phoneme)
        
        if type(articulatory_feats[0]) == list:
            for a in articulatory_feats:
                articulatory_feats_list.append(a)
        else:
            
            articulatory_feats_list.append(articulatory_feats)
            
    stacked_feats = np.stack(articulatory_feats_list) + 1

    # import ipdb; ipdb.set_trace();
    if not to_pad:
        return stacked_feats.astype(np.int32)
    
    padded_array = np.zeros((maxSeqLen - stacked_feats.shape[0], stacked_feats.shape[1]))
            
    return np.vstack((stacked_feats, padded_array)).astype(np.int32)


# def find_possible_phonemes(logits):
#     best_prob = 0
#     best_phoneme = None
    

#     for phoneme in all_phonemes:
#         feats = convert_phonemes_to_articulatory([phoneme], maxSeqLen = 1)        
#         for idx in range(len(feats)):
#             cur_prob = 1
#             for i in range(len(feats[idx])):
#                 cur_logits = torch.softmax(logits[i], dim=0)
#                 cur_prob *= cur_logits[feats[idx][i]]
        
#             if(cur_prob > best_prob):
#                 best_prob = cur_prob 
#                 best_phoneme = phoneme

#     return all_phonemes.index(best_phoneme)

matrix = []
corresponding_phonemes = []
num_feat_list = [3, 10, 7, 4, 4, 3]
all_phonemes = [
    'EPS','AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH', 'SIL']
# import ipdb; ipdb.set_trace();
for phoneme in all_phonemes:
    feat_list = convert_phonemes_to_articulatory([phoneme], to_pad=False)

    for idx in range(len(feat_list)):
        corresponding_phonemes.append(all_phonemes.index(phoneme))
        feats = feat_list[idx]
        val = 0
        list_art_feats = [0]*(sum(num_feat_list) + len(num_feat_list)) #37 things because of the silences
        for feat in range(len(feats)):
            list_art_feats[val + feats[feat]] = 1
            val += num_feat_list[feat] + 1
        
        matrix.append(list_art_feats)
# import ipdb; ipdb.set_trace();
matrix = torch.from_numpy(np.array(matrix).T).to('cuda:2').to(torch.float32)

corresponding_phonemes = torch.tensor(corresponding_phonemes).to('cuda:2')

def decode_articulatory_seq(art_preds):
    #input should be a list with 6 elements, each of shape (T, x) where x is the number of features for each articulatory feature

    #concat into shape T phoneme list
    concat_logits = art_preds[0]
    for i in range(1, len(art_preds)):
        concat_logits = torch.cat((concat_logits, art_preds[i]), dim=1)

    #matrix multiplying between the time steps and the phoneme matrix, note that adding the log probs is equivalent to taking the product of the probabilities
        
    seq = concat_logits @ matrix #shape (T, #phonemes)
    decoded_seq = torch.argmax(seq, dim=-1)
    decoded_seq = corresponding_phonemes[decoded_seq]
    return decoded_seq