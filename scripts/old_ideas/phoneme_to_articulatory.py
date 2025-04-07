import numpy as np


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
        "P": 0, "B": 1, "T": 0, "D": 1, "K": 0, "G": 1, "F": 0, "V": 1,
        "TH": 0, "DH": 1, "S": 0, "Z": 1, "SH": 0, "ZH": 1, "CH": 0, "JH": 1,
        "M": 1, "N": 1, "NG": 1, "L": 1, "R": 1, "W": 1, "Y": 1, "HH": 0,
        "IY": 1, "IH": 1, "EH": 1, "AE": 1, "AA": 1, "UW": 1, "UH": 1, "AO": 1,
        "OW": 1, "ER": 1, "AH": 1, "AW": 1, "AY": 1, "EY": 1, "OY": 1
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
    
    
def convert_phonemes_to_articulatory(phoneme_list, maxSeqLen=500):
    
    articulatory_feats_list = []
    
    for phoneme in phoneme_list:
        
        articulatory_feats = get_phoneme_features(phoneme)
        
        if type(articulatory_feats[0]) == list:
            for a in articulatory_feats:
                articulatory_feats_list.append(a)
        else:
            
            articulatory_feats_list.append(articulatory_feats)
            
    stacked_feats = np.stack(articulatory_feats_list) + 1
    
    padded_array = np.zeros((maxSeqLen - stacked_feats.shape[0], stacked_feats.shape[1]))
            
    return np.vstack((stacked_feats, padded_array)).astype(np.int32)