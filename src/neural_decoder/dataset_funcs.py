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



# Dynamic Batch Sampler with Bucketing
class DynamicBatchSampler:
    def __init__(self, lengths, batch_size, shuffle=True, bucket_size=8000,
                  min_samples_in_bucket=16):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = bucket_size

        # Create buckets based on length ranges
        self.buckets = {}
        for idx, length in enumerate(lengths):
            
            # // returns floor(length / bucket_size)
            # so all inputs from T:T+bucket_size in length
            # are placed in the same bucket. 
            bucket_id = length // bucket_size
            if bucket_id not in self.buckets:
                self.buckets[bucket_id] = []
            self.buckets[bucket_id].append(idx)
            
            
        prev_bucket_id = None
        for bucket_id, vals in sorted(self.buckets.items()):
            
            # if bucket is too small
            if len(vals) < min_samples_in_bucket:
                # check if previous bucket exists
                if prev_bucket_id is not None:
                    
                    # merge small bucket into previous big bucket
                    self.buckets[prev_bucket_id].extend(vals)
                   
                # if no valid previous bucket, move ids
                # to the next available bucket
                elif prev_bucket_id is None: 
                    self.buckets[bucket_id+1].extend(vals)
                    
                # delete small bucket
                del self.buckets[bucket_id]
                    
            # if bucket is big enough, mark it as the last
            # valid bucket
            else: 
                prev_bucket_id = bucket_id
                    
    def __iter__(self):
        
        # shuffles inputs within a bucket
        if self.shuffle:
            for bucket in self.buckets.values():
                np.random.shuffle(bucket)

        # divides each bucket into batches
        batches = []
        for bucket in self.buckets.values():
            for i in range(0, len(bucket), self.batch_size):
                batches.append(bucket[i:i + self.batch_size])
                
        if self.shuffle:
            np.random.shuffle(batches)


        return iter(batches)

    def __len__(self):
        return sum(len(bucket) // self.batch_size for bucket in self.buckets.values())


    def print_bucket_sizes(self):
        print("Number of examples in each bucket:")
        for bucket_id, bucket in self.buckets.items():
            print(f"Bucket {bucket_id}: {len(bucket)} examples")
    #