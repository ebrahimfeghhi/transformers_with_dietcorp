import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class SpeechDataset(Dataset):
    
    def __init__(self, data, transform=None):
        
        self.data = data
        self.transform = transform
        self.n_days = len(data)
        self.n_trials = sum([len(d["sentenceDat"]) for d in data])

        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []
        
        for day in range(self.n_days):
            
            for trial in range(len(data[day]["sentenceDat"])):
                
                self.neural_feats.append(data[day]["sentenceDat"][trial])
                self.phone_seqs.append(data[day]["phonemes"][trial])
                self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])
                self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                self.days.append(day)

    def __len__(self):
        
        return self.n_trials

    def __getitem__(self, idx):
        
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)

        return (
            neural_feats,
            torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
        )
        

class SpeechDataset_MAE(Dataset):
    
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.n_days = len(data)
        self.n_trials = sum([len(d["sentenceDat"]) for d in data])

        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []
        
        for day in range(self.n_days):
            
            for trial in range(len(data[day]["sentenceDat"])):
                
                self.neural_feats.append(data[day]["sentenceDat"][trial])
                self.phone_seqs.append(data[day]["phonemes"][trial])
                self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])
                self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                self.days.append(day)
                
        
        # sort trials by length for more effective chunking.         
        sorted_indices = sorted(range(len(self.neural_time_bins)), key=lambda i: self.neural_time_bins[i], reverse=True)
        self.neural_feats = [self.neural_feats[i] for i in sorted_indices]
        self.days = [self.days[i] for i in sorted_indices]
        self.neural_time_bins = [self.neural_time_bins[i] for i in sorted_indices]
        
    def shuffle_by_batch(self, batch_size):
        
        '''
        block shuffles by batch size self.neural_feats, self.days, and self.neural_time_bins
        by batch size. 
        '''
        n = len(self.neural_feats)
        assert len(self.days) == n and len(self.neural_time_bins) == n, "All arrays must be the same length"

        # Step 1: Group into batches
        indices = list(range(n))
        batches = [indices[i:i+batch_size] for i in range(0, n, batch_size)]

        # Step 2: Shuffle the batch order
        import random
        random.shuffle(batches)

        # Step 3: Flatten shuffled batches back into a single index list
        shuffled_indices = [i for batch in batches for i in batch]

        self.neural_feats = [self.neural_feats[i] for i in shuffled_indices]
        self.days = [self.days[i] for i in shuffled_indices]
        self.neural_time_bins = [self.neural_time_bins[i] for i in shuffled_indices]
        
    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)
        return (
            neural_feats, 
            torch.tensor(self.days[idx], dtype=torch.int64),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32))
        
def pad_to_multiple(tensor, multiple, dim=1, value=0):
    """
    Pads `tensor` along `dim` so that its size is divisible by `multiple`.
    """
    size = tensor.size(dim)
    padding_needed = (multiple - size % multiple) % multiple
    if padding_needed == 0:
        return tensor
    pad_dims = [0] * (2 * tensor.dim())
    pad_dims[-2 * dim - 1] = padding_needed  # padding at the end
    return F.pad(tensor, pad_dims, value=value)
