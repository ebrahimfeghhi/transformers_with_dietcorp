import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle 


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
        sorted_indices = sorted(range(len(self.neural_time_bins)), 
                        key=lambda i: self.neural_time_bins[i], reverse=True)
        self.neural_feats = [self.neural_feats[i] for i in sorted_indices]
        self.phone_seqs = [self.phone_seqs[i] for i in sorted_indices]
        self.neural_time_bins = [self.neural_time_bins[i] for i in sorted_indices]
        self.phone_seq_lens = [self.phone_seq_lens[i] for i in sorted_indices]
        self.days = [self.days[i] for i in sorted_indices]
        
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
            torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
        )
        
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


from torch.utils.data import Sampler
import random

class ShuffleByBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_len = len(dataset)

    def __iter__(self):
        n = self.dataset_len
        indices = list(range(n))

        # Step 1: Group into batches
        batches = [indices[i:i + self.batch_size] for i in range(0, n, self.batch_size)]

        # Step 2: Shuffle the batch order
        random.shuffle(batches)

        # Step 3: Yield batches (lists of indices)
        for batch in batches:
            yield batch

    def __len__(self):
        return (self.dataset_len + self.batch_size - 1) // self.batch_size
    
def getDatasetLoaders_MAE(
    datasetName,
    batchSize
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset_MAE(loadedData["train"], transform=None)
    train_sampler = ShuffleByBatchSampler(train_ds, batch_size=batchSize)
    test_ds = SpeechDataset_MAE(loadedData["test"])

        
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def getDatasetLoaders(
    datasetName,
    batchSize
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
        
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def segment_data(data: torch.Tensor, N: int, X_len: torch.Tensor, day_idx: torch.Tensor):
    
    """
    Segments data into time-aligned batches of shape (B', N, F), where each segment
    includes only trials with sufficient valid data (according to X_len). If a trial's
    valid length is between start and end, include the last N-length chunk ending at X_len.

    Args:
        data (torch.Tensor): Input tensor of shape (B, T, F)
        N (int): Length of each time segment
        X_len (torch.Tensor): Valid lengths per trial (B,)
        day_idx (torch.Tensor): Day that each trial from the batch comes from (B, )

    Yields:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Segments of shape (B', N, F)
            - Corresponding day indices of shape (B',)
    """
    B, T, F = data.shape
    max_len = X_len.max().item()

    for start in range(0, max_len - N + 1, N):
        
        segments = []
        segment_days = []
        end = start + N
        
        for b in range(B):
            
            # get 
            x_len = X_len[b].item()
            
            # no padding issues here because X_len is longer than end.
            if x_len >= end:
                segment = data[b, start:end, :]
                segments.append(segment)
                segment_days.append(day_idx[b])
                
            # if there is still some new signal, but not long enough for a chunk
            # take the last N non padded timesteps.
            elif x_len > start:
                segment = data[b, x_len-N:x_len, :]
                segments.append(segment)
                segment_days.append(day_idx[b])
                
            # if signal has finished, randomly select a chunk to preserve batch size. 
            else:
                max_start = x_len - N
                rand_start = torch.randint(0, max_start + 1, (1,)).item()
                segment = data[b, rand_start:rand_start + N, :]
                segments.append(segment)
                segment_days.append(day_idx[b])

        
        yield torch.stack(segments), torch.stack(segment_days)
        
        
def sliding_chunks(x, chunk_size=32, stride=4):
    """
    x: Tensor of shape (B, T, C)
    Returns: Tensor of shape (B, M, chunk_size, C)
    """
    B, T, C = x.shape

    # Unfold the time dimension (dim=1) using torch.nn.functional.unfold logic
    x = x.unfold(dimension=1, size=chunk_size, step=stride).permute(0, 1, 3, 2)  # (B, M, chunk_size, C)
    return x