import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle 
from tqdm import tqdm
import numpy as np


class SpeechDataset(Dataset):
    
    def __init__(self, data, transform=None, restricted_days=[], ventral_6v_only=False, return_transcript=False):
        
        self.data = data
        self.transform = transform
        self.return_transcript = return_transcript
    
        self.n_days = len(data)
        
            
        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []
        self.transcriptions = []
        
        #if channel_drop is not None:
        #    
        #     # select channel_drop integers from 0 - 64.
        #    channel_indices_array_1 = np.random.choice(np.arange(0, 65), size=channel_drop, replace=False)
    
            # select channel_drop integers from 128 - 192.
        #    channel_indices_array_2 = np.random.choice(np.arange(128, 193), size=channel_drop, replace=False)

            # add 64 to each of the indices because there are two features per channel.
        #    channel_indices_array_1 = np.concatenate([channel_indices_array_1, channel_indices_array_1+64])
        #    channel_indices_array_2 = np.concatenate([channel_indices_array_2, channel_indices_array_2+64])
        #    
        #    channel_indices = np.concatenate([channel_indices_array_1, channel_indices_array_2])

        for day in range(self.n_days):
            
            if restricted_days:
                
                # only train/test on restricted days
                if day not in restricted_days:
                    continue
            
            for trial in range(len(data[day]["sentenceDat"])):
                
                if ventral_6v_only:
                    self.neural_feats.append(data[day]["sentenceDat"][trial][:, 0:128])
                else:
                    self.neural_feats.append(data[day]["sentenceDat"][trial])
                    
                self.phone_seqs.append(data[day]["phonemes"][trial])
                self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])
                self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                self.transcriptions.append(data[day]['transcriptions'][trial])
                self.days.append(day)
                

        self.n_trials = len(self.days)

    def __len__(self):
        
        return self.n_trials

    def __getitem__(self, idx):
        
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)
            
        if self.return_transcript:
            
            return (
                neural_feats,
                torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
                torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
                torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
                torch.tensor(self.days[idx], dtype=torch.int64), 
                self.transcriptions[idx]
            )
            
        else:
            return (
                neural_feats,
                torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
                torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
                torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
                torch.tensor(self.days[idx], dtype=torch.int64)
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
    batchSize, 
    restricted_days=[],
    ventral_6v_only=False, 
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
        
  
    train_ds = SpeechDataset(loadedData["train"], transform=None, 
                             restricted_days=restricted_days, 
                             ventral_6v_only=ventral_6v_only)
    
    test_ds = SpeechDataset(loadedData["test"], 
                            restricted_days=restricted_days, 
                            ventral_6v_only=ventral_6v_only)

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
        batch_size=1,
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

def training_batch_generator(trainLoader, args):
    
    if args['batchStyle']:
        
        for i in range(args["nBatch"]):
            
            X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
            
            if i % 100 == 0:
                compute_val = True
            else:
                compute_val = False
                
            yield (
                X.to(args["device"]),
                y.to(args["device"]),
                X_len.to(args["device"]),
                y_len.to(args["device"]),
                dayIdx.to(args["device"]),
                compute_val
            )
            
    else:
        num_batches = len(trainLoader)
        for epoch in range(args["n_epochs"]):
            for batch_idx, (X, y, X_len, y_len, dayIdx) in enumerate(tqdm(trainLoader, desc=f"Training Epoch {epoch}")):
                compute_val = (batch_idx == num_batches - 1)
                yield (
                    X.to(args["device"]),
                    y.to(args["device"]),
                    X_len.to(args["device"]),
                    y_len.to(args["device"]),
                    dayIdx.to(args["device"]),
                    compute_val
                )