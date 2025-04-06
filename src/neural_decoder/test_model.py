from mae import MAE
from bit import BiT
from dataset import SpeechDataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import torch
from torch.utils.data import DataLoader
import numpy as np

args = {}
modelName = 'BiT'
args['outputDir'] = '/data/willett_data/outputs/' + modelName
args['datasetPath'] = '/data/willett_data/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.006
args['lrEnd'] = 0.006
args['patch_height'] = 32
args['patch_width'] = 2
args['n_heads'] = 20
args['mlp_ratio'] = 4
args['embedding_dim'] = 100
args['nBatch'] = 30000 #3000
args['n_layers'] = 3
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['l2_decay'] = 1e-5
args['device'] = 'cuda:2'


# load neural data 
def getDatasetLoaders(
    datasetName,
    batchSize,
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


import torch.nn.functional as F

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

trainLoader, testLoader, loadedData = getDatasetLoaders(
    args["datasetPath"],
    args["batchSize"],
)

X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
X = pad_to_multiple(X, multiple=32)
X = torch.unsqueeze(X, axis=1)

print(X.shape)

bit = BiT(trial_size=(X.shape[2], X.shape[3]), patch_size=(args['patch_height'], args['patch_width']), 
    num_classes=args['nClasses'], dim=args['embedding_dim'], depth=args['n_layers'], heads=args['n_heads'], 
    mlp_dim_ratio=args['mlp_ratio'], dropout=args['dropout'])

bit(X)


breakpoint()