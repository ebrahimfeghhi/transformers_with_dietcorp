import os
from glob import glob
from pathlib import Path
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]=""

import numpy as np
from omegaconf import OmegaConf
from neural_decoder.bit import BiT_Phoneme
import neural_decoder.lm_utils as utils
import pickle 
from neural_decoder.model import GRUDecoder
import torch

from torch.utils.data import DataLoader

from neural_decoder.model import GRUDecoder
from neural_decoder.dataset import SpeechDataset
from torch.nn.utils.rnn import pad_sequence


def loadModel(modelDir, nInputLayers=24, device="cuda:3"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model

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


base_dir = "/home3/skaasyap/willett"

lmDir = base_dir+'/lm/languageModel'
ngramDecoder = utils.build_lm_decoder(
    lmDir,
    acoustic_scale=0.8, #1.2
    nbest=1,
    beam=18
)

model = loadModel("/home3/skaasyap/willett/gru_model_weights/original_model")

trainLoader, testLoader, loadedData = getDatasetLoaders(
    "/home3/skaasyap/willett/data",
    8,
)

device = 'cuda:3'

loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
from edit_distance import SequenceMatcher

all_logits = []
decoded_seqs = []
true_seqs = []

with torch.no_grad():
    model.eval()
    allLoss = []
    total_edit_distance = 0
    total_seq_length = 0
    for X, y, X_len, y_len, testDayIdx in testLoader:
        X, y, X_len, y_len, testDayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            testDayIdx.to(device),
        )

        pred = model.forward(X, testDayIdx)
        all_logits.append(pred)
        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)
        allLoss.append(loss.cpu().detach().numpy())

        adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
            torch.int32
        )
        for iterIdx in range(pred.shape[0]):
            decodedSeq = torch.argmax(
                torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                dim=-1,
            )  # [num_seq,]
            decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
            decodedSeq = decodedSeq.cpu().detach().numpy()
            decodedSeq = np.array([i for i in decodedSeq if i != 0])
            # print("decodedSeq", decodedSeq.shape)
            trueSeq = np.array(
                y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
            )

            matcher = SequenceMatcher(
                a=trueSeq.tolist(), b=decodedSeq.tolist()
            )
            decoded_seqs.append(decodedSeq)
            true_seqs.append(trueSeq)
            total_edit_distance += matcher.distance()
            total_seq_length += len(trueSeq)

    avgDayLoss = np.sum(allLoss) / len(testLoader)
    cer = total_edit_distance / total_seq_length

    print(
        f"ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}"
    )


all_logits = [l for batch in all_logits for l in list(batch)]

acoustic_scale = 0.8
blank_penalty = np.log(2)

llm_outputs = []
# Generate nbest outputs from 5gram LM
nbest_outputs = []
for j in range(len(all_logits)):
    logits = all_logits[j].detach().cpu()
    # print(logits[:, 1:].shape, logits[:, 0:1].shape)
    logits = np.concatenate(
        [logits[:, 1:], logits[:, 0:1]], axis=-1
    )  # Blank is last token
    logits = utils.rearrange_speech_logits(logits[None, :, :], has_sil=True)
    nbest = utils.lm_decode(
        ngramDecoder,
        logits[0],
        blankPenalty=blank_penalty,
        returnNBest=False,
        rescore=False,
    )
    nbest_outputs.append(nbest)
    print(nbest)