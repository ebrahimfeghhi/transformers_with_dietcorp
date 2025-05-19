import re
import time
import pickle
import numpy as np

from edit_distance import SequenceMatcher
import torch
from dataset import SpeechDataset

import matplotlib.pyplot as plt


from neural_decoder.dataset import getDatasetLoaders
import neural_decoder.lm_utils as lmDecoderUtils
from neural_decoder.model import GRUDecoder
import pickle
import argparse

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
        input_dropout=args['inputDropout'],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


# parser = argparse.ArgumentParser(description="")
# parser.add_argument("--modelPath", type=str, default=None, help="Path to model")
# input_args = parser.parse_args()

modelPath = "/home3/skaasyap/willett/gru_model_weights/original_model"


with open(modelPath + "/args", "rb") as handle:
    args = pickle.load(handle)

args["datasetPath"] = "/home3/skaasyap/willett/data"
trainLoaders, testLoaders, loadedData = getDatasetLoaders(
    args["datasetPath"], 8
)

model = loadModel(modelPath, device="cuda:3")

device = "cuda:3"

model.eval()

rnn_outputs = {
    "logits": [],
    "logitLengths": [],
    "trueSeqs": [],
    "transcriptions": [],
}

partition = "test" # "test"
if partition == "competition":
    testDayIdxs = [4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20]
elif partition == "test":
    testDayIdxs = range(len(loadedData[partition]))

# import ipdb; ipdb.set_trace();

for testDayIdx in range(0, 24):
    i=testDayIdx
    test_ds = SpeechDataset([loadedData[partition][i]])
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0
    )
    for j, (X, y, X_len, y_len, _) in enumerate(test_loader):
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            torch.tensor([testDayIdx], dtype=torch.int64).to(device),
        )
        pred = model.forward(X, dayIdx)
        adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

        for iterIdx in range(pred.shape[0]):
            trueSeq = np.array(y[iterIdx][0 : y_len[iterIdx]].cpu().detach())

            rnn_outputs["logits"].append(pred[iterIdx].cpu().detach().numpy())
            rnn_outputs["logitLengths"].append(
                adjustedLens[iterIdx].cpu().detach().item()
            )
            rnn_outputs["trueSeqs"].append(trueSeq)

        transcript = loadedData[partition][i]["transcriptions"][j].strip()
        transcript = re.sub(r"[^a-zA-Z\- \']", "", transcript)
        transcript = transcript.replace("--", "").lower()
        rnn_outputs["transcriptions"].append(transcript)


# MODEL_CACHE_DIR = "/scratch/users/stfan/huggingface"
# # Load OPT 6B model
# llm, llm_tokenizer = lmDecoderUtils.build_opt(
#     cacheDir=MODEL_CACHE_DIR, device="auto", load_in_8bit=True
# )

# lmDir = "/oak/stanford/groups/henderj/stfan/code/nptlrig2/LanguageModelDecoder/examples/speech/s0/lm_order_exp/5gram/data/lang_test"
# ngramDecoder = lmDecoderUtils.build_lm_decoder(
#     lmDir, acoustic_scale=0.5, nbest=100, beam=18
# )

lmDir = "/home3/skaasyap/willett"+'/lm/languageModel'
ngramDecoder = lmDecoderUtils.build_lm_decoder(
    lmDir,
    acoustic_scale=0.8, #1.2
    nbest=1,
    beam=18
)
print("loaded LM")

# LM decoding hyperparameters
# acoustic_scale = 0.5
blank_penalty = np.log(2)
# llm_weight = 0.5

llm_outputs = []
# Generate nbest outputs from 5gram LM
start_t = time.time()
nbest_outputs = []
for j in range(len(rnn_outputs["logits"])):
    logits = rnn_outputs["logits"][j]
    logits = np.concatenate(
        [logits[:, 1:], logits[:, 0:1]], axis=-1
    )  # Blank is last token
    logits = lmDecoderUtils.rearrange_speech_logits(logits[None, :, :], has_sil=True)
    nbest = lmDecoderUtils.lm_decode(
        ngramDecoder,
        logits[0],
        blankPenalty=blank_penalty,
        returnNBest=False,
        rescore=False,
    )
    nbest_outputs.append(nbest)
time_per_sample = (time.time() - start_t) / len(rnn_outputs["logits"])
print(f"5gram decoding took {time_per_sample} seconds per sample")

for i in range(len(rnn_outputs["transcriptions"])):
    new_trans = [ord(c) for c in rnn_outputs["transcriptions"][i]] + [0]
    rnn_outputs["transcriptions"][i] = np.array(new_trans)

import ipdb; ipdb.set_trace();

# # Rescore nbest outputs with LLM
# start_t = time.time()
# llm_out = lmDecoderUtils.cer_with_gpt2_decoder(
#     llm,
#     llm_tokenizer,
#     nbest_outputs[:],
#     acoustic_scale,
#     rnn_outputs,
#     outputType="speech_sil",
#     returnCI=True,
#     lengthPenalty=0,
#     alpha=llm_weight,
# )
# # time_per_sample = (time.time() - start_t) / len(logits)
# print(f"LLM decoding took {time_per_sample} seconds per sample")

# print(llm_out["cer"], llm_out["wer"])
# with open(input_args.modelPath + "/llm_out", "wb") as handle:
#     pickle.dump(llm_out, handle)

# decodedTranscriptions = llm_out["decoded_transcripts"]
# with open(input_args.modelPath + "/5gramLLMCompetitionSubmission.txt", "w") as f:
#     for x in range(len(decodedTranscriptions)):
#         f.write(decodedTranscriptions[x] + "\n")