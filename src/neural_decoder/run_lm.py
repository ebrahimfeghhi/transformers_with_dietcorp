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
import matplotlib.pyplot as plt
from neural_decoder.dataset import getDatasetLoaders
import neural_decoder.lm_utils as lmDecoderUtils
from neural_decoder.lm_utils import build_llama_1B
from neural_decoder.model import GRUDecoder
from neural_decoder.bit import BiT_Phoneme
import pickle
import argparse
from lm_utils import _cer_and_wer
import json


def convert_sentence(s):
    s = s.lower()
    charMarks = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                 "'", ' ']
    ans = []
    for i in s:
        if(i in charMarks):
            ans.append(i)
    
    return ''.join(ans)

output_file = 'leia'
base_dir = "/home3/skaasyap/willett"

if output_file == 'obi':
    model_storage_path = '/data/willett_data/outputs/'
elif output_file == 'leia':
    model_storage_path = '/data/willett_data/leia_outputs/'
    
device = 'cuda:2'
load_lm = True
# LM decoding hyperparameters
acoustic_scale = 0.8
blank_penalty = np.log(2)

run_for_llm = True

if run_for_llm:
    print("LLM MODE")
    return_n_best = True
    rescore = True
else:
    return_n_best = False
    rescore = False
    

if load_lm: 
        
    lmDir = base_dir +'/lm/languageModel'
    ngramDecoder = lmDecoderUtils.build_lm_decoder(
        lmDir,
        acoustic_scale=acoustic_scale, #1.2
        nbest=1,
        beam=18
    )
    print("loaded LM")
    
    
models_to_run = [f'neurips_transformer_time_masked_seed_{i}' for i in range(1)]
partition = "test" # "test"
fill_max_day = False

if partition == 'test':
    saveFolder_transcripts = "/data/willett_data/model_transcriptions/"
else:
    saveFolder_transcripts = "/data/willett_data/model_transcriptions_comp/"
    
for model_name_str in models_to_run:
    
    modelPath = f"{model_storage_path}{model_name_str}"
    
    with open(modelPath + "/args", "rb") as handle:
        args = pickle.load(handle)
    
    
    if args['datasetPath'].rsplit('/', 1)[-1] == 'data_log_both':
        data_file = '/data/willett_data/ptDecoder_ctc_both'
    elif args['datasetPath'].rsplit('/', 1)[-1] == 'data':
        data_file = '/data/willett_data/ptDecoder_ctc'
    else:
        data_file = args['datasetPath']
        
    print(data_file)
    
    trainLoaders, testLoaders, loadedData = getDatasetLoaders(
        data_file, 8
    )
    
    # if true, model is a GRU
    if 'nInputFeatures' in args.keys():
        
        if 'max_mask_pct' not in args:
            args['max_mask_pct'] = 0
        if 'num_masks' not in args:
            args['num_masks'] = 0
        if 'input_dropout' not in args:
            args['input_dropout'] = 0
            
        print("Loading GRU")
        model = GRUDecoder(
            neural_dim=args["nInputFeatures"],
            n_classes=args["nClasses"],
            hidden_dim=args["nUnits"],
            layer_dim=args["nLayers"],
            nDays=args['nDays'],
            dropout=args["dropout"],
            device=args["device"],
            strideLen=args["strideLen"],
            kernelLen=args["kernelLen"],
            gaussianSmoothWidth=args["gaussianSmoothWidth"],
            bidirectional=args["bidirectional"],
            input_dropout=args['input_dropout'], 
            max_mask_pct=args['max_mask_pct'],
            num_masks=args['num_masks']
        ).to(device)

    else:
        
        if 'mask_token_zero' not in args:
            args['mask_token_zero'] = False
        
        print("Loading TRANSFORMER")
        
        # Instantiate model
        model = BiT_Phoneme(
            patch_size=args['patch_size'],
            dim=args['dim'],
            dim_head=args['dim_head'],
            nClasses=args['nClasses'],
            depth=args['depth'],
            heads=args['heads'],
            mlp_dim_ratio=args['mlp_dim_ratio'],
            dropout=args['dropout'],
            input_dropout=args['input_dropout'],
            look_ahead=args['look_ahead'],
            gaussianSmoothWidth=args['gaussianSmoothWidth'],
            T5_style_pos=args['T5_style_pos'],
            max_mask_pct=args['max_mask_pct'],
            num_masks=args['num_masks'], 
            mask_token_zeros=args['mask_token_zero'], 
            num_masks_channels=0, 
            max_mask_channels=0, 
            dist_dict_path=0, 
        ).to(device)
        
        
    ckpt_path = modelPath + '/modelWeights'
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    
    model.eval()

    model_outputs = {
        "logits": [],
        "logitLengths": [],
        "trueSeqs": [],
        "transcriptions": [],
    }
    
    total_edit_distance = 0
    total_seq_length = 0

    if partition == "competition":
        testDayIdxs = [4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20]
    elif partition == "test":
        testDayIdxs = range(len(loadedData[partition])) 
        
    ground_truth_sentences = []
    
    with torch.no_grad():
        
        for i, testDayIdx in enumerate(testDayIdxs):
            
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
                
                if fill_max_day:
                    dayIdx.fill_(args['maxDay'])
                
                pred = model.forward(X, X_len, dayIdx)
                
                if hasattr(model, 'compute_length'):
                    adjustedLens = model.compute_length(X_len)
                else:
                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
                    
                for iterIdx in range(pred.shape[0]):
                    trueSeq = np.array(y[iterIdx][0 : y_len[iterIdx]].cpu().detach())
                    model_outputs["logits"].append(pred[iterIdx].cpu().detach().numpy())
                    
                    model_outputs["logitLengths"].append(
                        adjustedLens[iterIdx].cpu().detach().item()
                    )
                    
                    model_outputs["trueSeqs"].append(trueSeq)
                    
                    decodedSeq = torch.argmax(
                        torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                        dim=-1,
                    ) 
                    
                    decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                    decodedSeq = decodedSeq.cpu().detach().numpy()
                    decodedSeq = np.array([i for i in decodedSeq if i != 0])
                    
                    matcher = SequenceMatcher(
                        a=trueSeq.tolist(), b=decodedSeq.tolist()
                    )
                    
                    total_edit_distance += matcher.distance()
                    total_seq_length += len(trueSeq)
                    
                transcript = loadedData[partition][i]["transcriptions"][j].strip()
                transcript = re.sub(r"[^a-zA-Z\- \']", "", transcript)
                transcript = transcript.replace("--", "").lower()
                model_outputs["transcriptions"].append(transcript)

        cer = total_edit_distance / total_seq_length
        
        print("Model performance: ", cer)

        if load_lm:
            
            llm_outputs = []
            start_t = time.time()
            nbest_outputs = []
            
            for j in range(len(model_outputs["logits"])):
                
                logits = model_outputs["logits"][j]
                
                logits = np.concatenate(
                    [logits[:, 1:], logits[:, 0:1]], axis=-1
                )  # Blank is last token
                
                logits = lmDecoderUtils.rearrange_speech_logits(logits[None, :, :], has_sil=True)
                
                nbest = lmDecoderUtils.lm_decode(
                    ngramDecoder,
                    logits[0],
                    blankPenalty=blank_penalty,
                    returnNBest=return_n_best,
                    rescore=rescore,
                )
                
                nbest_outputs.append(nbest)
                
            time_per_sample = (time.time() - start_t) / len(model_outputs["logits"])
            print(f"N-gram decoding took {time_per_sample} seconds per sample")
            
            if run_for_llm:
                print("SAVING OUTPUTS FOR LLM")
                with open(f"{saveFolder_transcripts}transcript.pkl", "wb") as f:
                    pickle.dump(model_outputs["transcriptions"], f)
                    
                with open(f"{saveFolder_transcripts}{model_name_str}.pkl", "wb") as f:
                    pickle.dump(nbest_outputs, f)
                
            else:
                # just get perf with greedy decoding
                for i in range(len(model_outputs["transcriptions"])):
                    model_outputs["transcriptions"][i] = model_outputs["transcriptions"][i].strip()
                    nbest_outputs[i] = nbest_outputs[i].strip()
                
                # lower case + remove puncs
                for i in range(len(model_outputs["transcriptions"])):
                    model_outputs["transcriptions"][i] = convert_sentence(model_outputs["transcriptions"][i])

                cer, wer = _cer_and_wer(nbest_outputs, model_outputs["transcriptions"], 
                                    outputType='speech', returnCI=True)

                print("CER and WER after 3-gram LM: ", cer, wer)            