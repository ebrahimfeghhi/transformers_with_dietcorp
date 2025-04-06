import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .slightly_modified_gru import GRUDecoder
from .stage2model import Speech2NeuralDecoder
from .dataset import SpeechDataset
import torchaudio
from .dataset_funcs import convert_to_phonemes, DynamicBatchSampler
import wandb
# wandb.init(mode="disabled")

from torch.utils.data import DataLoader
import torchaudio.transforms as T
from speechbrain.lobes.models.huggingface_transformers.hubert import HuBERT

import nltk
nltk.download('averaged_perceptron_tagger_eng')


# Custom collate function with padding
def collate_fn(batch):
    
    waveforms = [item[0].squeeze(0) for item in batch]  # Remove channel dimension
    lengths = torch.tensor([wav.shape[0] for wav in waveforms])
    lengths = lengths / max(lengths)
    
    # Pad sequences to match longest in batch
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(
        waveforms, 
        batch_first=True
    )
    
    # Process batch in a single list comprehension (avoids intermediate lists)
    processed_batch = [convert_to_phonemes(item[2]) for item in batch]

    # Unpack using numpy for tensor conversion
    transcripts, transcript_lengths = zip(*processed_batch)

    # Convert using numpy stacking for better performance
    transcripts = torch.from_numpy(np.stack(transcripts))  # For multi-dimensional arrays
    transcript_lengths = torch.as_tensor(np.array(transcript_lengths), dtype=torch.long)
    
    return padded_waveforms, transcripts, lengths, transcript_lengths

def trainModel(args):
    
    hubert_path = "facebook/hubert-large-ls960-ft"
    model_hubert = HuBERT(hubert_path, save_path="/home3/skaasyap/willett", freeze=True).to(args["device"])
    
    # load the original gru model 
    gru_model = loadModel("/home3/skaasyap/willett/original_model", device=args["device"])
    
    # freeze parameters
    for param in gru_model.parameters():
        param.requires_grad = False
        
    model = Speech2NeuralDecoder(
        latentspeech_dim=args['latentspeech_dim_stage2'],
        output_dim=args["outputdim_stage2"],
        hidden_dim=args["nUnits_stage2"],
        layer_dim=args["nLayers_stage2"],
        dropout=args["dropout_stage2"],
        device=args["device"],
        strideLen=args["strideLen_stage2"],
        kernelLen=args["kernelLen_stage2"],
        bidirectional=args["bidirectional_stage2"],
    ).to(args["device"])

    # import ipdb; ipdb.set_trace();
            
    librispeech_path = "/home3/skaasyap/willett/speech_data/librispeech"
    
    trainDataset = torchaudio.datasets.LIBRISPEECH(
        root=librispeech_path,  # or the full path to the parent directory
        url="train-clean-100",
        download=False
    )
    
    valDataset = torchaudio.datasets.LIBRISPEECH(
        root=librispeech_path,  # or the full path to the parent directory
        url="dev-clean",
        download=False
    )
    
    train_batch_sampler = DynamicBatchSampler(
        lengths=np.load('/data/LLMs/librispeech/LibriSpeech/train-clean-100/lengths.npy'),
        batch_size=args["batchSize"],
        shuffle=True,
        bucket_size=args["bucket_size"],
        min_samples_in_bucket=args["min_samples_in_bucket"]
    )
        
    # Create DataLoader
    trainLoader = DataLoader(
        trainDataset,
        batch_sampler=train_batch_sampler,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_batch_sampler = DynamicBatchSampler(
        lengths=np.load('/data/LLMs/librispeech/LibriSpeech/dev-clean/lengths.npy'),
        batch_size=8,
        shuffle=True,
        bucket_size=args["bucket_size"],
        min_samples_in_bucket=args["min_samples_in_bucket"]
    )
    
    # Create DataLoader
    valLoader = DataLoader(
        valDataset,
        batch_sampler=val_batch_sampler,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # gpu_id = 1 # Change this if you have multiple GPUs
    # total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
    # reserved_memory = torch.cuda.memory_reserved(gpu_id)
    # allocated_memory = torch.cuda.memory_allocated(gpu_id)
    # free_memory = total_memory - allocated_memory

    # print(f"Total GPU Memory: {total_memory / (1024**2):.2f} MB")
    # print(f"Allocated GPU Memory: {allocated_memory / (1024**2):.2f} MB")
    # print(f"Reserved GPU Memory: {reserved_memory / (1024**2):.2f} MB")
    # print(f"Free GPU Memory: {free_memory / (1024**2):.2f} MB")

    # import ipdb; ipdb.set_trace();
    wandb.init(project="Neural Decoder", entity="skaasyap-ucla", config=dict(args))
    
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)
    
    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nBatch"],
    )
    
    # gpu_id = 1 # Change this if you have multiple GPUs
    # total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
    # reserved_memory = torch.cuda.memory_reserved(gpu_id)
    # allocated_memory = torch.cuda.memory_allocated(gpu_id)
    # free_memory = total_memory - allocated_memory

    # print(f"Total GPU Memory: {total_memory / (1024**2):.2f} MB")
    # print(f"Allocated GPU Memory: {allocated_memory / (1024**2):.2f} MB")
    # print(f"Reserved GPU Memory: {reserved_memory / (1024**2):.2f} MB")
    # print(f"Free GPU Memory: {free_memory / (1024**2):.2f} MB")

    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    
    for batch in range(args["nBatch"]):
        
        model.train()

        X, y, X_len, y_len = next(iter(trainLoader))

        X, y, X_len, y_len = (
            X.to(args["device"]),
            y.to(args["device"]),
            X_len.to(args["device"]),
            y_len.to(args["device"]),
        )

        

        # Noise augmentation is faster on GPU
        #if args["whiteNoiseSD"] > 0:
        #    X += torch.randn(X.shape, device=args["device"]) * args["whiteNoiseSD"]

        #if args["constantOffsetSD"] > 0:
        #    X += (
        #        torch.randn([X.shape[0], 1, X.shape[2]], device=args["device"])
        #        * args["constantOffsetSD"]
        #    )

        # Compute prediction error
        time1 = time.time()
        X = model_hubert(X, wav_lens=X_len).detach()
        time2 = time.time()

        # batch size, X.shape[1]//1024 - 1, 1024
        #X = torch.randn(64, 800, 1024, device=args["device"])
        predicted_neural_data = model.forward(X)
        time3 = time.time()
        # import ipdb; ipdb.set_trace();
        pred = gru_model(predicted_neural_data, None, 'speech')
        time4 = time.time()
        X_len_ctc = torch.floor(predicted_neural_data.shape[1]*X_len)
        
        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len_ctc - gru_model.kernelLen) / gru_model.strideLen).to(torch.int32),
            y_len,
        )
        time6 = time.time()

        # print("batch: ", batch, "times: ", time2-time1, time3-time2, time4-time3, time5-time4, time6-time5)
        

        loss = torch.sum(loss)
        
        # Backpropagation
        time7 = time.time()
        optimizer.zero_grad()
  
        loss.backward()
  
        optimizer.step()
     
        scheduler.step()

        time8 = time.time()

        # print("backprop time:", time8-time7)
        # import ipdb; ipdb.set_trace();

        # torch.cuda.empty_cache()

        # gpu_id = 1 # Change this if you have multiple GPUs
        # total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        # reserved_memory = torch.cuda.memory_reserved(gpu_id)
        # allocated_memory = torch.cuda.memory_allocated(gpu_id)
        # free_memory = total_memory - allocated_memory

        # print(f"Total GPU Memory: {total_memory / (1024**2):.2f} MB")
        # print(f"Allocated GPU Memory: {allocated_memory / (1024**2):.2f} MB")
        # print(f"Reserved GPU Memory: {reserved_memory / (1024**2):.2f} MB")
        # print(f"Free GPU Memory: {free_memory / (1024**2):.2f} MB")
        # import ipdb; ipdb.set_trace();
        
        #print(endTime - startTime)
  
        # Eval
        if batch % 100 == 0:
            
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len in valLoader:
                    X, y, X_len, y_len = (
                        X.to(args["device"]),
                        y.to(args["device"]),
                        X_len.to(args["device"]),
                        y_len.to(args["device"]),
                    )

                    # gpu_id = 1 # Change this if you have multiple GPUs
                    # total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
                    # reserved_memory = torch.cuda.memory_reserved(gpu_id)
                    # allocated_memory = torch.cuda.memory_allocated(gpu_id)
                    # free_memory = total_memory - allocated_memory

                    # print(f"Total GPU Memory: {total_memory / (1024**2):.2f} MB")
                    # print(f"Allocated GPU Memory: {allocated_memory / (1024**2):.2f} MB")
                    # print(f"Reserved GPU Memory: {reserved_memory / (1024**2):.2f} MB")
                    # print(f"Free GPU Memory: {free_memory / (1024**2):.2f} MB")

                    # import ipdb; ipdb.set_trace();

                    X = model_hubert(X, wav_lens=X_len)
                    # X = np.random.rand(64, 821, 1024)
                    # X = torch.rand((64, 821, 1024), dtype=torch.float32).to(args['device'])
                    predicted_neural_data = model.forward(X)
                    pred = gru_model(predicted_neural_data, None, 'speech')
                    
                    X_len_ctc = torch.floor(predicted_neural_data.shape[1]*X_len)
                    
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len_ctc - gru_model.kernelLen) / gru_model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len_ctc - gru_model.kernelLen) / gru_model.strideLen).to(
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

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(valLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                  
                # Log the metrics to wandb
                wandb.log({
                    "batch": batch,
                    "ctc_loss": avgDayLoss,
                    "cer": cer,
                    "time_per_batch": (endTime - startTime) / 100
                })
                
                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)


def loadModel(modelDir, nInputLayers=24, device="cuda:1"):
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


#@hydra.main(version_base="1.1", config_path="conf", config_name="config")
#def main(cfg):
#    cfg.outputDir = os.getcwd()
#    # Initialize wandb with a project name (you can customize this)
#    trainModel(cfg)#
#
#if __name__ == "__main__":
#    main()