import os
import pickle
import time
import sys
from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from phoneme_to_articulatory import decode_articulatory_seq
from .model_articulation import GRUDecoder
from .dataset_articulation import SpeechDataset


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        
        X, y_phoneme, y_voiced, y_place, y_manner, \
        y_height, y_backness, y_roundedness, X_lens, \
        y_phone_lens, y_articulatory_lens, days = zip(*batch)
        
        # Pad inputs
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        
        # Pad phoneme targets
        y_phoneme_padded = pad_sequence(y_phoneme, batch_first=True, padding_value=0)

        # Pad articulatory feature targets separately
        y_voiced_padded = pad_sequence(y_voiced, batch_first=True, padding_value=0)
        y_place_padded = pad_sequence(y_place, batch_first=True, padding_value=0)
        y_manner_padded = pad_sequence(y_manner, batch_first=True, padding_value=0)
        y_height_padded = pad_sequence(y_height, batch_first=True, padding_value=0)
        y_backness_padded = pad_sequence(y_backness, batch_first=True, padding_value=0)
        y_roundedness_padded = pad_sequence(y_roundedness, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_phoneme_padded,
            y_voiced_padded,
            y_place_padded,
            y_manner_padded,
            y_height_padded,
            y_backness_padded,
            y_roundedness_padded,
            torch.stack(X_lens),
            torch.stack(y_phone_lens),
            torch.stack(y_articulatory_lens),
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

def trainModel(args):
    
    # for combining phoneme level and articulatory loss
    lambda_weight = 1  # Tune this parameter based on validation performance
    
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda:0"
    
    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes_list=args["nClasses_list"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

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

    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    for batch in range(args["nBatch"]):
        model.train()

        X, y_phoneme, y_voiced, y_place, y_manner, \
        y_height, y_backness, y_roundedness, X_len, y_phone_len, \
        y_articulatory_len, dayIdx = next(iter(trainLoader))
        
        
        X, y_phoneme, y_voiced, y_place, y_manner, \
        y_height, y_backness, y_roundedness, X_len, y_phone_len, \
        y_articulatory_len, dayIdx = (
            X.to(device),
            y_phoneme.to(device),
            
            y_voiced.to(device),
            y_place.to(device),
            y_manner.to(device),
            y_height.to(device),
            y_backness.to(device), 
            y_roundedness.to(device), 
            
            X_len.to(device),
            y_phone_len.to(device),
            y_articulatory_len.to(device), 
            
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )

        # Forward pass
        preds = model.forward(X, dayIdx)

        # Compute phoneme loss
        phoneme_pred = preds[0]  # Assuming phoneme is first output
        phoneme_pred = phoneme_pred.log_softmax(2).permute(1, 0, 2)
        input_lengths = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
        phoneme_loss = loss_ctc(phoneme_pred, y_phoneme, input_lengths, y_phone_len)

        # Compute articulatory feature losses
        articulatory_preds = preds[1:]  # The rest of the outputs
        articulatory_targets = [y_voiced, y_place, y_manner, y_height, y_backness, y_roundedness]

        # if(batch == 100):
        #     import ipdb; ipdb.set_trace();
        #print(articulatory_preds, articulatory_targets)

        articulatory_loss = 0
        for pred, target in zip(articulatory_preds, articulatory_targets):
            pred = pred.log_softmax(2).permute(1, 0, 2)
            articulatory_loss += loss_ctc(pred, target, input_lengths, y_articulatory_len)

    

        # Total loss
        loss = lambda_weight * phoneme_loss + (1-lambda_weight) * articulatory_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                eval_articulatory_loss = 0
                art_edit_distance = [0, 0, 0, 0, 0, 0]
                art_seq_length = [0, 0, 0, 0, 0, 0]
                
                
                for X, y_phoneme, y_voiced, y_place, y_manner, \
                    y_height, y_backness, y_roundedness, X_len, \
                    y_phone_len, y_articulatory_len, testDayIdx in testLoader:
                    
                    X, y_phoneme, y_voiced, y_place, y_manner, \
                    y_height, y_backness, y_roundedness, X_len, \
                    y_phone_len, y_articulatory_len, testDayIdx = (
                        X.to(device),
                        y_phoneme.to(device),
                        y_voiced.to(device),
                        y_place.to(device),
                        y_manner.to(device),
                        y_height.to(device),
                        y_backness.to(device),
                        y_roundedness.to(device),
                        X_len.to(device),
                        y_phone_len.to(device),
                        y_articulatory_len.to(device),
                        testDayIdx.to(device),
                    )

                    
                    pred_all = model.forward(X, testDayIdx)
                    pred = pred_all[0] # only take phoneme predictions for test
                    
                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y_phoneme,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_phone_len,
                    )

                    ###
                    eval_articulatory_preds = pred_all[1:]
                    if(eval_articulatory_preds[0].shape[1] < max(input_lengths)):
                        continue
                    
                    eval_articulatory_targets = [y_voiced, y_place, y_manner, y_height, y_backness, y_roundedness]

                    
                    for idx, (art_pred, art_target) in enumerate(zip(eval_articulatory_preds, eval_articulatory_targets)):
                        art_pred = art_pred.log_softmax(2).permute(1, 0, 2)
                        eval_articulatory_loss += loss_ctc(art_pred, art_target, adjustedLens, y_articulatory_len)

                        art_pred = art_pred.permute(1, 0, 2)
                        art_pred_vals = torch.argmax(art_pred, dim=-1)


                        
                        for b in range(art_pred.shape[0]):
                            art_unique_vals = torch.unique_consecutive(art_pred_vals[b][:adjustedLens[b]])
                        

                            art_unique_vals = art_unique_vals.cpu().detach().numpy()

                            cur_pred_vals = np.array([i for i in art_unique_vals if i != 0])

                            trueSeq = np.array(
                                art_target[b][0 : y_articulatory_len[b]].cpu().detach()
                            )
                            matcher = SequenceMatcher(
                                a=trueSeq.tolist(), b=cur_pred_vals.tolist()
                            )
                            art_edit_distance[idx] += matcher.distance()
                            
                            art_seq_length[idx] += len(trueSeq)

                    loss = lambda_weight * phoneme_loss + (1-lambda_weight) * eval_articulatory_loss
                    allLoss.append(loss.cpu().detach().numpy())

                    
                    # for iterIdx in range(eval_articulatory_preds[0].shape[0]):
                    #     logits = []
                    #     for feat in range(len(eval_articulatory_preds)):
                    #         logits.append(torch.tensor(eval_articulatory_preds[feat][iterIdx, 0 : adjustedLens[iterIdx], :])) 
                    #     # if batch==800:
                    #     #     import ipdb; ipdb.set_trace();
                    #     decodedSeq = decode_articulatory_seq(logits) #returns the corresponding phoneme seq
                        
                    #     decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                    #     decodedSeq = decodedSeq.cpu().detach().numpy()
                    #     decodedSeq = np.array([i for i in decodedSeq if i != 0])
                    #     trueSeq = np.array(
                    #         y_phoneme[iterIdx][0 : y_phone_len[iterIdx]].cpu().detach()
                    #     )

                    #     matcher = SequenceMatcher(
                    #         a=trueSeq.tolist(), b=decodedSeq.tolist()
                    #     )
                        
                    #     total_edit_distance += matcher.distance()
                        
                    #     total_seq_length += len(trueSeq)


                avgDayLoss = np.sum(allLoss) / len(testLoader)
                # cer = total_edit_distance / total_seq_length
                art_cer = [art_edit_distance[i]/art_seq_length[i] for i in range(len(art_edit_distance))]

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f},  art_cer: {art_cer}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            # if len(testCER) > 0 and art_cer < np.min(testCER):
            #     torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
                
            testLoss.append(avgDayLoss)
            testCER.append(art_cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            # with open(args["outputDir"] + "/trainingStats", "wb") as file:
            #     pickle.dump(tStats, file)


def loadModel(modelDir, nInputLayers=24, device="cuda:0"):
    
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

@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()