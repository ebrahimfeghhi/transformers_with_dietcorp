import pickle 
from neural_decoder.model import GRUDecoder
from neural_decoder.bit import BiT_Phoneme
from edit_distance import SequenceMatcher
import neural_decoder.lm_utils as utils
import numpy as np
import torch

def loadModel(modelDir, nInputLayers=24, device="cuda:3"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    # print(args)

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
    return model, args


def loadTransformerModel(modelDir, device="cuda:3"):

    

    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)


    args['max_mask_pct'] = 0.075
    args['num_masks'] = 20
    args['mask_token_zero'] = True
    args['num_masks_channels'] = 4
    args['max_mask_channels'] = 4

    args['dist_dict_path'] = '/home3/skaasyap/willett/outputs/dist_dict.pt'

    args['consistency'] = False # apply consistency regularized CTC
    args['consistency_scalar'] = 0.2 # loss scaling factor
    # print(args)

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
        look_ahead=0,
        gaussianSmoothWidth=args['gaussianSmoothWidth'],
        T5_style_pos=args['T5_style_pos'], 
        max_mask_pct=args['max_mask_pct'], 
        num_masks=args['num_masks'], 
        consistency=args['consistency'], 
        mask_token_zeros=args['mask_token_zero'], 
        num_masks_channels=args['num_masks_channels'], 
        max_mask_channels=args['max_mask_channels'], 
        dist_dict_path=args['dist_dict_path']
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model, args

from torch.utils.data import DataLoader

from neural_decoder.model import GRUDecoder
from neural_decoder.dataset import SpeechDataset
from torch.nn.utils.rnn import pad_sequence

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

def rnn_forward(model, testLoader, device):
    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    all_logits = []
    trial_lengths = []
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

            pred = model.forward(X, X_len, testDayIdx)
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
            trial_lengths.append(adjustedLens)
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
                decoded_seqs.append(decodedSeq)
                true_seqs.append(trueSeq)
                total_edit_distance += matcher.distance()
                total_seq_length += len(trueSeq)

        avgDayLoss = np.sum(allLoss) / len(testLoader)
        cer = total_edit_distance / total_seq_length

        print(
            f"ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}"
        )

        return all_logits, trial_lengths, avgDayLoss, cer


def transformer_forward(model, testLoader, device):
    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    transformer_logits = []
    trial_lengths = []
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

            pred = model.forward(X, X_len, testDayIdx)
            transformer_logits.append(pred)
            
            adjustedLens = model.compute_length(X_len)
            trial_lengths.append(adjustedLens)
            
            loss = loss_ctc(
                torch.permute(pred.log_softmax(2), [1, 0, 2]),
                y,
                adjustedLens,
                y_len,
            )
            
            allLoss.append(loss.cpu().detach().numpy())

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

        avgDayLoss = np.mean(allLoss)
        cer = total_edit_distance / total_seq_length

        print(
            f"ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}"
        )

        return transformer_logits, trial_lengths, avgDayLoss, cer
    

def run_ngram_model(all_logits, trial_lengths, ngramDecoder, blank_penalty):
    llm_outputs = []
    for j in range(len(all_logits)):
        logits = all_logits[j][:trial_lengths[j], :]
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
        llm_outputs.append(nbest)

    return llm_outputs

def convert_sentence(s):
    s = s.lower()
    charMarks = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                 "'", ' ']
    ans = []
    for i in s:
        if(i in charMarks):
            ans.append(i)
    
    return ''.join(ans)