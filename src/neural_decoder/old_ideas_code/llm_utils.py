import os
from time import time

import numpy as np
from tqdm.notebook import trange, tqdm
import neural_decoder.old_ideas.rnnEval as rnnEval

device = 'cuda:2'

def gpt2_lm_decode(model, tokenizer, nbest, acousticScale, lengthPenlaty, alpha,
                   returnConfidence=False):
    
    hypotheses = []
    acousticScores = []
    oldLMScores = []
    for out in nbest:
        hyp = out[0].strip()
        if len(hyp) == 0:
            continue
        hyp = hyp.replace('>', '')
        hyp = hyp.replace('  ', ' ')
        hyp = hyp.replace(' ,', ',')
        hyp = hyp.replace(' .', '.')
        hyp = hyp.replace(' ?', '?')
        hypotheses.append(hyp)
        acousticScores.append(out[1])
        oldLMScores.append(out[2])

    if len(hypotheses) == 0:
        return "" if not returnConfidence else ("", 0.)
    
    acousticScores = np.array(acousticScores)
    newLMScores = np.array(rescore_with_gpt2(model, tokenizer, hypotheses, lengthPenlaty))
    oldLMScores = np.array(oldLMScores)

    totalScores = alpha * newLMScores + (1 - alpha) * oldLMScores + acousticScale * acousticScores
    maxIdx = np.argmax(totalScores)
    bestHyp = hypotheses[maxIdx]
    if not returnConfidence:
        return bestHyp
    else:
        totalScores = totalScores - np.max(totalScores)
        probs = np.exp(totalScores)
        return bestHyp, probs[maxIdx] / np.sum(probs)
    
    
def cer_with_gpt2_decoder(model, tokenizer, nbestOutputs, acousticScale,
                          inferenceOut, outputType='handwriting',
                          returnCI=False,
                          lengthPenalty=0.0,
                          alpha=1.0):
    decodedSentences = []
    confidences = []
    for i in trange(len(nbestOutputs)):
        decoded, confidence = gpt2_lm_decode(model, tokenizer, nbestOutputs[i], acousticScale, lengthPenalty, alpha, returnConfidence=True)
        decodedSentences.append(decoded)
        confidences.append(confidence)

    if outputType == 'handwriting':
        trueSentences = _extract_true_sentences(inferenceOut)
    elif outputType == 'speech' or outputType == 'speech_sil':
        trueSentences = _extract_transcriptions(inferenceOut)
    trueSentencesProcessed = []
    for trueSent in trueSentences:
        if outputType == 'handwriting':
            trueSent = trueSent.replace('>',' ')
            trueSent = trueSent.replace('~','.')
            trueSent = trueSent.replace('#','')
        if outputType == 'speech' or outputType == 'speech_sil':
            trueSent = trueSent.strip()
        trueSentencesProcessed.append(trueSent)

    cer, wer = _cer_and_wer(decodedSentences, trueSentencesProcessed, outputType, returnCI)

    return {
        'cer': cer,
        'wer': wer,
        'decoded_transcripts': decodedSentences,
        'confidences': confidences
    }
    
def rescore_with_gpt2(model, tokenizer, hypotheses, lengthPenalty):
    model_class = type(model).__name__
    if model_class.startswith('TF'):
        inputs = tokenizer(hypotheses, return_tensors='tf', padding=True)
        outputs = model(inputs)
        logProbs = tf.math.log(tf.nn.softmax(outputs['logits'], -1))
        logProbs = logProbs.cpu().numpy()
    else:
        import torch
        inputs = tokenizer(hypotheses, return_tensors='pt', padding=True)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logProbs = torch.nn.functional.log_softmax(outputs['logits'].float(), -1).cpu().numpy()

    newLMScores = []
    B, T, _ = logProbs.shape
    for i in range(B):
        n_tokens = np.sum(inputs['attention_mask'][i].cpu().numpy())

        newLMScore = 0.
        for j in range(1, n_tokens):
            newLMScore += logProbs[i, j - 1, inputs['input_ids'][i, j].cpu().numpy()]

        newLMScores.append(newLMScore - n_tokens * lengthPenalty)

    return newLMScores

def _extract_transcriptions(inferenceOut):
    transcriptions = []
    for i in range(len(inferenceOut['transcriptions'])):
        endIdx = np.argwhere(inferenceOut['transcriptions'][i] == 0)[0, 0]
        trans = ''
        for c in range(endIdx):
            trans += chr(inferenceOut['transcriptions'][i][c])
        transcriptions.append(trans)

    return transcriptions

def _extract_true_sentences(inferenceOut):
    charMarks = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                 '>',',',"'",'~','?']

    trueSentences = []
    for i in range(len(inferenceOut['trueSeqs'])):
        trueSent = ''
        endIdx = np.argwhere(inferenceOut['trueSeqs'][i] == -1)
        endIdx = endIdx[0,0]
        for c in range(endIdx):
            trueSent += charMarks[inferenceOut['trueSeqs'][i][c]]
        trueSentences.append(trueSent)

    return trueSentences

def _cer_and_wer(decodedSentences, trueSentences, outputType='handwriting',
                 returnCI=False):
    allCharErr = []
    allChar = []
    allWordErr = []
    allWord = []
    for x in range(len(decodedSentences)):
        decSent = decodedSentences[x]
        trueSent = trueSentences[x]

        nCharErr = rnnEval.wer([c for c in trueSent], [c for c in decSent])
        if outputType == 'handwriting':
            trueWords = trueSent.replace(">", " > ").split(" ")
            decWords = decSent.replace(">", " > ").split(" ")
        elif outputType == 'speech' or outputType == 'speech_sil':
            trueWords = trueSent.split(" ")
            decWords = decSent.split(" ")
        nWordErr = rnnEval.wer(trueWords, decWords)

        allCharErr.append(nCharErr)
        allWordErr.append(nWordErr)
        allChar.append(len(trueSent))
        allWord.append(len(trueWords))

    cer = np.sum(allCharErr) / np.sum(allChar)
    wer = np.sum(allWordErr) / np.sum(allWord)

    if not returnCI:
        return cer, wer
    else:
        allChar = np.array(allChar)
        allCharErr = np.array(allCharErr)
        allWord = np.array(allWord)
        allWordErr = np.array(allWordErr)

        nResamples = 10000
        resampledCER = np.zeros([nResamples,])
        resampledWER = np.zeros([nResamples,])
        for n in range(nResamples):
            resampleIdx = np.random.randint(0, allChar.shape[0], [allChar.shape[0]])
            resampledCER[n] = np.sum(allCharErr[resampleIdx]) / np.sum(allChar[resampleIdx])
            resampledWER[n] = np.sum(allWordErr[resampleIdx]) / np.sum(allWord[resampleIdx])
        cerCI = np.percentile(resampledCER, [2.5, 97.5])
        werCI = np.percentile(resampledWER, [2.5, 97.5])

        return (cer, cerCI[0], cerCI[1]), (wer, werCI[0], werCI[1])