from transformers import AutoTokenizer, AutoModelForCausalLM
import os, torch
import numpy as np
device = 'cuda:2'

os.environ["TOKENIZERS_PARALLELISM"] = "false"   # kill the fork warning
model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype="auto"
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

breakpoint()


from neural_decoder.llm_utils import cer_with_gpt2_decoder
import time

import pickle
nbest_path = "/data/willett_data/model_transcriptions/neurips_transformer_time_masked_seed_0_nbest.pkl"
with open(nbest_path, mode = 'rb') as f:
    nbest = pickle.load(f)
    
model_outputs_path = "/data/willett_data/model_transcriptions/neurips_transformer_time_masked_seed_0_model_outputs.pkl"
with open(model_outputs_path, mode = 'rb') as f:
    model_outputs = pickle.load(f)
    
    
for i in range(len(model_outputs['transcriptions'])):
    new_trans = [ord(c) for c in model_outputs['transcriptions'][i]] + [0]
    model_outputs['transcriptions'][i] = np.array(new_trans)
    
acoustic_scale = 0.8
llm_weight = 0.5

# Rescore nbest outputs with LLM
start_t = time.time()
llm_out = cer_with_gpt2_decoder(
    model,
    tokenizer,
    nbest[:],
    acoustic_scale,
    model_outputs,
    outputType="speech_sil",
    returnCI=True,
    lengthPenalty=0,
    alpha=llm_weight,
)

print(llm_out["cer"], llm_out["wer"])
