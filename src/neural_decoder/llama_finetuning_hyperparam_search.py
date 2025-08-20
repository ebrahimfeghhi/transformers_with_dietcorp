from unsloth import FastLanguageModel
import torch
import numpy as np
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import train_on_responses_only

def compute_wer(r, h):
    """
    Calculation of WER with Levenshtein distance.
    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.
    Parameters
    ----------
    r : list
    h : list
    Returns
    -------
    int
    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

def _cer_and_wer(decodedSentences, trueSentences, outputType='speech',
                 returnCI=False):
    allCharErr = []
    allChar = []
    allWordErr = []
    allWord = []
    for x in range(len(decodedSentences)):
        decSent = decodedSentences[x]
        trueSent = trueSentences[x]

        nCharErr = compute_wer([c for c in trueSent], [c for c in decSent])
        if outputType == 'handwriting':
            trueWords = trueSent.replace(">", " > ").split(" ")
            decWords = decSent.replace(">", " > ").split(" ")
        elif outputType == 'speech' or outputType == 'speech_sil':
            trueWords = trueSent.split(" ")
            decWords = decSent.split(" ")
        nWordErr = compute_wer(trueWords, decWords)

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


def train_model(
    metrics_save_path,
    model_save_path,
    tokenizer_save_path,
    r=16,
    lora_alpha=16,
    learning_rate=2e-4,
    num_train_epochs=1,
    batch_size=16,
    gradient_accumulation_steps=4,
):

    max_seq_length = None
    dtype = None
    load_in_4bit = True

    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    EOS_TOKEN = tokenizer.eos_token

    base_path = '/home/ubuntu/transformers_with_dietcorp/src/neural_decoder/jsonl_files/'
    data_files = {
        'val_no_gt': f"{base_path}val_no_gt.jsonl",
        'val': f"{base_path}val.jsonl",
        'train': f"{base_path}train.jsonl",
        'test': f"{base_path}eval.jsonl"
    }

    dataset = load_dataset("json", data_files=data_files)

    def formatting_func(examples):
        texts = []
        for p, c in zip(examples["prompt"], examples["completion"]):
            user_text = p[0]["content"]
            assistant_text = c[0]["content"]
            text = (
                "<|start_header_id|>user<|end_header_id|>\n\n"
                + user_text
                + "\n<|start_header_id|>assistant<|end_header_id|>\n\n"
                + assistant_text
                + EOS_TOKEN
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        dataset_text_field="text",
        packing=False,
        args=SFTConfig(
            completion_only_loss=True,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            eval_steps=50,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    trainer_stats = trainer.train()

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)

    FastLanguageModel.for_inference(model)

    batch_size = 1
    split = "val_no_gt"

    last_lines_val = []
    n = len(dataset[split])

    for batch_idx in range(880):
        if batch_idx % 100 == 0:
            print(batch_idx)

        # Grab a batch of texts
        batch = dataset[split][batch_idx]
        val_texts = batch["text"]

        # Tokenize the batch
        inputs = tokenizer(
            val_texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to('cuda')

        # Generate
        outputs = model.generate(
            **inputs,
            use_cache=True, 
            max_new_tokens=400
        )

        # Decode all outputs
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Keep just the last non-empty line from each sequence
        for seq in decoded:
            lines = [l.strip() for l in seq.splitlines() if l.strip()]
            last_line = lines[-1] if lines else ""
            last_lines_val.append(last_line)


    with open("/home/ubuntu/data/model_transcriptions/txt_files/ground_truth_sentences.txt", "r", encoding="utf-8") as f:
        val_gt_lines = [line.strip() for line in f]

    metrics = _cer_and_wer(last_lines_val, val_gt_lines)

    with open(metrics_save_path, "w") as f:
        f.write(str(metrics))

    return metrics
