# Masked Transformers with Test-Time Adaptation for Speech Neuroprostheses

Adapted from: https://github.com/cffan/neural_seq_decoder/tree/master

## Requirements
- Python ≥ 3.9 (recommend using Python version 3.9.21)

## Installation

For the original install environment, run: 

```bash
pip install -e . 
```

Additional packages used for this paper can be found in the environment.yml file.


## How to Run

1. **Download the neural dataset**  
   https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq

2. **Format the dataset**  
   Convert the speech BCI dataset using `notebooks/formatCompetitionData.ipynb`

3. **Train the models**  
   Run the Python scripts in `./scripts/`:
   - `train_gru.py` – trains the original GRU-based baseline algorithm  
   - `train_transformer.py` – trains the Transformer-based model  
   - `train_cort.py` – performs MEMO style test-time adaptation (**C**ontinual **O**nline **R**ecalibration with **T**ime Masking) on a pretrained Transformer
   - `train_mae.py` – trains a masked autoencoder to predict masked tokens


4. **Evaluate with an N-gram language model**  
   Use the notebooks in `./src/neural_decoder/` to generate the competition `.txt` files :
   - `n_gram_lm.ipynb`  
   - `n_gram_lm_cort.ipynb`

   In order to load the n-gram language model, follow these steps:
   1. git clone https://github.com/fwillett/speechBCI
   2. cd speechBCI/LanguageModelDecoder/runtime/server/x86
   3. Activate virtual environment used for this repo. 
   4. python setup.py install
   5. find build \\( -name "lm_decoder*.so" -o -name "lm_decoder*.pyd" \\)
   6. Move the file to your desired location, and rename it "lm_decoder.so". Recommend not including it in git repo, because the build depends on each system.
   7. Change file path in lm_utils to where lm_decoder.so is located. 

5. **Submit predictions**  
   Upload the generated `.txt` files to EvalAI for test-set WER:  
   https://eval.ai/web/challenges/challenge-page/2099/overview

6. **Generate figures and results**  
   Code to generate results/figures is in `notebooks/figures/`.