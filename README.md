# Neural Sequence Decoder

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
   - `train_model.py` – trains the original GRU-based baseline algorithm  
   - `train_full.py` – trains the Transformer-based model  
   - `train_memo.py` – performs MEMO test-time adaptation on a pretrained Transformer

4. **Evaluate with an N-gram language model**  
   Use the notebooks in `./src/neural_decoder/`:
   - `n_gram_lm.ipynb`  
   - `n_gram_lm_memo.ipynb`

   In the case that the n-gram language model does not load properly, you can import it in the following manner:
   a) HELLLOOOO

5. **Submit predictions**  
   Upload the generated `.txt` files to EvalAI for test-set WER:  
   https://eval.ai/web/challenges/challenge-page/2099/overview

6. **Generate figures and results**  
   Code to generate results/figures is in `notebooks/figures/`.
