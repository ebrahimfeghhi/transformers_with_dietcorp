from inference_functions import load_bit_phoneme_model, evaluate_model
from dataset import getDatasetLoaders

data_file = '/data2/neural_data/ptDecoder_ctc_both_char_phoneme'
trainLoaders, testLoaders, loadedData = getDatasetLoaders(
        data_file, 8, None, 
        False
    )


device = 'cuda'
bit_phoneme_filepath = "/data2/models/time_masked_transfomer_characters_phonemes_80ms_seed_0/"
model, args = load_bit_phoneme_model(bit_phoneme_filepath)
model = model.to(device)

outputs, cer, per_day_cer = evaluate_model(model, loadedData, args, partition='test', device='cuda')
