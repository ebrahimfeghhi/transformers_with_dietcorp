from inference_funcs import load_bit_phoneme_model

device = 'cuda'
bit_phoneme_filepath = "/data/willett_data/outputs/time_masked_transfomer_characters_phonemes_80ms_seed_0/"
model, args = load_bit_phoneme_model(bit_phoneme_filepath)
model = model.to(device)