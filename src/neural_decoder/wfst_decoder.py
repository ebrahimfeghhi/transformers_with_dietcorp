from nemo.collections.asr.parts.submodules.wfst_decoder import RivaGpuWfstDecoder

langugage_model_fst_path = "/data/code/nejm-brain-to-text/language_model/pretrained_language_models/openwebtext_1gram_lm_sil/TLG_with_symbols.fst"

decoder = RivaGpuWfstDecoder(lm_fst=langugage_model_fst_path, decoding_mode="nbest", 
                             beam_size=16, lm_weight=0.2, nbest_size=10)
breakpoint()