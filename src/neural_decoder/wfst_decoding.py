from nemo.collections.asr.parts.submodules.wfst_decoder import RivaGpuWfstDecoder

langugage_model_fst_path = "/data/lm/TLG.mixed_lm.3-gram.pruned.3e-7.fst"

decoder = RivaGpuWfstDecoder(lm_fst=langugage_model_fst_path, decoding_mode="nbest", beam_size=16, lm_weight=0.2, nbest_size=10)

breakpoint()