import nemo
import nemo, nemo.collections.asr as nemo_asr
citrinet = nemo_asr.models.EncDecCTCModelBPE.from_pretrained('stt_en_citrinet_512')