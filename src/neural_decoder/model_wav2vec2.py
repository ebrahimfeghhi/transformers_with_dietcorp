import torch
from torch import nn
from .input_model_wav2vec2 import GRUInputModule
from transformers import Wav2Vec2Model
from transformers import Wav2Vec2ForCTC


class TransferModel(nn.Module):
    def __init__(
        self,
        input_model,
        pretrained_model = 'wav2vec2_phoneme',
        output_dim = None,
        device='cuda',
    ):
        super(TransferModel, self).__init__()
        self.device = device
        self.input_module = input_model.to(self.device)

        if pretrained_model == 'wav2vec2':
            #TODO: if you get an issue with the attn_implementation thing, you have to install some package to run it
            self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self",
                                                           torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(self.device)
            del self.backbone.feature_extractor
            del self.backbone.feature_projection


            #TODO: just some placeholder fc to convert from wav2vec2 outputs and convert to phonemes, needs to be changed
            self.fc_module = nn.Sequential(
                nn.Linear(self.backbone.config.output_hidden_size, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            )
        
        elif pretrained_model == 'wav2vec2_phoneme':

            #TODO: there's also other options for this on huggingface, worth looking into
            self.backbone = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(self.device)

            del self.backbone.feature_extractor
            del self.backbone.feature_projection
            
            #TODO: the pretrained phoneme model has output dim of 392 since it included non-english phonemes, 
            #this just a placeholder fc to get to our correct output dim, probably needs to be changed
            self.fc_module = nn.Linear(self.backbone.lm_head.out_features, output_dim)

        else:
            raise NotImplementedError

        for _, param in self.backbone.named_parameters():
            param.requires_grad_ = False

        
    def forward(self, neuralInput, dayIdx):
        out1 = self.input_module(neuralInput, dayIdx)
        out2 = self.backbone(out1)
        return self.fc_module(out2)