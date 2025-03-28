import torch
from torch import nn
import math

class TransformerModel(nn.Module):
    def __init__(self,
                latentspeech_dim,
                output_dim,
                num_heads,
                hidden_dim,
                num_layers,
                dropout=0,
                device="cuda:2"):
        super().__init__()

        self.latentspeech_dim = latentspeech_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.device = device
        
        self.input_projection = nn.Linear(self.latentspeech_dim, self.output_dim)

        self.register_buffer("positional_embedding", self.get_sinusoidal_positional_encoding(2048, self.output_dim))

        

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.output_dim, 
            nhead=self.num_heads, 
            dim_feedforward=self.hidden_dim, 
            dropout=self.dropout, 
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
    
    def get_sinusoidal_positional_encoding(self, seq_len, model_dim):
        position = torch.arange(seq_len).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))  
        
        pe = torch.zeros(seq_len, model_dim)  
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)  
        
        return pe.unsqueeze(0)

    def forward(self, latentspeechInput):
        X = self.input_projection(latentspeechInput)
        # import ipdb; ipdb.set_trace();
        # print(X.shape)
        X = X + self.positional_embedding[:, :latentspeechInput.shape[1], :].expand(latentspeechInput.shape[0], -1, -1).to(self.device)
        out = self.transformer_encoder(X)
        return out
    