import torch
from torch import nn

class Speech2NeuralDecoder(nn.Module):
    def __init__(
        self,
        latentspeech_dim,
        output_dim,
        hidden_dim,
        layer_dim,
        dropout=0,
        device="cuda",
        strideLen=1,
        kernelLen=4,
        bidirectional=True,
    ):
        super(Speech2NeuralDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.latentspeech_dim = latentspeech_dim
        self.output_dim = output_dim
        self.device = device
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.kernelLen = kernelLen
        self.strideLen = strideLen
        
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )

        # GRU layers
        self.gru_decoder = nn.GRU(
            (latentspeech_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

     
        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, output_dim
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, output_dim)  # +1 for CTC blank

    def forward(self, latentspeechInput):
        
        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(latentspeechInput, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        hid, _ = self.gru_decoder(stridedInputs)

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out
