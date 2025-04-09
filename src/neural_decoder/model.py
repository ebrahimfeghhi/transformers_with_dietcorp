import torch
from torch import nn

from .augmentations import GaussianSmoothing
import torch.nn.functional as F

class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        #self.gaussianSmoother = GaussianSmoothing(
        #    neural_dim, 20, self.gaussianSmoothWidth, dim=1
        #)
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # GRU layers
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
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

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes)  # +1 for CTC blank

    def extract_last_k_of_chunks(self, x: torch.Tensor, chunk_size: int = 32, last_k: int = 4) -> torch.Tensor:
        """
        Given a tensor of shape (B, T, N), extract the last `last_k` features
        from each `chunk_size`-sized chunk along the last dimension (N).

        Returns a tensor of shape (B, T, num_chunks * last_k).
        """
        B, T, N = x.shape
        if N % chunk_size != 0:
            raise ValueError(f"N={N} must be divisible by chunk_size={chunk_size}")
        
        num_chunks = N // chunk_size
        
        # Reshape to separate chunks
        x_chunks = x.view(B, T, num_chunks, chunk_size)
        
        # Extract last `last_k` features from each chunk
        last_k_features = x_chunks[:, :, :, -last_k:]  # shape: (B, T, num_chunks, last_k)
        
        # Flatten chunk and feature dimensions back together
        return last_k_features.reshape(B, T, num_chunks * last_k)
    
    def forward(self, neuralInput, dayIdx):
        
        # neuralInput is of shape nBatch x SeqLen x NeuralFeats
        
        # (nBatch x SeqLen x NeuralFeats) -> (nBatch x NeuralFeats x SeqLen) 
        #neuralInput = torch.permute(neuralInput, (0, 2, 1))
        #neuralInput = self.gaussianSmoother(neuralInput)
        #neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        # transformedNeural is of shape Batch Size x Timesteps x Num Features 
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )
        
        outputs = self.extract_last_k_of_chunks(stridedInputs)
        stridedInputs = stridedInputs[:, :-1, :]
        outputs = outputs[:, 1:]
        
        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # get seq
        seq_out = self.fc_decoder_out(hid)
        
        recon_loss = F.mse_loss(seq_out, outputs)
        
        return recon_loss
