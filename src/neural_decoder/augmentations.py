import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


import numpy as np

def mask_electrodes(X, max_mask_size):
    
    X = X.clone()
    
    batch_size, _, _  = X.shape
    
    area_6v_superior = np.array([
    [62,  51,  43,  35,  94,  87,  79,  78],
    [60,  53,  41,  33,  95,  86,  77,  76],
    [63,  54,  47,  44,  93,  84,  75,  74],
    [58,  55,  48,  40,  92,  85,  73,  72],
    [59,  45,  46,  38,  91,  82,  71,  70],
    [61,  49,  42,  36,  90,  83,  69,  68],
    [56,  52,  39,  34,  89,  81,  67,  66],
    [57,  50,  37,  32,  88,  80,  65,  64]
    ])

    area_6v_inferior = np.array([
        [125, 126, 112, 103,  31,  28,  11,  8],
        [123, 124, 110, 102,  29,  26,   9,  5],
        [121, 122, 109, 101,  27,  19,  18,  4],
        [119, 120, 108, 100,  25,  15,  12,  6],
        [117, 118, 107,  99,  23,  13,  10,  3],
        [115, 116, 106,  97,  21,  20,   7,  2],
        [113, 114, 105,  98,  17,  24,  14,  0],
        [127, 111, 104,  96,  30,  22,  16,  1]
    ])
        
    for b in range(batch_size):
        
        M = np.random.randint(0, max_mask_size+1)
        
        if M > 0:
            
            masked_indices = return_mask_electrodes_optimized(M)
            rows, cols = np.array(masked_indices).T  # Shape (2, M)
            superior_masked_indices = area_6v_superior[rows, cols]
            inferior_masked_indices = area_6v_inferior[rows, cols]
            masked_channels = np.concatenate((superior_masked_indices, inferior_masked_indices))
            masked_channels_all = np.concatenate((masked_channels, masked_channels+128))
            X[b, :, masked_channels_all] = 0
            
    return X

def return_mask_electrodes_optimized(M, grid_size=8):
    """
    Optimized electrode masking with vectorized operations.
    
    Args:
        M (int): Number of electrodes to mask
        grid_size (int): Size of square grid (default 8x8)
        
    Returns:
        ndarray: Masked electrode indices sorted by distance
    """
    # Precompute grid coordinates using broadcasting
    rows, cols = np.divmod(np.arange(grid_size**2), grid_size)
    
    # Random center selection
    center_idx = np.random.randint(grid_size**2)
    
    # Vectorized distance calculation
    distances = np.hypot(rows - rows[center_idx], 
                        cols - cols[center_idx])
    
    # Create mask excluding center and sort
    mask = np.ones(grid_size**2, bool)
    valid_indices = np.where(mask)[0]
    
    # Sort with tie-breaking using 64-bit precision
    sorted_indices = valid_indices[
        np.lexsort((np.random.random(len(valid_indices)),  # Tiebreaker
                   distances[valid_indices]))
    ]
    
    return [(idx // grid_size, idx % grid_size) for idx in sorted_indices[:M]]

class WhiteNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise

class MeanDriftNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        _, C = x.shape
        noise = torch.randn(1, C) * self.std
        return x + noise

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding="same")