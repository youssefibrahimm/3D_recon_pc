import torch.nn as nn
import torch.nn.functional as F
class Decoder(nn.Module):
    def __init__(
        self, 
        latent_size: int, 
        point_size: int, 
        hidden_sizes: list[int] = [1024, 2048, 4096], 
        dropout: float = 0.3, 
        activation_fn: type[nn.Module] = nn.ReLU
    ) -> None:
        """
        Initialize the Decoder module.

        Parameters:
        latent_size (int): The size of the latent space.
        point_size (int): The number of points in the output point cloud.
        hidden_sizes (list[int]): The number of neurons in each fully connected layer.
        dropout (float): The dropout probability. Defaults to 0.3.
        activation_fn (type[nn.Module]): The activation function class used in each layer. Defaults to nn.ReLU.
        """
        super().__init__()
        assert len(hidden_sizes) > 0, f'Hidden_sizes must be non-empty, but got {hidden_sizes}'
        assert point_size*3 >= hidden_sizes[-1], f'Last layer must have at least as many neurons as the number of points, but got {hidden_sizes[-1]} and {point_size*3}'
        self._latent_size = latent_size
        self._point_size = point_size
        self._dropout = dropout
        self._activation_fn = activation_fn
        sizes = [latent_size] + hidden_sizes + [point_size * 3]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.Dropout(dropout))
            layers.append(activation_fn())
        self._fc_layers = nn.Sequential(*layers)

    def forward(self, input):
        x = input
        x = self._fc_layers(x)
        return x.view(-1, self._point_size, 3)
