import torch
from collections import OrderedDict
import cplxmodule
from bispectral_networks.nn.layers import (
    Bispectral,
    RowNorm,
    CplxToComplex

)


class BispectralEmbedding(torch.nn.Module):
    def __init__(
        self,
        size_in,
        hdim,
        field="complex",
        constrained=False,
        bias=False,
        device="cpu",
        projection=True,
        linear_out=False,
        weight_init=cplxmodule.nn.init.cplx_trabelsi_independent_,
        name="bispectral-embedding",
    ):

        super().__init__()
        self.size_in = size_in
        self.name = name
        self.hdim = hdim
        self.field = field
        self.constrained = constrained
        self.bias = bias
        self.device = device
        self.weight_init = weight_init
        self.projection = projection
        self.linear_out = linear_out
        self.build_layers()
        
    def build_layers(self):
        layers = [
            Bispectral(
                     self.size_in,
                     self.hdim,
                     weight_init=self.weight_init,
                     device=self.device,
                ),
            RowNorm(),
            CplxToComplex()
        ]
        
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x, term=0):
        x, recon = self.layers[0].forward(x, return_inv=True)
        for layer in self.layers[1:]:
            x = layer.forward(x)
        return x, recon