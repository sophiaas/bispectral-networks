import torch
import numpy as np
import cplxmodule
from cplxmodule import Cplx
from cplxmodule.nn import CplxParameter
from .functional import linear, linear_conjtx


class RowNorm(torch.nn.Module):
    def forward(self, x):
        x = x - torch.mean(x.real + 1j * x.imag, axis=-1, keepdim=True)
        x = x / torch.linalg.norm(x.real + 1j * x.imag, axis=-1, keepdim=True)
        return Cplx(x.real, x.imag)
    
    
class CplxToComplex(torch.nn.Module):
    def forward(self, x):
        return x.real + 1j * x.imag
    

class Bispectral(torch.nn.Module):
    def __init__(
        self,
        size_in,
        size_out,
        weight_init=cplxmodule.nn.init.cplx_trabelsi_independent_,
        device="cpu",
    ):
        super().__init__()

        self.size_in, self.size_out = size_in, size_out
        self.device = device
        self.weight_init = weight_init
    
        self.reset_parameters()

    def forward(self, x, return_inv=False):
        return self.forward_(x, return_inv=return_inv)

    def reset_parameters(self):
        self.reset_parameters_()

    def reset_parameters_(self):
        size_out = self.size_out
        size_in = self.size_in

        self.W = Cplx.empty(size_out, size_in).to(self.device)
        self.weight_init(self.W)
        self.W = CplxParameter(self.W)

    def forward_(self, x, return_inv=False):
        if type(x) != Cplx:
            x = x.type(self.W.data.dtype)
            x = Cplx(x)
        if return_inv:
            l, l_inv = self.forward_linear(x, return_inv=return_inv)     
        else:
            l = self.forward_linear(x)
                    
        l_ = l.real + 1j * l.imag
        l_ = l_.unsqueeze(-1)
        l_cross = torch.matmul(l_, torch.swapaxes(l_, 1, -1))
        l_cross = l_cross.reshape(l.shape[0], -1)
        l_cross = Cplx(l_cross.real, l_cross.imag)

        W_ = self.W.real + 1j * self.W.imag
        all_crosses = (W_[:, None, :]  * W_[None, :, :]).conj()
        all_crosses = all_crosses.reshape((-1, self.size_out)).to(x.device)
        all_crosses = Cplx(all_crosses.real, all_crosses.imag)
                
        conj_term = linear(x, all_crosses)
        out = l_cross * conj_term
        
        # Take only upper triangular
        out = out.reshape(-1, self.size_out, self.size_out)
        idxs = np.triu_indices(self.size_out, k=0, m=None)
        out = out[:, idxs[0], idxs[1]]
            
        if return_inv:
            return out, l_inv
        else:
            return out

    def forward_linear(self, x, return_inv=False):
        if type(x) != Cplx:
            x = x.type(self.W.data.dtype)
            x = Cplx(x)
            
        l = linear(x, self.W)
            
        l_inv = linear_conjtx(self.W, l).real.T

        if return_inv:
            return l, l_inv
        else:
            return l