import torch.nn.functional as F
import torch
from cplxmodule import Cplx
from cplxmodule.nn import CplxParameter


def linear(x, W, b=None):
    if type(x) == Cplx or type(x) == CplxParameter or type(W) == Cplx or type(W) == CplxParameter:
        re = F.linear(x.real, W.real) - F.linear(x.imag, W.imag)
        if b is not None:
            re = re + b.real
        im = F.linear(x.real, W.imag) + F.linear(x.imag, W.real)
        if b is not None:
            im = im + b.imag
        out = Cplx(re, im)
    else:
        out = F.linear(x, W, b)
    return out


def linear_conjtx(x, W, b=None):
    re = F.linear(x.real.T, W.real) - F.linear(-x.imag.T, W.imag)
    if b is not None:
        re = re + b.real
    im = F.linear(x.real.T, W.imag) + F.linear(-x.imag.T, W.real)
    if b is not None:
        im = im + b.imag
    out = Cplx(re, im)
    return out
