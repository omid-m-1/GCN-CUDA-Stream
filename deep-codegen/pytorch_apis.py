import torch as th
from . import gp_apis

# SpMMv
class spmmv_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input, rowPtr, colInd, values, print_flag):
        # Dimension and degree
        V = rowPtr.shape[0] - 1
        F_in = input.shape[1]
        degrees = rowPtr[1:] - rowPtr[:-1]
        degrees[degrees == 0] = 1
        # Output tensor
        output = th.zeros_like(input, device=input.device)
        # Load custom kernel API
        output = gp_apis.spmmv(input, output, rowPtr, colInd, values, degrees, V, F_in, print_flag)
        # Cache inputs for backward step
        ctx.backward_cache = (rowPtr, colInd, values, degrees)
        return output
    @staticmethod
    def backward(ctx, dZ):
        # Load cached variables
        rowPtr, colInd, values, degrees = ctx.backward_cache
        V = rowPtr.shape[0] - 1
        F_in = dZ.shape[1]
        # Grad tensor
        grad_input = th.zeros_like(dZ, device=dZ.device)
        # Load custom kernel API
        gp_apis.spmmv(dZ, grad_input, rowPtr, colInd, values, degrees, V, F_in)
        return grad_input, None, None, None, None

# Apply SpMMv
def spmmv(input, rowPtr, colInd, values, print_flag = False):
    return spmmv_impl.apply(input, rowPtr, colInd, values, print_flag)


