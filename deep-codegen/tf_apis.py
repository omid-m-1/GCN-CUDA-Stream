import tensorflow as tf
import gp_apis

def spmmv(input, dim_0, dim_1, rowPtr, colInd, values, degrees, device0):
    @tf.custom_gradient
    def _lambda(X):
        return spmmv_real(X, dim_0, dim_1, rowPtr, colInd, values, degrees, device0)
    return _lambda(input)

def spmmv_real(input, dim_0, dim_1, rowPtr, colInd, values, degrees, device0):
    out = gp_apis.gp_spmmv(input, dim_0, dim_1, rowPtr, colInd, values, degrees, device0)
    def grad(dZ):
        return gp_apis.gp_spmmv(dZ, dim_0, dim_1, rowPtr, colInd, values, degrees, device0)
    return out, grad

