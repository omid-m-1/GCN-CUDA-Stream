import tensorlow as tf
import kernel as gpk
def gp_spmmv(X, dim_0, dim_1, rowPtr, colInd, values, degrees):
    X_dl = tf.experimental.dlpack.to_dlpack(X)
    rowPtr_dl = tf.experimental.dlpack.to_dlpack(rowPtr)
    colInd_dl = tf.experimental.dlpack.to_dlpack(colInd)
    values_dl = tf.experimental.dlpack.to_dlpack(values)
    degrees_dl = tf.experimental.dlpack.to_dlpack(degrees)
    #declare the output tensor here
    res = tf.zeros([dim_0, dim_1])
    res_dl = tf.experimental.dlpack.to_dlpack(res)
    gpk.spmmv(X_dl, res_dl, rowPtr_dl, colInd_dl, values_dl, degrees_dl)
    return res
