#pragma once
#include "csr.h"
#include "op.h"

void spmmv(array2d_t<float>& input, array2d_t<float>& output, array1d_t<int>& rowPtr, array1d_t<int>& colInd, array1d_t<float>& values, array1d_t<int>& degrees, int V, int F_in, int64_t stream_id, bool print_stream);
