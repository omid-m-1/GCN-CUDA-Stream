inline void export_kernel(py::module &m) { 
    m.def("spmmv",[](py::capsule& input, py::capsule& output, py::capsule& rowPtr, py::capsule& colInd, py::capsule& values, py::capsule& degrees, int V, int F_in, int64_t stream_id, bool print_stream){
        array2d_t<float> input_array = capsule_to_array2d<float>(input);
        array2d_t<float> output_array = capsule_to_array2d<float>(output);
        array1d_t<int> rowPtr_array = capsule_to_array1d<int>(rowPtr);
        array1d_t<int> colInd_array = capsule_to_array1d<int>(colInd);
        array1d_t<float> values_array = capsule_to_array1d<float>(values);
        array1d_t<int> degrees_array = capsule_to_array1d<int>(degrees);
    return spmmv(input_array, output_array, rowPtr_array, colInd_array, values_array, degrees_array, V, F_in, stream_id, print_stream);
    }
  );
}
