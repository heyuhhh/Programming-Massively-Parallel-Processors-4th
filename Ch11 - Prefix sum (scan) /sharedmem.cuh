

template <typename T>
struct SharedMemory {
    // Ensure that we won't compile any un-specialized types
    __device__ T *getPointer() {
        extern __device__ void error(void);
        error();
        return NULL;
    }
};

template <>
struct SharedMemory<int> {
    __device__ int *getPointer() {
        extern __shared__ int s_int[];
        return s_int;
    }
};

template <>
struct SharedMemory<float> {
    __device__ float *getPointer() {
        extern __shared__ float s_float[];
        return s_float;
    }
};