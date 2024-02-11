#include "../utils/config.cuh"

__device__
void inclusive_scan_device_BrentKung(volatile float* a, const int N, const int tid);

__device__
void exclusive_scan_device_BrentKung(volatile float* a, const int N, const int tid);

__global__
void scan_SinglePass(float* a, float* flag, float* aux, const int N, int mode);

__global__
void scan_KoggleStone(float* a, const int N, int mode);

__global__
void scan_BrentKung(float* a, const int N, int mode);

__global__
void addLastElement(float* a, float* b, const int N);

__global__
void scan_on_block(float* a, float* b, const int N, int mode);



// int

__device__
void inclusive_scan_device_BrentKung(volatile int* a, const int N, const int tid);

__device__
void exclusive_scan_device_BrentKung(volatile int* a, const int N, const int tid);

__global__
void scan_SinglePass(int* a, int* flag, int* aux, const int N, int mode);

__global__
void scan_KoggleStone(int* a, const int N, int mode);

__global__
void scan_BrentKung(int* a, const int N, int mode);

__global__
void addLastElement(int* a, int* b, const int N);

__global__
void scan_on_block(int* a, int* b, const int N, int mode);