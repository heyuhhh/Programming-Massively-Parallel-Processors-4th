#include "scan_kernel.cuh"

void scan(float* out, float* in, const int N, int mode);

void inclusive_scan(float* out, float* in, const int N);

void exclusive_scan(float* out, float* in, const int N);

void scan(int* out, int* in, const int N, int mode);

void inclusive_scan(int* out, int* in, const int N);

void exclusive_scan(int* out, int* in, const int N);