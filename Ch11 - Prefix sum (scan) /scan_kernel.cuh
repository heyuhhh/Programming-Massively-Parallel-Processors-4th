#define SECTION_SIZE 256
#define COARSE_FACTOR 4

__device__
void inclusive_scan_device_BrentKung(volatile float* a, const int N, const int tid);

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