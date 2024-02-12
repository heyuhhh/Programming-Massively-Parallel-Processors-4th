#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define SECTION_SIZE 256
#define COARSE_FACTOR 4
#define TILE_SIZE (SECTION_SIZE * COARSE_FACTOR)