/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>

#include <algorithm>
#include <time.h>
#include <limits.h>

template <typename T, bool floatKeys>
bool testSort(int argc, char **argv) {
  int cmdVal;
  int keybits = 32;

  unsigned int numElements = 10000000;
  bool keysOnly = true;
  bool quiet = true;

  unsigned int numIterations = (numElements >= 16777216) ? 10 : 100;
  
  thrust::host_vector<T> h_keys(numElements);
  thrust::host_vector<T> h_keysSorted(numElements);
  thrust::host_vector<unsigned int> h_values;

  if (!keysOnly) h_values = thrust::host_vector<unsigned int>(numElements);

  // Fill up with some random data
  thrust::default_random_engine rng(clock());

  if (floatKeys) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    for (int i = 0; i < (int)numElements; i++) h_keys[i] = u01(rng);
  } else {
    thrust::uniform_int_distribution<unsigned int> u(0, UINT_MAX);

    for (int i = 0; i < (int)numElements; i++) h_keys[i] = u(rng);
  }

  if (!keysOnly) thrust::sequence(h_values.begin(), h_values.end());

  // Copy data onto the GPU
  thrust::device_vector<T> d_keys;
  thrust::device_vector<unsigned int> d_values;

  // run multiple iterations to compute an average sort time
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);

  float totalTime = 0;

  for (unsigned int i = 0; i < numIterations; i++) {
    // reset data before sort
    d_keys = h_keys;

    if (!keysOnly) d_values = h_values;

    cudaEventRecord(start_event, 0);

    if (keysOnly)
      thrust::sort(d_keys.begin(), d_keys.end());
    else
      thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    float time = 0;
    cudaEventElapsedTime(&time, start_event, stop_event);
    totalTime += time;
  }

  totalTime /= (1.0e3f * numIterations);
  printf(
      "radixSortThrust, Throughput = %.4f MElements/s, Time = %.5f s, Size = "
      "%u elements\n",
      1.0e-6f * numElements / totalTime, totalTime, numElements);


  // Get results back to host for correctness checking
  thrust::copy(d_keys.begin(), d_keys.end(), h_keysSorted.begin());

  if (!keysOnly)
    thrust::copy(d_values.begin(), d_values.end(), h_values.begin());


  // Check results
  bool bTestResult =
      thrust::is_sorted(h_keysSorted.begin(), h_keysSorted.end());

  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);

  if (!bTestResult && !quiet) {
    return false;
  }

  return bTestResult;
}

int main(int argc, char **argv) {
  // Start logs
  printf("%s Starting...\n\n", argv[0]);

  bool bTestResult = false;

  bTestResult = testSort<unsigned int, false>(argc, argv);

  printf(bTestResult ? "Test passed\n" : "Test failed!\n");
}