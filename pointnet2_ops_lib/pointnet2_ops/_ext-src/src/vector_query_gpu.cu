#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: vectors(b, m, 6) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void query_vector_point_kernel(int b, int n, int m, float radius,
                                        int nsample,
                                        const float *__restrict__ vectors,
                                        const float *__restrict__ xyz,
                                        int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  vectors += batch_index * m * 6;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;

  for (int j = index; j < m; j += stride) {
    float v1_x = vectors[j * 6 + 0];
    float v1_y = vectors[j * 6 + 1];
    float v1_z = vectors[j * 6 + 2];

    float v2_x = vectors[j * 6 + 3];
    float v2_y = vectors[j * 6 + 4];
    float v2_z = vectors[j * 6 + 5];

    float vsub_x = v2_x - v1_x;
    float vsub_y = v2_y - v1_y;
    float vsub_z = v2_z - v1_z;
    
    float vsub_norm = vsub_x * vsub_x + vsub_y * vsub_y + vsub_z * vsub_z;

    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float d = 0.0;

      // Check for V1 bound
      float sub_x = x - v1_x;
      float sub_y = y - v1_y;
      float sub_z = z - v1_z;
      float dot = sub_x * vsub_x + sub_y * vsub_y + sub_z * vsub_z;

      if(dot <= 0.0) {
        d = sub_x * sub_x + sub_y * sub_y + sub_z * sub_z;

      } else {

        // Check for V2 bound
        sub_x = x - v2_x;
        sub_y = y - v2_y;
        sub_z = z - v2_z;
        dot = sub_x * vsub_x + sub_y * vsub_y + sub_z * vsub_z;

        if(dot >= 0.0) {
          d = sub_x * sub_x + sub_y * sub_y + sub_z * sub_z;

        } else {

          // Otherwise calculate distance to the line with cross product
          float cross_x = vsub_y * sub_z - vsub_z * sub_y;
          float cross_y = vsub_z * sub_x - vsub_x * sub_z;
          float cross_z = vsub_x * sub_y - vsub_y * sub_x;

          float cross_norm = cross_x * cross_x + cross_y * cross_y + cross_z * cross_z;

          d = cross_norm / vsub_norm;
        }
      }

      if (d < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
          }
        }
        idx[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}

void query_vector_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *vectors,
                                     const float *xyz, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  query_vector_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, nsample, vectors, xyz, idx);

  CUDA_CHECK_ERRORS();
}
