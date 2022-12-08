#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: new_xyz(b, m, ncomponents) xyz(b, n, ncomponents)
// output: idx(b, m, nsample)
__global__ void gag_query_ball_point_kernel(int b, int n, int m,
                                            float radius, int nsample, float lambda, 
                                            int ncomponents, int component_only,
                                            const float *__restrict__ new_xyz,
                                            const float *__restrict__ xyz,
                                            int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  int offset = ncomponents + 3;

  xyz += batch_index * n * offset;
  new_xyz += batch_index * m * offset;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * offset + 0];
    float new_y = new_xyz[j * offset + 1];
    float new_z = new_xyz[j * offset + 2];
    float new_c = new_xyz[j * offset + 3];

    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * offset + 0];
      float y = xyz[k * offset + 1];
      float z = xyz[k * offset + 2];
      float c = xyz[k * offset + 3];
      
      if(component_only == 0 || c == new_c) {

        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);

        float weight = (1 - lambda);

        for (int l = 0; l < ncomponents; ++l) {
          float new_c = new_xyz[j * offset + 3 + l];

          if (new_c == c) {
            weight = lambda;
            break;
          }
        }

        d2 *= weight;

        if (d2 < radius2) {
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
}

void gag_query_ball_point_kernel_wrapper(int b, int n,
                                     int m, float radius,
                                     int nsample, float lambda, int ncomponents,
                                     int component_only,
                                     const float *new_xyz,
                                     const float *xyz, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  gag_query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, nsample, lambda, ncomponents, component_only, new_xyz, xyz, idx);

  CUDA_CHECK_ERRORS();
}


// input: unknown(b, n, ncomponents) known(b, m, ncomponents)
// output: dist2(b, n, 3), idx(b, n, 3)
__global__ void gag_three_nn_kernel(int b, int n, int m,
                                const float *__restrict__ unknown,
                                const float *__restrict__ known,
                                const float lambda,
                                const int ncomponents,
                                float *__restrict__ dist2,
                                int *__restrict__ idx) {
  int batch_index = blockIdx.x;

  int offset = ncomponents + 3;

  unknown += batch_index * n * offset;
  known += batch_index * m * offset;
  dist2 += batch_index * n * 3;
  idx += batch_index * n * 3;

  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int j = index; j < n; j += stride) {
    float ux = unknown[j * offset + 0];
    float uy = unknown[j * offset + 1];
    float uz = unknown[j * offset + 2];

    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
    int besti1 = 0, besti2 = 0, besti3 = 0;
    for (int k = 0; k < m; ++k) {
      float x = known[k * offset + 0];
      float y = known[k * offset + 1];
      float z = known[k * offset + 2];
      float c = known[j * offset + 3];

      float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
      const float weight = (1 - lambda);

      for (int l = 0; l < ncomponents; ++l) {
        float uc = unknown[j * offset + 3 + l];

        if (uc == c) {
          const float weight = lambda;
          break;
        }
      }

      d *= weight;

      if (d < best1) {
        best3 = best2;
        besti3 = besti2;
        best2 = best1;
        besti2 = besti1;
        best1 = d;
        besti1 = k;
      } else if (d < best2) {
        best3 = best2;
        besti3 = besti2;
        best2 = d;
        besti2 = k;
      } else if (d < best3) {
        best3 = d;
        besti3 = k;
      }
    }
    dist2[j * 3 + 0] = best1;
    dist2[j * 3 + 1] = best2;
    dist2[j * 3 + 2] = best3;

    idx[j * 3 + 0] = besti1;
    idx[j * 3 + 1] = besti2;
    idx[j * 3 + 2] = besti3;
  }
}

void gag_three_nn_kernel_wrapper(int b, int n, int m, const float *unknown,
                             const float *known, const float lambda, const int ncomponents,
                             float *dist2, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  gag_three_nn_kernel<<<b, opt_n_threads(n), 0, stream>>>(b, n, m, unknown, known, lambda, ncomponents,
                                                      dist2, idx);

  CUDA_CHECK_ERRORS();
}
