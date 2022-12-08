#include "vector_query.h"
#include "utils.h"

void query_vector_point_kernel_wrapper(int b, int n, int m, float radius,
                                       int nsample, const float *vectors,
                                       const float *xyz, int *idx);

at::Tensor vector_query(at::Tensor vectors, at::Tensor xyz, const float radius,
                        const int nsample)
{
    CHECK_CONTIGUOUS(vectors);
    CHECK_CONTIGUOUS(xyz);
    CHECK_IS_FLOAT(vectors);
    CHECK_IS_FLOAT(xyz);

    if (vectors.is_cuda())
    {
        CHECK_CUDA(xyz);
    }

    at::Tensor idx =
        torch::zeros({vectors.size(0), vectors.size(1), nsample},
                     at::device(vectors.device()).dtype(at::ScalarType::Int));

    if (vectors.is_cuda())
    {
        query_vector_point_kernel_wrapper(xyz.size(0), xyz.size(1), vectors.size(1),
                                          radius, nsample, vectors.data_ptr<float>(),
                                          xyz.data_ptr<float>(), idx.data_ptr<int>());
    }
    else
    {
        AT_ASSERT(false, "CPU not supported");
    }

    return idx;
}