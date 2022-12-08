#include "geometry_aware_grouping.h"
#include "utils.h"

void gag_query_ball_point_kernel_wrapper(int b, int n, int m,
                                         float radius, int nsample, float lambda,
                                         int ncomponents, int component_only,
                                         const float *new_xyz,
                                         const float *xyz, int *idx);

void gag_three_nn_kernel_wrapper(int b, int n, int m,
                                 const float *unknown, const float *known,
                                 const float lambda, const int ncomponents,
                                 float *dist2, int *idx);

at::Tensor gag_ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                          const int nsample, const float lambda, const int ncomponents, const int component_only)
{
    CHECK_CONTIGUOUS(new_xyz);
    CHECK_CONTIGUOUS(xyz);
    CHECK_IS_FLOAT(new_xyz);
    CHECK_IS_FLOAT(xyz);

    if (new_xyz.is_cuda())
    {
        CHECK_CUDA(xyz);
    }

    at::Tensor idx =
        torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                     at::device(new_xyz.device()).dtype(at::ScalarType::Int));

    if (new_xyz.is_cuda())
    {
        gag_query_ball_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                            radius, nsample, lambda, ncomponents, component_only,
                                            new_xyz.data_ptr<float>(),
                                            xyz.data_ptr<float>(), idx.data_ptr<int>());
    }
    else
    {
        AT_ASSERT(false, "CPU not supported");
    }

    return idx;
}

std::vector<at::Tensor> gag_three_nn(at::Tensor unknowns, at::Tensor knows, const float lambda, const int ncomponents)
{
    CHECK_CONTIGUOUS(unknowns);
    CHECK_CONTIGUOUS(knows);
    CHECK_IS_FLOAT(unknowns);
    CHECK_IS_FLOAT(knows);

    if (unknowns.is_cuda())
    {
        CHECK_CUDA(knows);
    }

    at::Tensor idx =
        torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                     at::device(unknowns.device()).dtype(at::ScalarType::Int));
    at::Tensor dist2 =
        torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                     at::device(unknowns.device()).dtype(at::ScalarType::Float));

    if (unknowns.is_cuda())
    {
        gag_three_nn_kernel_wrapper(unknowns.size(0), unknowns.size(1), knows.size(1),
                                    unknowns.data_ptr<float>(), knows.data_ptr<float>(),
                                    lambda, ncomponents,
                                    dist2.data_ptr<float>(), idx.data_ptr<int>());
    }
    else
    {
        AT_ASSERT(false, "CPU not supported");
    }

    return {dist2, idx};
}