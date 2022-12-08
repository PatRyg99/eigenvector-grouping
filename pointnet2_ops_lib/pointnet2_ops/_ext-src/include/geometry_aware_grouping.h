#pragma once
#include <torch/extension.h>

at::Tensor gag_ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                          const int nsample, const float lambda, const int ncomponents, const int component_only);
std::vector<at::Tensor> gag_three_nn(at::Tensor unknowns, at::Tensor knows, const float lambda, const int ncomponents);
