#pragma once
#include <torch/extension.h>

at::Tensor vector_query(at::Tensor vectors, at::Tensor xyz, const float radius,
                        const int nsample);