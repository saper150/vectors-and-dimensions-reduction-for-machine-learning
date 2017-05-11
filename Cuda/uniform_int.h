#pragma once
#include <curand.h>
#include <curand_kernel.h>

__device__ int uniform_int(
	curandState& state,
	const int high
) {
	const float randomF = curand_uniform(&state)*(high + 0.99999f);
	return (int)truncf(randomF);
}