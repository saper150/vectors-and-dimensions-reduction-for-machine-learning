
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <thrust\device_vector.h>
#include <thrust/extrema.h>


struct subject
{
	float fitness;
	int accuracy;
	int length;
};



extern "C"
{
	__declspec(dllexport)
		int Sum(void* p, int size) {
		int* toSum = (int*)p;
		thrust::device_ptr<int> d(toSum);
		return thrust::reduce(d, d + size);
	}




	__declspec(dllexport)
		float Avrage(void* p, int size) {
		int* toSum = (int*)p;
		thrust::device_ptr<int> devicePtr(toSum);
		return thrust::reduce(devicePtr, devicePtr + size)/(float)size;

	}

	__declspec(dllexport)
		float Maxf(void* p, int size) {
		thrust::device_ptr<float> devicePtr((float*)p);
		return *thrust::max_element(devicePtr, devicePtr + size);
	}

	__declspec(dllexport)
		int Maxi(void* p, int size) {
		thrust::device_ptr<int> devicePtr((int*)p);
		return *thrust::max_element(devicePtr, devicePtr + size);
	}

	__declspec(dllexport)
		int Minf(void* p, int size) {
		thrust::device_ptr<float> devicePtr((float*)p);
		return *thrust::min_element(devicePtr, devicePtr + size);
	}
	__declspec(dllexport)
		int Mini(void* p, int size) {
		thrust::device_ptr<int> devicePtr((int*)p);
		return *thrust::min_element(devicePtr, devicePtr + size);
	}




	__declspec(dllexport)
		subject FindFitest(
			float* fitnesses,
			int* accuracyes,
			int* lengths,
			int size
		) {

		subject s;
		thrust::device_ptr<float> deviceFitness(fitnesses);
		auto max = thrust::max_element(deviceFitness, deviceFitness + size);
		s.fitness = *max;

		int position = max - deviceFitness;
		cudaMemcpy(&s.accuracy, accuracyes + position, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&s.length, lengths + position, sizeof(int), cudaMemcpyDeviceToHost);

		return s;
	}




}