
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <thrust\device_vector.h>
#include <thrust/extrema.h>
#include <thrust/system/cuda/execution_policy.h>



struct subject
{
	float fitness;
	int index;
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
		float Avragei(int* p, int size) {
		thrust::device_ptr<int> devicePtr(p);
		return thrust::reduce(devicePtr, devicePtr + size)/(float)size;

	}
	__declspec(dllexport)
		float Avragef(float* p, int size) {
		thrust::device_ptr<float> devicePtr(p);
		return thrust::reduce(devicePtr, devicePtr + size) / (float)size;

	}

	__declspec(dllexport)
		float Maxf(float* p, int size) {
		thrust::device_ptr<float> devicePtr(p);
		return *thrust::max_element(devicePtr, devicePtr + size);
	}

	__declspec(dllexport)
		int Maxi(int* p, int size) {
		thrust::device_ptr<int> devicePtr(p);
		return *thrust::max_element(devicePtr, devicePtr + size);
	}

	__declspec(dllexport)
		int Minf(float* p, int size) {
		thrust::device_ptr<float> devicePtr(p);
		return *thrust::min_element(devicePtr, devicePtr + size);
	}
	__declspec(dllexport)
		int Mini(int* p, int size) {
		thrust::device_ptr<int> devicePtr(p);
		return *thrust::min_element(devicePtr, devicePtr + size);
	}

	__declspec(dllexport) 
		void sort_by_key(float* keys,int* values,int size) {
		thrust::device_ptr<float> deviceKeys(keys);
		thrust::device_ptr<int> deviceValues(values);
		thrust::sort_by_key(deviceKeys, deviceKeys + size, deviceValues,thrust::less<float>());
	}

	__declspec(dllexport)
		void sort_by_keyDesc(float* keys, int* values, int size) {
		thrust::device_ptr<float> deviceKeys(keys);
		thrust::device_ptr<int> deviceValues(values);
		thrust::sort_by_key(deviceKeys, deviceKeys + size, deviceValues, thrust::greater<float>());
	}

	__declspec(dllexport)
		void sequence(int* values,int size) {
		thrust::device_ptr<int> deviceValues(values);
		thrust::sequence(deviceValues, deviceValues + size);
	}

	__declspec(dllexport)
		void sort_by_key_multiple(float* keys, int* values,int col,int row) {
		thrust::device_ptr<float> deviceKeys(keys);
		thrust::device_ptr<int> deviceValues(values);

		std::vector<cudaStream_t> streams;
		streams.resize(row);
		for (int i = 0; i < row; i++)
		{
			cudaStreamCreate(&streams[i]);
		}

		for (int i = 0; i < row; i++)
		{
			const auto currentKeys = deviceKeys + i*col;
			const auto currentValues = deviceValues+ i*col;
			thrust::sort_by_key(thrust::cuda::par.on(streams[i]), currentKeys, currentKeys + col, currentValues);
		}

		for (auto& stream : streams) {
			cudaStreamSynchronize(stream);
		}
		for (auto& stream : streams) {
			cudaStreamDestroy(stream);
		}

	}



	__declspec(dllexport)
		subject FindFitest(
			float* fitnesses,
			int* sortedIndeces,
			int size
		) {

		subject s;
		cudaMemcpy(&s.index, sortedIndeces+size-1, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&s.fitness, fitnesses+size-1, sizeof(float), cudaMemcpyDeviceToHost);

		return s;
	}

}