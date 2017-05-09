#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "HeapData.h"
#include <math.h>

extern "C" {

	__constant__ int popSize;
	__constant__ int atributeCount;
	__constant__ int teachingVectorsCount;
	__constant__ int testVectorsCount;
	__constant__ int k;
	__constant__ int countToPass;


	__constant__ float alpha;

	__global__ void fitnessFunction(
		float* accuracies,
		float avrageAccuracy,
		int* vectorSizes,
		float avrageVectorSize,
		float* fitness
	) {

		const int id = threadIdx.x;

		if (vectorSizes[id] == 0) {
			fitness[id] = 0.f;
			return;
		}

		const float a = alpha * accuracies[id] / avrageAccuracy;
		const float b = (1 - alpha) * avrageVectorSize / vectorSizes[id];
		const float res = a + b;
		fitness[id] = res*res;
	
	}



	__global__ void geneticKnn(
		const float* testVectors,
		const int* testClasses,
		const float* teachingVectors,
		const int* teachingClasses,
		const unsigned char* population,
		float* accuracy

	)
	{

		const int id = threadIdx.x + blockIdx.x * blockDim.x;
		if (id >= testVectorsCount) return;

		const unsigned char* currentGen = population + atributeCount*blockIdx.y;
		const float* currentVector = testVectors + atributeCount*id;

		HeapData* heap = new HeapData[k];

		initializeHeap(heap, k);

		for (int i = 0; i < teachingVectorsCount; i++)
		{
			const float* other = teachingVectors + atributeCount*i;
			float result = 0;
			for (int j = 0; j < atributeCount; j++)
			{
				if (currentGen[j]) {
					const float diffrence = currentVector[j] - other[j];
					result += diffrence*diffrence;
				}
			}

			result = sqrt(result);
			if (result < heap[0].val) {
				heap[0].val = result;
				heap[0].label = teachingClasses[i];
				hipify(heap, k);
			}
		}

		int correctCount = 0;
		for (int i = 0; i < k; i++)
		{
			if (heap[i].label == testClasses[id]) {
				correctCount++;
			}
		}

		if (correctCount >= countToPass) {
			atomicAdd(accuracy + blockIdx.y, 1.0f);
		}

	}


}