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



	__global__ void geneticKnn(
		const float* testVectors,
		const int* testClasses,
		const float* teachingVectors,
		const int* teachingClasses,
		unsigned char* population,
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
				if (currentGen[i]) {
					const float diffrence = currentVector[i] - other[i];
					result += diffrence*diffrence;
				}
			}
			result = sqrt(result);
			if (result < heap[0].val) {
				heap[0].val = result;
				heap[0].label = i;
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
			atomicAdd(accuracy + id, 1.0f);
		}

	}


}