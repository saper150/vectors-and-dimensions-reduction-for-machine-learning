#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include "float.h"
#include <float.h>

#include "HeapData.h"

extern "C"  
{

	__global__ void knnKernal(
		float* teachingVectors,
		int teachingCount,
		float* testVectors,
		int testCount,
		int* labels,
		int vectorLen,
		int k,
		HeapData<int>* heapMemory
	)
	{
		const int id = blockIdx.x*blockDim.x + threadIdx.x;
		if (id >= testCount) return;

		HeapData<int>* heap = heapMemory + (id*k);

		const float* currentRow = testVectors + (vectorLen*id);

		for (int i = 0; i < k; i++) {
			heap[i].val = FLT_MAX;
			
		}

		for (int learningIndex = 0; learningIndex < teachingCount; learningIndex++) {
			const float* currentTeachingVector = teachingVectors + (vectorLen*learningIndex);

			float result = 0.0f;
			for (int paramIndex = 0; paramIndex < vectorLen; paramIndex++) {

				const float d = currentTeachingVector[paramIndex] - currentRow[paramIndex];
				result += d*d;

			}
			result = sqrtf(result);
			if (result < heap[0].val) {
				heap[0].val= result;
				heap[0].label = labels[learningIndex];
				hipify(heap, k);

			}
		}

	}


	__global__ void enn(
		float * vectors,
		int vectorCount,
		int atributeCount,
		int* classes,
		int k,
		int countToPass,
		HeapData<int>* heapMemory,
		unsigned char* result
		) 
	{
	
		const int id = blockIdx.x*blockDim.x + threadIdx.x;
		if (id >= vectorCount) return;

		HeapData<int>* heap = heapMemory + (id*k);

		const float* currentRow = vectors + (atributeCount*id);

		for (int i = 0; i < k; i++) {
			heap[i].val = FLT_MAX;

		}
		for (int i = 0; i < vectorCount; i++)
		{
			const float* row = vectors + (atributeCount*i);
			if (row != currentRow) {
				float result = 0.f;
				for (int j = 0; j < atributeCount; j++)
				{
					const float d = currentRow[j] - row[j];
					result += d*d;
				}
				result = sqrtf(result);
				if (result < heap[0].val) {
					heap[0].val = result;
					heap[0].label = classes[i];
					hipify(heap, k);
				}
			}
		}

		int correctCount = 0;
		for (int i = 0; i < k; i++)
		{
			if (heap[i].label == classes[id])
				correctCount++;
		}
		
		if (correctCount >= countToPass) {
			result[id] = 1;
		}
		else {
			result[id] = 0;
		}
	
	}

	//__global__ void findNeighbours(
	//	float * vectors,
	//	int vectorCount,
	//	int atributeCount,
	//	int k,
	//	int* classes,
	//	HeapData* heapMemory,
	//	float* nearestEnemyDistance
	//) {

	//	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	//	if (id >= vectorCount) return;

	//	nearestEnemyDistance[id] = FLT_MAX;

	//	HeapData* heap = heapMemory + (id*k);

	//	const float* currentRow = vectors + (atributeCount*id);

	//	for (int i = 0; i < k; i++) {
	//		heap[i].val = FLT_MAX;
	//	}

	//	for (int i = 0; i < vectorCount; i++)
	//	{
	//		const float* row = vectors + (atributeCount*i);
	//		if (row != currentRow) {
	//			float result = 0.f;
	//			for (int j = 0; j < atributeCount; j++)
	//			{
	//				const float d = currentRow[j] - row[j];
	//				result += d*d;
	//			}
	//			result = sqrtf(result);
	//			
	//			if (classes[i] != classes[id]    //is eneamy
	//				&& result<nearestEnemyDistance[id]
	//				) 
	//			{
	//				nearestEnemyDistance[id] = result;
	//			}

	//			if (result < heap[0].val) {
	//				heap[0].val = result;
	//				heap[0].label = i;
	//				hipify(heap, k);
	//			}
	//		}
	//	}
	//}



}