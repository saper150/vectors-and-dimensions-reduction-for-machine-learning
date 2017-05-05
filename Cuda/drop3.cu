#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include "float.h"
#include <float.h>
#include <thrust/sort.h>
#include "HeapData.h"

#define _SIZE_T_DEFINED 
#ifndef __CUDACC__ 
#define __CUDACC__ 
#endif 
#ifndef __cplusplus 
#define __cplusplus 
#endif

extern "C" {

	__global__ void findNeighbours(
		float * vectors,
		int vectorCount,
		int atributeCount,
		int neighborCount,
		int* classes,
		HeapData* heapMemory,
		float* nearestEnemyDistance
	) {

		const int id = blockIdx.x*blockDim.x + threadIdx.x;
		if (id >= vectorCount) return;

		nearestEnemyDistance[id] = FLT_MAX;

		HeapData* heap = heapMemory + (id*neighborCount);

		const float* currentRow = vectors + (atributeCount*id);

		for (int i = 0; i < neighborCount; i++) {
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

				if (classes[i] != classes[id]    //is eneamy
					&& result < nearestEnemyDistance[id]
					)
				{
					nearestEnemyDistance[id] = result;
				}

				if (result < heap[0].val) {
					heap[0].val = result;
					heap[0].label = i;
					hipify(heap, neighborCount);
				}
			}
		}

		thrust::sort(heap, heap + neighborCount, HeapCompare());

	}





	__global__ void calculateDistances(
		float * vectors,
		int vectorCount,
		int vectorToExamine,
		int atributeCount,
		unsigned char* vectorsInDataset,
		float* results
	) {

		const int id = blockIdx.x*blockDim.x + threadIdx.x;
		if (id >= vectorCount) return;
		if (vectorsInDataset[id] == 0) return;

		const float* examinedVector = vectors + (atributeCount*vectorToExamine);
		const float* other = vectors + (atributeCount*id);

		float result = 0.f;
		for (int j = 0; j < atributeCount; j++)
		{
			const float d = examinedVector[j] - other[j];
			result += d*d;
		}
		result = sqrtf(result);
		results[id] = result;

	}

}