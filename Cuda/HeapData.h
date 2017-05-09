#pragma once

#include <float.h>

struct HeapData {
	float val;
	int label;
};
struct HeapCompare {
	__host__ __device__ bool operator()(const HeapData& d1, const HeapData& d2) {
		return d1.val < d2.val;
	}
};

__device__ void initializeHeap(HeapData* heap,const int n) {
	for (int i = 0; i < n; i++)
	{
		heap[i].val = FLT_MAX;
	}
}



__device__ void hipify(HeapData* heap, int k) {
	int currentIndex = 0;
	int leftIndex = 1;
	int rigthIndex = 2;

	while (rigthIndex < k)
	{
		const int biggerIndex = heap[rigthIndex].val > heap[leftIndex].val 
			? rigthIndex : leftIndex;
		if (heap[currentIndex].val > heap[biggerIndex].val) {
			rigthIndex = k;
		}
		else {
			const HeapData tmp = heap[currentIndex];
			heap[currentIndex] = heap[biggerIndex];
			heap[biggerIndex] = tmp;

			currentIndex = biggerIndex;
			leftIndex = (biggerIndex * 2) + 1;
			rigthIndex = (biggerIndex * 2) + 2;
		}
	}

}
