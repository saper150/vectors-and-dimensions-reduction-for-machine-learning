#pragma once

#include <float.h>


template<typename T>
struct HeapData {
	float val;
	T label;
};


struct HeapCompare {
	template<typename T>
	__host__ __device__ bool operator()(const HeapData<T>& d1, const HeapData<T>& d2) {
		return d1.val < d2.val;
	}
};

template<typename T>
__device__ void initializeHeap(HeapData<T>* heap,const int n) {
	for (int i = 0; i < n; i++)
	{
		heap[i].val = FLT_MAX;
		//heap[i].label = -1;
	}
}


template<typename T>
__device__ void hipify(HeapData<T>* heap, int k) {
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
			const HeapData<T> tmp = heap[currentIndex];
			heap[currentIndex] = heap[biggerIndex];
			heap[biggerIndex] = tmp;

			currentIndex = biggerIndex;
			leftIndex = (biggerIndex * 2) + 1;
			rigthIndex = (biggerIndex * 2) + 2;
		}
	}

}
