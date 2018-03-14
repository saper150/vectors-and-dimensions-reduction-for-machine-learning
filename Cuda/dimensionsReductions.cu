#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "HeapData.h"
#include <math.h>

#include <float.h>
#include "splitCopy.h"

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
		const int* vectorSizes,
		const int* populationIndeces,
		unsigned char* isInCashe,
		HeapData<int>* heapMemory,
		float* accuracy

	)
	{
		__shared__ int indecesCashe[1000];

		const int currentVectorSize = vectorSizes[blockIdx.y];

		{
			const int* currentIndeces = populationIndeces + atributeCount*blockIdx.y;
			splitCopy(indecesCashe, currentIndeces, currentVectorSize);
			__syncthreads();
		}

		const int id = threadIdx.x + blockIdx.x * blockDim.x;
		if (id >= testVectorsCount) return;

		if (isInCashe[blockIdx.y]) {
			return;
		}

		const float* currentVector = testVectors + atributeCount*id;
		float currentVectorCashe[200];
		for (int i = 0; i < atributeCount; i++)
		{
			currentVectorCashe[i] = currentVector[i];
		}


		int a = (threadIdx.y*k*testVectorsCount) + (id*k);
		HeapData<int> * heap =  heapMemory + (blockIdx.y*k*testVectorsCount) + (id*k);

		initializeHeap(heap, k);

		for (int i = 0; i < teachingVectorsCount; i++)
		{

			const float* currentOther = teachingVectors + atributeCount*i;
			float result = 0;
			for (int j = 0; j < currentVectorSize; j++)
			{
				const int indece = indecesCashe[j];
				const float diffrence = currentVectorCashe[indece] - currentOther[indece];
				result += diffrence*diffrence;	
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








	__global__ void geneticKnnRegresion(
		const float* testVectors,
		const float* testValues,
		const float* teachingVectors,
		const float* teachingValues,
		const int* vectorSizes,
		const int* populationIndeces,
		unsigned char* isInCashe,
		HeapData<float>* heapMemory,
		float* squaredDiff

	)
	{
		__shared__ int indecesCashe[1000];

		const int currentVectorSize = vectorSizes[blockIdx.y];

		{
			const int* currentIndeces = populationIndeces + atributeCount*blockIdx.y;
			splitCopy(indecesCashe, currentIndeces, currentVectorSize);
			__syncthreads();
		}

		const int id = threadIdx.x + blockIdx.x * blockDim.x;
		if (id >= testVectorsCount) return;

		if (isInCashe[blockIdx.y]) {
			return;
		}

		const float* currentVector = testVectors + atributeCount*id;
		float currentVectorCashe[200];
		for (int i = 0; i < atributeCount; i++)
		{
			currentVectorCashe[i] = currentVector[i];
		}


		int a = (threadIdx.y*k*testVectorsCount) + (id*k);
		HeapData<float> * heap = heapMemory + (blockIdx.y*k*testVectorsCount) + (id*k);

		initializeHeap(heap, k);

		for (int i = 0; i < teachingVectorsCount; i++)
		{

			const float* currentOther = teachingVectors + atributeCount*i;
			float result = 0;
			for (int j = 0; j < currentVectorSize; j++)
			{
				const int indece = indecesCashe[j];
				const float diffrence = currentVectorCashe[indece] - currentOther[indece];
				result += diffrence*diffrence;
			}

			result = sqrt(result);

			if (result < heap[0].val) {
				heap[0].val = result;
				heap[0].label = teachingValues[i];
				hipify(heap, k);
			}

		}

		float avrage = 0;

		for (int i = 0; i < k; i++)
		{
			avrage += heap[i].label;
		}

		avrage /= (float)k;
	
		const float dif = avrage - testValues[blockIdx.y];
		atomicAdd(squaredDiff + blockIdx.y, dif*dif);

	}




































	struct Lock
	{
		int mutex = 0;

		__device__ void lock() {
			while (atomicCAS(&mutex, 0, 1) != 0);
			
		}
		__device__ void unlock() {
			atomicExch(&mutex, 0);
		}

	};

	struct Node {
		Lock lock;
		Node* one = nullptr;
		Node* zero = nullptr;
	};

	__global__ void saveToCashe(
		const unsigned char* population,
		const float* accuracyValues,
		Node* root
	) 
	{
		const int warp = (blockIdx.x*blockDim.x+threadIdx.x) % 32;
		if (warp != 0) {
			return;
		}
		const int id = (blockIdx.x*blockDim.x + threadIdx.x)/32;
		if (id >= popSize) return;

		const unsigned char * gen = population + id*atributeCount;

		int i = 0;
		Node* currentNode = root;

		for (int i = 0; i < atributeCount-1; i++)
		{
			Node ** nextNode = gen[i] ? &currentNode->one : &currentNode->zero;
			if (!*nextNode) {
				currentNode->lock.lock();
				if (!*nextNode) {
					*nextNode = new Node();
				}
				currentNode->lock.unlock();
			}
			currentNode = *nextNode;
		}

		float ** accuracyValue = gen[atributeCount-1] ? (float**)&currentNode->one : (float**)&currentNode->zero;
		if (!*accuracyValue) {
			currentNode->lock.lock();
			if (!*accuracyValue) {
				*accuracyValue = new float(accuracyValues[id]);
				float a = **accuracyValue;
				float b = a;
			}
			currentNode->lock.unlock();
		}


	}



	__global__ void readCashe(
		const unsigned char* population,
		float* result,
		unsigned char* isInCashe,
		Node * root
	) {
	
		const int id = threadIdx.x;
		const unsigned char * gen = population + atributeCount*id;

		Node * currentNode = root;
		for (int i = 0; i < atributeCount-1; i++)
		{
			Node* nextNode = gen[i] ? currentNode->one : currentNode->zero;
			if (!nextNode) {
				isInCashe[id] = 0;
				result[id] = 0;
				return;
			}
			currentNode = nextNode;
		}

		float* casheValue = gen[atributeCount-1] ? (float*)currentNode->one : (float*)currentNode->zero;
		if (casheValue) {
			result[id] = *casheValue;
			isInCashe[id] = 1;
		}
		else {
			isInCashe[id] = 0;
			result[id] = 0;
		}

	}


}

