#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <builtin_types.h> 

#include"qSort.h"

#include "splitCopy.h"
#include <math.h>


extern "C" {



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



	__constant__ int testVectorsCount;
	__constant__ int teachingVectorsCount;
	__constant__ int attributeCount;


	__global__ void calculateNearestNeabours(
		float * teachingVectors,
		float* testVectors,
		float* resultDistancMemory,
		int* neaboursIndexes
	) {
		__shared__ float cashe[1000];

		const int id = blockIdx.x*blockDim.x + threadIdx.x;
		if (id >= testVectorsCount) return;

		float * currentVector = testVectors + (attributeCount*id);
		float currentCashe[200];
		for (int i = 0; i < attributeCount; i++)
		{
			currentCashe[i] = currentVector[i];
		}


		float* currentDistanceResult =
			resultDistancMemory + (teachingVectorsCount*id);

		int* currentNeabourIndexes =
			neaboursIndexes + (teachingVectorsCount*id);


		for (int i = 0; i < teachingVectorsCount; i++)
		{
			const float* row = teachingVectors + (attributeCount*i);
			splitCopy(cashe, row, attributeCount);
			__syncthreads();
			float result = 0.f;
			for (int j = 0; j < attributeCount; j++)
			{
				const float d = currentCashe[j] - cashe[j];
				result += d*d;
			}
			currentDistanceResult[i] = sqrtf(result);
			currentNeabourIndexes[i] = i;
		}

		//sortByKey(
		//	currentDistanceResult,
		//	currentNeabourIndexes,
		//	teachingVectorsCount
		//);

	}


	__constant__ int k;
	__constant__ int countToPass;
	__constant__ int genLength;

	__global__ void calculateAccuracy(
		const int* testClasses,
		const int* teachingClasses,
		const unsigned char* populationGens,
		const int* neaboursIndexes,
		float* correctCounts,
		int startIndex
	) {
		__shared__ int cashedGen;
		__shared__ int cashedTeachingClass;

		const int neaboursSize = teachingVectorsCount;

		const int id = threadIdx.x + blockIdx.x * blockDim.x;
		if (id >= testVectorsCount) return;

		const int currentPopulationGenIndex = blockIdx.y;

		const unsigned char* currentGen =
			populationGens + (genLength*currentPopulationGenIndex);

		const int* currentNeaboursIndexes =
			neaboursIndexes + (neaboursSize*id);


		int neaboursFound = 0;
		int correctCount = 0;
		for (int i = startIndex; i < neaboursSize && neaboursFound < k; i++)
		{

				const int p0 = currentNeaboursIndexes[i];
			//if (threadIdx.x == 0) {
			//	cashedGen = currentGen[p0];
			//	cashedTeachingClass = teachingClasses[p0];
			//}
			//__syncthreads();

			if (currentGen[p0]) {
				neaboursFound++;
				correctCount += (testClasses[id] == teachingClasses[p0] ? 1 : 0);
			}
		}

		if (correctCount >= countToPass) {
			atomicAdd(correctCounts + currentPopulationGenIndex, 1.f);
		}
	}






	__global__ void calculateAccuracyRegresion(
		const float* testValues,
		const float* teachingValues,
		const unsigned char* populationGens,
		const int* neaboursIndexes,
		float* squaredDiff
	) {

		const int neaboursSize = teachingVectorsCount;

		const int id = threadIdx.x + blockIdx.x * blockDim.x;
		if (id >= testVectorsCount) return;

		const int currentPopulationGenIndex = blockIdx.y;

		const unsigned char* currentGen =
			populationGens + (genLength*currentPopulationGenIndex);

		const int* currentNeaboursIndexes =
			neaboursIndexes + (neaboursSize*id);


		int neaboursFound = 0;
		float avrage= 0;
		for (int i = 0; i < neaboursSize && neaboursFound < k; i++)
		{
			const int p0 = currentNeaboursIndexes[i];
			if (currentGen[p0]) {
				neaboursFound++;
				avrage += teachingValues[p0];
			}
		}
		avrage = avrage / (float)k;

		const float dif = avrage - testValues[currentPopulationGenIndex];
		atomicAdd(squaredDiff+ currentPopulationGenIndex, dif*dif);

	}



	__global__ void RMSE(
		float* squaredDiff
	) {
		squaredDiff[threadIdx.x] = sqrtf(squaredDiff[threadIdx.x] / (float)testVectorsCount);
	}

}

