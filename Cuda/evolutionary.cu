#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include "float.h"
#include <float.h>
#include <thrust/sort.h>

#include <curand.h>
#include <curand_kernel.h>

#include <time.h>
#include <math.h>
#include"qSort.h"

__device__ int uniform_int(
	curandState& state,
	const int high
) {
	const float randomF = curand_uniform(&state)*(high+0.99999f);
	return (int)truncf(randomF);
}

__device__ 
float fitness(
	const int accuracy, 
	const int genLength,
	const float avgAccuracy,
	const float avgGenLen,
	const float alpha
) 
{
	const float a = alpha * accuracy / avgAccuracy;
	const float b = (1 - alpha) * avgGenLen / genLength;
	const float res = a + b;
	return res * res;
}




extern "C" {


	__constant__ int popSize;
	__constant__ int genLength;
	__constant__ int k;
	__constant__ int countToPass;

	__constant__ int testVectorsCount;
	__constant__ int teachingVectorsCount;

	__constant__ int attributeCount;

	__constant__ float stdAccuracy;

	__constant__ float stdVectorLen;

	__constant__ float cutOffPoint;






	/*__global__ void calculateClasses2(
		float * teachingVectors,
		float* testVectors,
		int* teachingClasses,
		float* resultDistancMemory,
		int* resultClassesMemory,
		int* neaboursIndexes
	) {
		__shared__ float s[10000];
		const int maxNumberOfVectorsInCashe = 10000 / attributeCount;


		const int id = blockIdx.x*blockDim.x + threadIdx.x;
		if (id >= testVectorsCount) return;


		int iteration = 0;

		const int countToPrefeatch = maxNumberOfVectorsInCashe / blockDim.x;
		float*  destination = s + (countToPrefeatch*attributeCount*blockIdx.x);

		float* src = teachingVectors + attributeCount*iteration;

		for (int i = 0; i < countToPrefeatch; i++)
		{
			float* currentDestination = destination + attributeCount*i;
			float* currentSrc = src + attributeCount*i;
			for (int j = 0; j < attributeCount; j++)
			{
				currentDestination[j] = currentSrc[j];
			}

		}

		s[threadIdx.x] = teachingVectors[threadIdx.x];


		float * currentVector = testVectors + (attributeCount*id);
		float* currentDistanceResult =
			resultDistancMemory + (teachingVectorsCount*id);
		int* currentClassResult =
			resultClassesMemory + (teachingVectorsCount*id);

		int* currentNeabourIndexes =
			neaboursIndexes + (teachingVectorsCount*id);


		for (int i = 0; i < teachingVectorsCount; i++)
		{
			const float* row = teachingVectors + (attributeCount*i);
			float result = 0.f;
			for (int j = 0; j < attributeCount; j++)
			{
				const float d = currentVector[j] - row[j];
				result += d*d;
			}
			currentDistanceResult[i] = sqrtf(result);
			currentClassResult[i] = teachingClasses[i];
			currentNeabourIndexes[i] = i;
		}

		sortByKey2(
			currentDistanceResult,
			currentClassResult,
			currentNeabourIndexes,
			teachingVectorsCount);

	}



*/





	__global__ void calculateNearestNeabours(
		float * teachingVectors,
		float* testVectors,
		float* resultDistancMemory,
		int* neaboursIndexes
	) {



		const int id = blockIdx.x*blockDim.x + threadIdx.x;
		if (id >= testVectorsCount) return;

		float * currentVector = testVectors + (attributeCount*id);
		float* currentDistanceResult = 
			resultDistancMemory + (teachingVectorsCount*id);

		int* currentNeabourIndexes = 
			neaboursIndexes + (teachingVectorsCount*id);


		for (int i = 0; i < teachingVectorsCount; i++)
		{
			const float* row = teachingVectors + (attributeCount*i);
			float result = 0.f;
			for (int j = 0; j < attributeCount; j++)
			{
				const float d = currentVector[j] - row[j];
				result += d*d;
			}
			currentDistanceResult[i] = sqrtf(result);
			currentNeabourIndexes[i] = i;
		}

		sortByKey(
			currentDistanceResult, 
			currentNeabourIndexes,
			teachingVectorsCount
		);

	}



	//__global__ void createInitialPopulation(
	//	const unsigned char* parent,
	//	unsigned char* outputPopulation
	//) {
	//	const int id = threadIdx.x + blockIdx.x * blockDim.x;
	//	
	//	if (id >= popSize) return;

	//	curandState state;
	//	curand_init(clock64(), id,0, &state);
	//	float a = curand_uniform(&state);
	//	unsigned char* currentPopulation =
	//		outputPopulation + (genLength*id);
	//	
	//	for (int i = 0; i < genLength; i++)
	//	{
	//		float chanceToFlip = parent[i] ? 0.2f : 0.1f;

	//		if (curand_uniform(&state) < chanceToFlip) {
	//			currentPopulation[i] = !parent[i];
	//		}
	//		else {
	//			currentPopulation[i] = parent[i];
	//		}
	//	}

	//}

	__global__ void calculateAccuracy(
			const int* testClasses,
			const int* teachingClasses,
			const unsigned char* populationGens,
			const int* neaboursIndexes,
			int* correctCounts
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
		int correctCount = 0;
		for (int i = 0; i < neaboursSize && neaboursFound < k; i++)
		{
			const int p0 = currentNeaboursIndexes[i];
			if (currentGen[p0]) {
				neaboursFound++;
				correctCount += (testClasses[id] == teachingClasses[p0] ? 1 : 0);
			}
		}

		if (correctCount >= countToPass) {
			atomicAdd(correctCounts + currentPopulationGenIndex,1);
		}
	}





	//__global__ void calculateAccuracyRegresion(
	//	const float* testValues,
	//	const float* teachingValues,
	//	const float* distances,
	//	const unsigned char* populationGens,
	//	float* accuracyes
	//) {
	//	const int neaboursSize = teachingVectorsCount;

	//	const int id = threadIdx.x + blockIdx.x * blockDim.x;
	//	if (id >= testVectorsCount) return;

	//	const int currentPopulationGenIndex = blockIdx.y;

	//	const unsigned char* currentGen =
	//		populationGens + (genLength*currentPopulationGenIndex);

	//	const int* currentNeaboursIndexes =
	//		neaboursIndexes + (neaboursSize*id);


	//	int neaboursFound = 0;
	//	int correctCount = 0;
	//	float guesedVal = 0;
	//	for (int i = 0; i < neaboursSize && neaboursFound < k; i++)
	//	{
	//		const int p0 = currentNeaboursIndexes[i];
	//		if (currentGen[p0]) {
	//			neaboursFound++;
	//			guesedVal += teachingValues[p0];
	//		}
	//	}

	//	guesedVal /= (float)k;

	//	const float diffrence = 
	//		guesedVal - testValues[currentPopulationGenIndex];

	//	accuracyes[currentPopulationGenIndex] 
	//		= 1.f / sqrt(diffrence*diffrence);

	//}

	__global__ void countVectors(
		unsigned char* gens,
		int* vectorSizes
	) {
		
		const int id = threadIdx.x + blockIdx.x * blockDim.x;
		//if (id >= popSize) return;

		vectorSizes[id] = 0;
		const unsigned char* currentGen = gens + genLength*id;

		for (int i = 0; i < genLength; i++)
		{
			if (currentGen[i])
				vectorSizes[id]++;
		}

	}


	__global__ void genetic(
		const int* accuracy,
		const int* genLengths,
		const float alpha,
		const unsigned char* currentPopulation,
		unsigned char* nextPopulation,
		float crossoverRate,
		float mutationRate,
		float avgAccuracy,
		float avgGenLen,
		float* fitness
	) {
		extern __shared__ int shared[];
		int* turnamentWinners = (int*)shared;

		const int id = threadIdx.x;

		// calculating fitness
		{
			if (genLengths[id] <= k) {
				fitness[id] = 0;
			}
			else
			{
				const float a = alpha * accuracy[id] / avgAccuracy;
				const float b = (1 - alpha) * avgGenLen / genLengths[id];
				const float res = a + b;
				fitness[id] = res*res;
			}

		}

		curandState state;
		curand_init(clock64(), id, 0, &state);

		__syncthreads();
		//turnament selection
		{
			const int other = uniform_int(state, popSize-1);
			turnamentWinners[id] = fitness[id] > fitness[other] ? id : other;
		}


		__syncthreads();

		unsigned char* destination = nextPopulation + genLength*id;
		//crossover
		{

			const int parent1Index = uniform_int(state, popSize - 1);
			const int parent2Index = uniform_int(state, popSize - 1);
			const int crossoverPoint = uniform_int(state, genLength - 1);
			const float crossoverChance = curand_uniform(&state);


			const unsigned char* firstParent =
				currentPopulation + turnamentWinners[parent1Index] * genLength;

			const unsigned char* secondParent =
				currentPopulation + turnamentWinners
				[
					crossoverChance < crossoverRate ? parent2Index : parent1Index 
				] * genLength;

			memcpy(destination, firstParent, crossoverPoint);
			memcpy(destination + crossoverPoint, secondParent + crossoverPoint, genLength - crossoverPoint);

/*
			if (r < crossoverRate) {
				const int crossoverPoint = uniform_int(state, genLength-1);

				const unsigned char* firstParent = 
					currentPopulation + turnamentWinners[uniform_int(state,popSize-1)] * genLength;
				const unsigned char* secondParent =
					currentPopulation + turnamentWinners[uniform_int(state, popSize - 1)] * genLength;

				memcpy(destination, firstParent, crossoverPoint);
				memcpy(destination+crossoverPoint, secondParent+crossoverPoint, genLength - crossoverPoint);

			}
			else {
				const unsigned char* firstParent =
					currentPopulation + turnamentWinners[uniform_int(state, popSize - 1)] * genLength;
				memcpy(destination, firstParent, genLength);
			}
*/
		}

		//mutation
		{
			for (int i = 0; i < genLength; i++)
			{
				if (curand_uniform(&state) < mutationRate) {
					destination[i] = !destination[i];
				}
			}

		}


	}

}

