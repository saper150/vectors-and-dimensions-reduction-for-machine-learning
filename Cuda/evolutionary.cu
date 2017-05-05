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

#include"qSort.h"


extern "C" {

	__global__ void calculateDistances(
		float * teachingVectors,
		int teachingVectorsCount,
		float* testVectors,
		int testVectorsCount,
		int* teachingClasses,
		int atributeCount,
		float* resultDistancMemory,
		int* resultClassesMemory
	) {
		const int id = blockIdx.x*blockDim.x + threadIdx.x;
		if (id >= testVectorsCount) return;

		float * currentVector = testVectors + (atributeCount*id);
		float* currentDistanceResult = 
			resultDistancMemory + (teachingVectorsCount*id);
		int* currentClassResult 
			= resultClassesMemory + (teachingVectorsCount*id);


		for (int i = 0; i < teachingVectorsCount; i++)
		{
			const float* row = teachingVectors + (atributeCount*i);
			float result = 0.f;
			for (int j = 0; j < atributeCount; j++)
			{
				const float d = currentVector[j] - row[j];
				result += d*d;
			}
			currentDistanceResult[i] = sqrtf(result);
			currentClassResult[i] = teachingClasses[i];
		}

	
		sortByKey(
			currentDistanceResult, 
			currentClassResult,
			teachingVectorsCount);

	}



	__global__ void createInitialPopulation(
		unsigned char* parent,
		int genCount,
		unsigned char* outputPopulation,
		int populationSize
	) {
		const int id = threadIdx.x + blockIdx.x * blockDim.x;
		
		if (id >= populationSize) return;

		curandState state;
		curand_init(clock64(), id,0, &state);

		unsigned char* currentPopulation =
			outputPopulation + (genCount*id);

		for (int i = 0; i < genCount; i++)
		{
			float chanceToFlip = parent[i] ? 0.8 : 0.2f;

			if (curand_uniform(&state) < chanceToFlip) {
				currentPopulation[i] = !parent[i];
			}
			else {
				currentPopulation[i] = parent[i];
			}
		}

	}

	__global__ void findNewNeabours(
			int* classes,
			int thredCount,
			int* neabours,
			int neaboursSize,
			unsigned char* populationGens,
			int popSize,
			int k,
			int countToPass,
			int* correctCounts
			
	) {

		const int id = threadIdx.x + blockIdx.x * blockDim.x;
		if (id >= thredCount) return;

		const int currentPopulationGenIndex = blockIdx.y;

		const unsigned char* currentGen = 
			populationGens + (popSize*currentPopulationGenIndex);

		const int* currentNeabours = neabours + (neaboursSize*id);

		int neaboursFound = 0;
		int i;
		int correctCount = 0;
		do
		{
			if (currentGen[i]) {
				neaboursFound++;
				if (classes[id] == currentNeabours[i]) {
					correctCount++;
				}
			}
			i++;
		} while (neaboursFound < k);

		if (correctCount >= countToPass) {
			atomicAdd(correctCounts+currentPopulationGenIndex, 1);
		}
	}

	__global__ void countVectors(
		unsigned char* gens,
		int genLength,
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

	__global__ void calculateFitnes(
		const int* accuracy,
		const int* vectorSizes,
		const float alpha,
		float* fitnes
	) {
	
		const int id = threadIdx.x + blockIdx.x * blockDim.x;
		fitnes[id] = alpha * (1.f / (float)vectorSizes[id]) + (1 - alpha)*accuracy[id];
	}



	__global__ void crossOver(
		float* fitness,
		unsigned char* population,
		int popSize
	) 
	{
	
	
	
	}



}

