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
	const float randomF = curand_uniform(&state)*(high + 0.99999f);
	return (int)truncf(randomF);
}


extern "C" {


	__constant__ int popSize;
	__constant__ int genLength;

	__constant__ int mutationRate;
	__constant__ int crossoverRate;
	__constant__ float alpha;





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
		const unsigned char* currentPopulation,
		unsigned char* nextPopulation,
		float* sortedFitness,
		int* fitnessIndeces
	) {
		extern __shared__ int shared[];
		int* turnamentWinners = (int*)shared;

		const int id = threadIdx.x;

		curandState state;
		curand_init(clock64(), id, 0, &state);

		//turnament selection
		{
			const int other = uniform_int(state, popSize - 1);
			turnamentWinners[id] = sortedFitness[id] > sortedFitness[other] 
				? fitnessIndeces[id] : fitnessIndeces[other];
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

