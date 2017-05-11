#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include "float.h"
#include <float.h>
#include <thrust/sort.h>

#include <time.h>
#include <math.h>
#include"qSort.h"

#include "uniform_int.h"


extern "C" {

	__constant__ int genLength;


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

	__constant__ int popSize;
	__constant__ float mutationRate;
	__constant__ float crossoverRate;
	__constant__ float alpha;

	__constant__ int eliteIndex;

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

		if (id >= eliteIndex) {
			const unsigned char * eliteParent = currentPopulation + fitnessIndeces[id];
			memcpy(destination, eliteParent, genLength);
			return;
		}

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
				] * genLength + crossoverPoint;

			unsigned char* destinationSecondPart = destination + crossoverPoint;

			for (int i = 0; i < crossoverPoint; i++)
			{
				destination[i] = firstParent[i];
			}

			//unsigned char* secondPartDestination = destination + crossoverPoint;
			//const unsigned char* secondPartOfSecondParent = secondParent+ crossoverPoint;

			const int count = genLength - crossoverPoint;
			for (int i = 0; i < count; i++)
			{
				destinationSecondPart[i] = secondParent[i];
			}


			//memcpy(destination, firstParent, crossoverPoint);
			//memcpy(destination + crossoverPoint, secondParent + crossoverPoint, genLength - crossoverPoint);

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

