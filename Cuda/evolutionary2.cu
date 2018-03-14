#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>
#include <thrust/sort.h>

#include <time.h>
#include <math.h>
#include"qSort.h"

#include "uniform_int.h"

template<typename T,typename U>
__device__ void vectorisedMemcpy(const T* src,T* desc,int size) 
{
	static_assert(sizeof(U)>=sizeof(T),"");
	constexpr int ration = (sizeof(U) / sizeof(T));

	const int bulkSize = size / ration;
	for (int i = 0; i < bulkSize; i++)
	{
		reinterpret_cast<U*>(desc)[i] = reinterpret_cast<const U*>(src)[i];
	}

	for (int i = size - (size/ ration) ; i < size; i++)
	{
		desc[i] = src[i];
	}

}



extern "C" {

	__constant__ int genLength;


	__constant__ int popSize;
	__constant__ float mutationRate;
	__constant__ float crossoverRate;
	__constant__ float alpha;

	__constant__ int eliteIndex;

	__global__ void genetic(
		const unsigned char* currentPopulation,
		unsigned char* nextPopulation,
		const float* sortedFitness,
		const int* fitnessIndeces
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

		if (id>=eliteIndex) {
			const unsigned char*  eliteParent = currentPopulation + fitnessIndeces[id]*genLength;
			for (int i = 0; i < genLength; i++)
			{
				destination[i] = eliteParent[i];
			}

			return;
		}

		//crossover old
		{

			//const int parent1Index = uniform_int(state, popSize - 1);
			//const int parent2Index = uniform_int(state, popSize - 1);
			//const int crossoverPoint = uniform_int(state, genLength - 1);
			//const float crossoverChance = curand_uniform(&state);


			//const unsigned char* firstParent =
			//	currentPopulation + turnamentWinners[parent1Index] * genLength;

 		//	const unsigned char* secondParent =
			//	currentPopulation + turnamentWinners
			//	[
			//		crossoverChance < crossoverRate ? parent2Index : parent1Index
			//	] * genLength + crossoverPoint;

			//unsigned char* destinationSecondPart = destination + crossoverPoint;


			//for (int i = 0; i < crossoverPoint; i++)
			//{
			//	destination[i] = firstParent[i];
			//}

			//const int count = genLength - crossoverPoint;
			//for (int i = 0; i < count; i++)
			//{
			//	destinationSecondPart[i] = secondParent[i];
			//}


		}


		{
			const int crossoverCount = 10;
			
			int crossoverPoints[1026];
			crossoverPoints[0] = 0;
			for (int i = 1; i < crossoverCount+1; i++)
			{
				crossoverPoints[i] = uniform_int(state, genLength - 1);
			}

			crossoverPoints[crossoverCount + 1] = genLength;

			thrust::sort(crossoverPoints+1, crossoverPoints + crossoverCount+1,thrust::less<int>());

			for (int i = 0; i < crossoverCount; i++)
			{
				const unsigned char * __restrict__ parent =
					currentPopulation+ (turnamentWinners[uniform_int(state, popSize - 1)])*genLength;


				for (int j = crossoverPoints[i]; j < crossoverPoints[i+1]; j++)
				{
					destination[j] = parent[j];
				}

			}

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

