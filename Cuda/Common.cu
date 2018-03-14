
#include <device_launch_parameters.h> 
#include <builtin_types.h> 

#include <float.h>




#include <math.h>


extern "C" {

	__constant__ int genLength;


	__global__ void countVectors(
		unsigned char* gens,
		int* vectorSizes
	) {
		
		const int id = threadIdx.x;
		//if (id >= popSize) return;

		//vectorSizes[id] = 0;
		int size = 0;
		const unsigned char* currentGen = gens + genLength*id;
		for (int i = 0; i < genLength; i++)
		{

			if (currentGen[i])
				size++;
		}
		vectorSizes[id] = size;

	}





	__global__ void countVectorsIndeces(
		unsigned char* gens,
		int* indeces,
		int* vectorSizes
	) {

		const int id = threadIdx.x + blockIdx.x * blockDim.x;
		//if (id >= popSize) return;

		vectorSizes[id] = 0;
		int* currentIndeces = indeces + genLength*id;
		const unsigned char* currentGen = gens + genLength*id;

		for (int i = 0; i < genLength; i++)
		{
			if (currentGen[i]) {
				currentIndeces[vectorSizes[id]] = i;
				vectorSizes[id]++;
			}
		}

	}

}