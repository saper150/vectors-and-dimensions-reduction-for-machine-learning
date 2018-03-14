#pragma once

template<typename T>
__device__ void splitCopy(
	T* destination,
	const T* source,
	const int size
)
{
	const int elementsPerThread = size / blockDim.x;
	const int rest = size % blockDim.x;

	const int plusOne = elementsPerThread + 1;
	if (threadIdx.x < rest)
	{
		const int startCopyIndex = (elementsPerThread + 1) * threadIdx.x;

		for (int i = 0; i < plusOne; i++)
		{
			const int startPlusI = startCopyIndex + i;
			destination[startPlusI] = source[startPlusI];
		}
	}
	else {
		const int startCopyIndex = (elementsPerThread * threadIdx.x) + rest;
		for (int i = 0; i < elementsPerThread; i++)
		{
			const int startPlusI = startCopyIndex + i;
			destination[startPlusI] = source[startPlusI];
		}

	}

}