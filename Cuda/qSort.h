#pragma once

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <float.h>


__device__ void swap(float* keys,int* values,const int a,const int b) {
	const float tmpKey = keys[a];
	const int tmpValue = values[a];

	keys[a] = keys[b];
	values[a] = values[b];

	keys[b] = tmpKey;
	values[b] = tmpValue;

}


__device__
void shiftRight(float* key, int* values, const int low,const int high)
{
	int root = low;
	while ((root * 2) + 1 <= high)
	{
		int leftChild = (root * 2) + 1;
		int rightChild = leftChild + 1;
		int swapIdx = root;
		/*Check if root is less than left child*/
		if (key[swapIdx] < key[leftChild])
		{
			swapIdx = leftChild;
		}
		/*If right child exists check if it is less than current root*/
		if ((rightChild <= high) && (key[swapIdx] < key[rightChild]))
		{
			swapIdx = rightChild;
		}
		/*Make the biggest element of root, left and right child the root*/
		if (swapIdx != root)
		{
			swap(key, values, root, swapIdx);
			/*Keep shifting right and ensure that swapIdx satisfies
			heap property aka left and right child of it is smaller than
			itself*/
			root = swapIdx;
		}
		else
		{
			break;
		}
	}
	return;
}
__device__

void heapify(float* key, int* values, const int low, const int high)
{
	/*Start with middle element. Middle element is chosen in
	such a way that the last element of array is either its
	left child or right child*/
	int midIdx = (high - low - 1) / 2;
	while (midIdx >= 0)
	{
		shiftRight(key,values, midIdx, high);
		--midIdx;
	}
	return;
}

__device__
void sortByKey(float* key, int* values, const int size)
{

	/*This will put max element in the index 0*/
	heapify(key,values, 0, size - 1);
	int high = size - 1;
	while (high > 0)
	{
		/*Swap max element with high index in the array*/
		swap(key, values, 0, high);
		--high;
		/*Ensure heap property on remaining elements*/
		shiftRight(key,values, 0, high);
	}
	return;
}





//
//
//
//__device__ void swap2(float* keys, int* values,int* values2, const int a, const int b) {
//	const float tmpKey = keys[a];
//	const int tmpValue = values[a];
//	const int tmpValue2 = values2[a];
//
//
//	keys[a] = keys[b];
//	values[a] = values[b];
//	values2[a] = values2[b];
//
//	keys[b] = tmpKey;
//	values[b] = tmpValue;
//	values2[b] = tmpValue2;
//
//}
//
//
//
//
//
//
//
//__device__
//void shiftRight2(float* key, int* values,int* values2, const int low, const int high)
//{
//	int root = low;
//	while ((root * 2) + 1 <= high)
//	{
//		int leftChild = (root * 2) + 1;
//		int rightChild = leftChild + 1;
//		int swapIdx = root;
//		/*Check if root is less than left child*/
//		if (key[swapIdx] < key[leftChild])
//		{
//			swapIdx = leftChild;
//		}
//		/*If right child exists check if it is less than current root*/
//		if ((rightChild <= high) && (key[swapIdx] < key[rightChild]))
//		{
//			swapIdx = rightChild;
//		}
//		/*Make the biggest element of root, left and right child the root*/
//		if (swapIdx != root)
//		{
//			swap2(key, values,values2, root, swapIdx);
//			/*Keep shifting right and ensure that swapIdx satisfies
//			heap property aka left and right child of it is smaller than
//			itself*/
//			root = swapIdx;
//		}
//		else
//		{
//			break;
//		}
//	}
//	return;
//}
//__device__
//
//void heapify2(float* key, int* values,int* values2, const int low, const int high)
//{
//	/*Start with middle element. Middle element is chosen in
//	such a way that the last element of array is either its
//	left child or right child*/
//	int midIdx = (high - low - 1) / 2;
//	while (midIdx >= 0)
//	{
//		shiftRight2(key, values,values2, midIdx, high);
//		--midIdx;
//	}
//	return;
//}
//
//__device__
//void sortByKey2(float* key, int* values, int* values2, const int size)
//{
//
//	/*This will put max element in the index 0*/
//	heapify2(key, values,values2, 0, size - 1);
//	int high = size - 1;
//	while (high > 0)
//	{
//		/*Swap max element with high index in the array*/
//		swap2(key, values,values2, 0, high);
//		--high;
//		/*Ensure heap property on remaining elements*/
//		shiftRight2(key, values,values2, 0, high);
//	}
//	return;
//}
