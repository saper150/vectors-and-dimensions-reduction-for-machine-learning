using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


public static class Utils
{
    public static Data[] findKminimum(int k, float[] where) {
        Data[] heap = new Data[k];
        for (int i = 0; i < heap.Length; i++)
        {
            heap[i].val = float.MaxValue;
        }
        for (int i = 0; i < where.Length; i++)
        {
            if (where[i] < heap[0].val) {
                heap[0].val = where[i];
                heap[0].index = i;
                hipify(heap);
            }
        }
        return heap;
    }

    public static void hipify(Data[] heap) {
        int currentIndex = 0;
        int leftIndex = 1;
        int rigthIndex = 2;

        while (rigthIndex < heap.Length)
        {
            int biggerIndex = heap[rigthIndex].val > heap[leftIndex].val 
                ? rigthIndex : leftIndex;
            if (heap[currentIndex].val > heap[biggerIndex].val)
            {
                return;
            }
            else {
                Data tmp = heap[currentIndex];
                heap[currentIndex] = heap[biggerIndex];
                heap[biggerIndex] = tmp;

                currentIndex = biggerIndex;
                leftIndex = (biggerIndex * 2) + 1;
                rigthIndex= (biggerIndex * 2) + 2;


            }


        }

    }
    
}

