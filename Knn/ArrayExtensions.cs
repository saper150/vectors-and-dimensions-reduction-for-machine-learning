using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


public static class ArrayExtensions
{
    public static T[] Filter<T>(this T[] data, int[] indexesToStay)
    {
        T[] newArr = new T[indexesToStay.Length];
        for (int i = 0; i < indexesToStay.Length; i++)
        {
            newArr[i] = data[indexesToStay[i]];
        }
        return newArr;
    }

    public static T[][] Filter<T>(this T[][] data, int[] indexesToStay)
    {
        T[][] newArr = new T[indexesToStay.Length][];
        for (int i = 0; i < indexesToStay.Length; i++)
        {
            newArr[i] = data[indexesToStay[i]];
        }
        return newArr;
    }

    public static int[] createIndexesToStay(this bool[] isIndexInResult)
    {
        List<int> a = new List<int>();
        for (int i = 0; i < isIndexInResult.Length; i++)
        {
            if (isIndexInResult[i])
            {
                a.Add(i);
            }
        }
        return a.ToArray();
    }
    public static int[] createIndexesToStay(this byte[] isIndexInResult)
    {
        List<int> a = new List<int>();
        for (int i = 0; i < isIndexInResult.Length; i++)
        {
            if (isIndexInResult[i] == 1)
            {
                a.Add(i);
            }
        }

        return a.ToArray();
    }

    public static void Swap<T>(this T[] arr, int a, int b)
    {
        T tmp = arr[a];
        arr[a] = arr[b];
        arr[b] = tmp;

    }

    public static T[][] Filter<T>(this T[][] data, bool[] isIndexInResult)
    {
        return data.Filter(isIndexInResult.createIndexesToStay());
    }
    public static T[] Filter<T>(this T[] data, bool[] isIndexInResult)
    {
        return data.Filter(isIndexInResult.createIndexesToStay());
    }

    public static T[] CreateCopy<T>(this T[] toCopy)
    {
        T[] arr = new T[toCopy.Length];
        Array.Copy(toCopy, arr, toCopy.Length);
        return arr;
    }

}

