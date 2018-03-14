using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


public struct Data
{
    public int index;
    public float val;

}



public class Sorting
{

    private static CudaDataSet<T> copyResult<T>(CudaDataSet<T> src, int[] sorted)
    {

        FlattArray<float> newArr =
            new FlattArray<float>(src.Vectors.GetLength(0), src.Vectors.GetLength(1));

        T[] newVal = new T[src.Vectors.GetLength(0)];

        int attrCount = src.Vectors.GetLength(1);
        Parallel.For(0, sorted.Length, i =>
        {
            Array.Copy(
                src.Vectors.Raw,
                sorted[i] * attrCount,
                newArr.Raw,
                i * attrCount,
                attrCount
                );

            newVal[i] = src.Classes[sorted[i]];

        });

        return new CudaDataSet<T>()
        {
            Vectors = newArr,
            Classes = newVal
        };

    }
    private static int[] SortingIndeces(float[] s)
    {

        int[] indexes = new int[s.Length];
        for (int i = 0; i < indexes.Length; i++)
        {
            indexes[i] = i;
        }
        var copy = s.CreateCopy();
        Array.Sort(copy, indexes);
        return indexes;

    }
    private static int[] SortingIndecesDesc(float[] s)
    {
        int[] indexes = new int[s.Length];
        for (int i = 0; i < indexes.Length; i++)        
        {
            indexes[i] = i;
        }
        var copy = s.CreateCopy();
        Array.Sort(copy,indexes);
        Array.Reverse(indexes);
        return indexes;

    }

    private static T[][] copyData<T>(T[][] src, int[] sorted)
    {
        var newArr = new T[src.Length][];

        for (int i = 0; i < newArr.Length; i++)
        {
            newArr[i] = src[sorted[i]];
        }
        return newArr;
    }


    public static void Sort(HostDataset host,int[] sortingIndeces) {
        float[][] sortedVectors = new float[host.Vectors.Length][];
        int[] sortedClasses = new int[host.Classes.Length];
        int[] sortedIndeces = new int[host.OrginalIndeces.Length];

        for (int i = 0; i < sortingIndeces.Length; i++)
        {
            sortedVectors[i] = host.Vectors[sortingIndeces[i]];
            sortedClasses[i] = host.Classes[sortingIndeces[i]];
            sortedIndeces[i] = host.OrginalIndeces[sortingIndeces[i]];

        }
        host.Vectors = sortedVectors;
        host.Classes = sortedClasses;
        host.OrginalIndeces = sortedIndeces;

    }
    public static void SortDesc(HostDataset host,float[] sortBy) {
        Sort(host, SortingIndecesDesc(sortBy));
    }
    public static void Sort(HostDataset host,float[] sortBy) {
        Sort(host, SortingIndeces(sortBy));
    }

    public static void Remap(int[][] toRemap,int[] remapBy) {

        for (int i = 0; i < toRemap.Length; i++)
        {
            for (int j = 0; j < toRemap[i].Length; j++)
            {
                toRemap[i][j] = remapBy[toRemap[i][j]];
            }
        }
    }

    public static void sortAndRemap(int[][] toSort,float[] sortBy)
    {
        var copy = sortBy.CreateCopy();
        var sortingIndeces = SortingIndeces(copy);
        Array.Sort(copy, toSort);
        Remap(toSort, sortingIndeces);

    }
    public static void sortAndRemapDesc(int[][] toSort, float[] sortBy)
    {
        var copy = sortBy.CreateCopy();
        var sortingIndeces = SortingIndecesDesc(copy);
        Array.Sort(copy, toSort);
        Array.Reverse(toSort);
        Remap(toSort, sortingIndeces);
    }


    public static T[][] Sort<T>(T[][] toSort, float[] sortBy)
    {
        return copyData(toSort, SortingIndeces(sortBy));
    }
    public static T[][] SortDesc<T>(T[][] toSort, float[] sortBy)
    {
        return copyData(toSort, SortingIndecesDesc(sortBy));
    }

    public static CudaDataSet<T> Sort<T>(CudaDataSet<T> toSort, float[] sortBy)
    {

        return copyResult(toSort, SortingIndeces(sortBy));

    }
    public static CudaDataSet<T> SortDesc<T>(CudaDataSet<T> toSort, float[] sortBy)
    {

        return copyResult(toSort, SortingIndecesDesc(sortBy));
    }

}

