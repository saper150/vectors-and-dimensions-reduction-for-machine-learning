using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;



public static class FlattArrayExtensions
{
    public static FlattArray<T> Filter<T>(
        this FlattArray<T> toFilter,
        int[] indexesToStay)
    {
        int colCount = toFilter.GetLength(1);
        T[] newData = new T[indexesToStay.Length * colCount];

        for (int i = 0; i < indexesToStay.Length; i++)
        {
            Array.Copy(
                toFilter.Raw,
                indexesToStay[i] * colCount,
                newData,
                i * colCount,
                colCount
                );
        }
        return new FlattArray<T>(newData,colCount);

    }

    public static void Swap<T>(this FlattArray<T> arr, int row1, int row2) {

        var tmpVector = new T[arr.GetLength(1)];
        Array.Copy(
            arr.Raw,
            arr.GetLength(1) * row1,
            tmpVector,
            0,
            arr.GetLength(1)
            );

        Array.Copy(
           arr.Raw,
           arr.GetLength(1) * row2,
           arr.Raw,
           arr.GetLength(1) * row1,
           arr.GetLength(1)
           );

        Array.Copy(
           tmpVector,
           0,
           arr.Raw,
           arr.GetLength(1) * row2,
           arr.GetLength(1)
        );


    }


    public static T[][] To2d<T>(this FlattArray<T> arr)
    {

        T[][] res = new T[arr.GetLength(0)][];

        int colCount = arr.GetLength(1);
        for (int i = 0; i < res.Length; i++)
        {
            res[i] = new T[colCount];
            Array.Copy(
                arr.Raw,
                i*colCount,
                res[i],
                0,
                colCount
                );
        }
        return res;

    }



}




public class FlattArray<T>
{
    public FlattArray(T[,] arr)
    {
        Raw = new T[arr.GetLength(0) * arr.GetLength(1)];
        lengths = new int[] { arr.GetLength(0), arr.GetLength(1) };

        for (int i = 0; i < arr.GetLength(0); i++)
        {
            for (int j = 0; j < arr.GetLength(1); j++)
            {
                Raw[arr.GetLength(1) * i + j] = arr[i, j];
            }
        }
    }
    public FlattArray(T[][] data)
    {
        Raw = new T[data.Length * data[0].Length];
        lengths = new int[] { data.Length, data[0].Length };

        for (int i = 0; i < data.Length; i++)
        {
            Array.Copy(
                data[i],
                0,
                Raw,
                GetLength(1) * i,
                GetLength(1)
            );
        }

    }

    public FlattArray(int row,int col)
    {
        Raw = new T[row * col];
        lengths = new int[] {row, col};
    }
    public FlattArray(T[] arr, int col)
    {
        if (arr.Length % col != 0)
            throw new Exception("size dont match dimensions");
        Raw = arr;
        lengths = new int[] { arr.Length / col, col };


    }


    public T this[int i, int j]
    {
        get
        {
            return Get(i, j);
        }
        set {
            int index = i * lengths[1] + j;
            Raw[index] = value;
        }
    }

    public T Get(int row, int col)
    {

        int index = row * lengths[1] + col;

        return Raw[index];

    }

    public int GetLength(int i) => lengths[i];
    public T[] Raw { get; set; }

    int[] lengths;

}