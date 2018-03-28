using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

public static class CudaDataSetExtensions {
    public static CudaDataSet<T> Filter<T>(
        this CudaDataSet<T> data,
        int[] indexesToStay)
    {
        return new CudaDataSet<T>()
        {
            Vectors = data.Vectors.Filter(indexesToStay),
            Classes = data.Classes.Filter(indexesToStay),
            orginalIndeces = data.orginalIndeces.Filter(indexesToStay)

        };

    }

    public static void ResetIndeces<T>(this CudaDataSet<T> set)
    {
        set.orginalIndeces = DataSetHelper.CreateIndeces(set);
    }

    public static HostDataset ToHostDataSet(this CudaDataSet<int> set)
    {
        return new HostDataset()
        {
            Vectors = set.Vectors.To2d(),
            Classes = set.Classes,
            OrginalIndeces = set.orginalIndeces
        };

    }


}


public class CudaDataSet<T> {
    public FlattArray<float> Vectors;
    public T[] Classes;
    public int[] orginalIndeces;
}

public class HostDataset {
    public float[][] Vectors;
    public int[] Classes;
    public int[] OrginalIndeces;
}

public static class DataSetHelper {
    static Random rand = new Random();

    public static void Shuffle<T>(CudaDataSet<T> data)
    {

        for (int i = 0; i < data.Classes.Length; i++)
        {
            var toSwap = rand.Next(data.Classes.Length - i);
            data.Swap(toSwap, i);
        }

    }

    private static string RowToString(CudaDataSet<int> set, int row)
    {
        return string.Join(",",
             set.Vectors.To2d()[row]
             .Select(x => x.ToString())
             .Concat(new string[] { set.Classes[row].ToString() })
        );
    }


    public static CudaDataSet<int>[] PrepareSet(string path, float ration)
    {
        var data = LoadDataSet(path);
        Normalize(data);
        Shuffle(data);
        return Split(data, new float[] { ration, 1 - ration });
    }

    public static void SaveDataSet(CudaDataSet<int> set, string path)
    {
        using (var writer = new StreamWriter(path))
        {
            writer.WriteLine(
                string.Join(",", Enumerable.Repeat("param" , set.Vectors.GetLength(1)).ToArray()) + ",label"
            );
            for (int i = 0; i < set.Classes.Length; i++)
            {
                var row = RowToString(set, i);
                writer.WriteLine(RowToString(set, i));
            }
        }
    }

    public static void CreateTrainingAndTestDataset(string path, float ration)
    {
        var result = PrepareSet(path, ration);
        string fileName = Path.GetFileNameWithoutExtension(path);
        SaveDataSet(result[0], path.Replace(fileName, fileName + "-test"));
        SaveDataSet(result[1], path.Replace(fileName, fileName + "-train"));
    }

    public static CudaDataSet<int> LoadDataSet(string path, bool preserveClasses = false)
    {
        var lines = File.ReadLines(path);
        int columnCount = lines.First().Split(',').Length;

        var labels = Enumerable.Repeat(Type.param, columnCount - 1)
            .Concat(new Type[] { Type.label })
            .ToArray();

        var reader = new LabelReader(labels)
        {
            Header = true,
            PreserveClasses = preserveClasses
        };
        reader.ReadFile(path);
        return reader.DataSet;
    }

    public static CudaDataSet<T>[] Split<T>(CudaDataSet<T> data, float[] parts)
    {
        var result = new CudaDataSet<T>[parts.Length];
        var splitedClasses = Split(data.Classes, parts);
        var splitedVectors = Split(data.Vectors, parts);
        var splitedIndeces = Split(data.orginalIndeces, parts);

        for (int i = 0; i < splitedClasses.Length; i++)
        {
            result[i] = new CudaDataSet<T>()
            {
                Vectors = splitedVectors[i],
                Classes = splitedClasses[i],
                orginalIndeces = splitedIndeces[i]
            };
        }
        return result;

    }
    public static FlattArray<T>[] Split<T>(FlattArray<T> data, float[] parts)
    {
        var splited = new FlattArray<T>[parts.Length];
        int startIndex = 0;

        for (int i = 0; i < parts.Length; i++)
        {
            int endIndex = startIndex +
                (int)Math.Round(data.GetLength(0) * parts[i]);

            if (endIndex >= data.GetLength(0))
            {
                endIndex = data.GetLength(0);
            }
            int len = endIndex - startIndex;

            T[] arr = new T[len * data.GetLength(1)];
            Array.Copy(data.Raw,
                startIndex * data.GetLength(1),
                arr,
                 0,
                 len * data.GetLength(1)
                );
            startIndex = endIndex;
            splited[i] = new FlattArray<T>(arr, data.GetLength(1));


        }
        return splited;
    }


    public static T[][] Split<T>(T[] data, float[] parts)
    {
        var splited = new T[parts.Length][];
        int startIndex = 0;

        for (int i = 0; i < parts.Length; i++)
        {
            int endIndex = startIndex +
                (int)Math.Round(data.Length * parts[i]);
            if (endIndex >= data.Length)
            {
                endIndex = data.Length;
            }
            int len = endIndex - startIndex;
            splited[i] = new T[len];
            Array.Copy(data, startIndex, splited[i], 0, len);
            startIndex = endIndex;

        }
        return splited;
    }


    private static void Swap<T>(this CudaDataSet<T> data, int row1, int row2)
    {
        data.Vectors.Swap(row1, row2);
        data.Classes.Swap(row1, row2);
        data.orginalIndeces.Swap(row1, row2);

    }

    public static void Normalize<T>(CudaDataSet<T> data)
    {


        var vectors = data.Vectors;
        float[] avg = ColumnsAvrages(data);


        float[] standardDeviation = new float[vectors.GetLength(0)];

        for (int row = 0; row < standardDeviation.GetLength(0); row++)
        {
            for (int col = 0; col < vectors.GetLength(1); col++)
            {
                var difference = vectors[row, col] - avg[col];
                standardDeviation[col] += difference * difference;
            }
        }

        for (int i = 0; i < vectors.GetLength(1); i++)
        {
            standardDeviation[i] = (float)Math.Sqrt(standardDeviation[i] / vectors.GetLength(0));
        }

        for (int row = 0; row < standardDeviation.GetLength(0); row++)
        {
            for (int col = 0; col < vectors.GetLength(1); col++)
            {
                vectors[row, col] = (vectors[row, col] - standardDeviation[col]) / avg[col];
            }
        }
    }


    public static float[] Variances(FlattArray<float> vectors)
    {

        float[] variances = new float[vectors.GetLength(1)];
        Parallel.For(0, vectors.GetLength(1), col =>
        {
            float avrage = ColumnAvrage(vectors, col);
            float[] diffrencesSquared = new float[vectors.GetLength(0)];
            for (int row = 0; row < vectors.GetLength(0); row++)
            {
                float f = vectors[row, col] - avrage;
                diffrencesSquared[row] = f * f;
            }
            variances[col] = diffrencesSquared.Average();

        });

        return variances;
    }


    public static float ColumnAvrage(FlattArray<float> data, int column)
    {
        float avg = 0;
        for (int i = 0; i < data.GetLength(0); i++)
        {
            avg += data[i, column];
        }
        return avg / data.GetLength(0);

    }

    public static float[] ColumnsAvrages<T>(CudaDataSet<T> data)
    {

        var vectors = data.Vectors;
        float[] avg = new float[vectors.GetLength(1)];

        for (int i = 0; i < vectors.GetLength(1); i++)
        {
            avg[i] = ColumnAvrage(vectors, i);
        }

        return avg;
    }

    public static int[] CreateIndeces<T>(CudaDataSet<T> set)
    {
        int len = set.Classes.Length;
        int[] res = new int[len];

        for (int i = 0; i < len; i++)
        {
            res[i] = i;
        }

        return res;
    }
    public static CudaDataSet<int> readIris()
    {
        var irisDescription = new Type[] {
                Type.param,
                Type.param,
                Type.param,
                Type.param,
                Type.label
            };

        LabelReader reader = new LabelReader(irisDescription);
        reader.ReadFile("dataSets/iris.csv");
        return reader.DataSet;
    }
    public static CudaDataSet<int> readPoker()
    {
        var pokerDescription = new Type[] {
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.label
        };



        LabelReader reader = new LabelReader(pokerDescription);
        reader.ReadFile("dataSets/poker.csv");
        return reader.DataSet;
    }

    public static CudaDataSet<int> ReadMagic()
    {
        var magidDescription = new Type[] {
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.label,

        };

        LabelReader reader = new LabelReader(magidDescription);
        reader.ReadFile("dataSets/magic.csv");
        return reader.DataSet;

    }

    public static CudaDataSet<int> ReadPenBase()
    {
        var PenDescription = new Type[] {
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.param,
            Type.label,
        };

        LabelReader reader = new LabelReader(PenDescription);
        reader.ReadFile("dataSets/penbased.csv");
        return reader.DataSet;
    }

    public static CudaDataSet<int> ReadNormalizedIris()
    {
        var irisDescription = new Type[] {
                Type.param,
                Type.param,
                Type.param,
                Type.param,
                Type.label
            };

        LabelReader reader = new LabelReader(irisDescription);
        //reader.Separator = '\t';
        reader.ReadFile("dataSets/iris_std_sh.txt");
        return reader.DataSet;

    }


    public static CudaDataSet<float> ReadHouse()
    {

        var HouseDescription = new List<Type>();
        for (int i = 0; i < 16; i++)
        {
            HouseDescription.Add(Type.param);
        }
        HouseDescription.Add(Type.label);


        RegresionReader reader = new RegresionReader(HouseDescription.ToArray());
        reader.ReadFile("dataSets/house.csv");
        return reader.DataSet;
    }

    //public static CudaDataSet<float> ReadCompactiv()
    //{

    //    var HouseDescription = new List<Type>();
    //    for (int i = 0; i < 16; i++)
    //    {
    //        HouseDescription.Add(Type.param);
    //    }
    //    HouseDescription.Add(Type.label);


    //    RegresionReader reader = new RegresionReader(HouseDescription.ToArray());
    //    reader.ReadFile("dataSets/penbased.csv");
    //    return reader.DataSet;
    //}



}