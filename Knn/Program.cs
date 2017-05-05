using ManagedCuda;
using ManagedCuda.NPP.NPPsExtensions;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;



struct HeapData
{
    public float val;
    public int label;
}


class Program
{


    static float Knn(CudaDataSet teaching, CudaDataSet test, int k = 3, int threadsPerBlock = 256)
    {
        CudaContext context = new CudaContext();
        var kernel = context.LoadKernel("kernels/kernel.ptx", "knnKernal");
        kernel.GridDimensions = test.Vectors.GetLength(0) / threadsPerBlock + 1;
        kernel.BlockDimensions = threadsPerBlock;


        CudaDeviceVariable<float> teachingDevice = teaching.Vectors.Raw;
        CudaDeviceVariable<float> testDevice = test.Vectors.Raw;
        CudaDeviceVariable<int> labelDevice = teaching.Classes;
        CudaDeviceVariable<int> testLabels = test.Classes;

        CudaDeviceVariable<HeapData> heapMemory =
            new CudaDeviceVariable<HeapData>(test.Vectors.GetLength(0) * k);

        heapMemory.Memset(uint.MaxValue);


        kernel.Run(
            teachingDevice.DevicePointer,//teaching vectors
            teaching.Vectors.GetLength(0),//teaching count
            testDevice.DevicePointer,//testVectors
            test.Vectors.GetLength(0),//test count
            labelDevice.DevicePointer,//labels
            test.Vectors.GetLength(1),//vectorLen
            k,
            heapMemory.DevicePointer
        );

        FlattArray<HeapData> maxL =
            new FlattArray<HeapData>(heapMemory, k);

        int failureCount = 0;
        for (int i = 0; i < maxL.GetLength(0); i++)
        {
            var labelCounts = new Dictionary<int, int>();
            for (int j = 0; j < maxL.GetLength(1); j++)
            {
                if (labelCounts.ContainsKey(maxL[i, j].label))
                {
                    labelCounts[maxL[i, j].label]++;
                }
                else
                {
                    labelCounts[maxL[i, j].label] = 1;
                }
            }

            var guesedLabel = labelCounts.Aggregate((max, current) =>
            {
                if (current.Value > max.Value)
                {
                    return current;
                }
                else
                {
                    return max;
                }
            }).Key;

            if (guesedLabel != test.Classes[i])
            {
                failureCount++;
            }
        }

        context.Dispose();

        return (float)(test.Vectors.GetLength(0) - failureCount) / (float)test.Vectors.GetLength(0);

    }

    static void Main(string[] args)
    {

        System.Globalization.CultureInfo customCulture = (System.Globalization.CultureInfo)System.Threading.Thread.CurrentThread.CurrentCulture.Clone();
        customCulture.NumberFormat.NumberDecimalSeparator = ".";
        System.Threading.Thread.CurrentThread.CurrentCulture = customCulture;

        var data = DataSetHelper.readIris();
        DataSetHelper.Normalize(data);
        DataSetHelper.Shuffle(data);
        var splited = DataSetHelper.Split(data, new float[] { 0.75f, 0.25f });

        //  e.CalculateClasses(splited[0], splited[1]);

        splited[0].ResetIndeces();
        Drop3 drop = new Drop3();
        drop.CasheSize = 5;
        drop.K = 3;
        var indexesToStay = drop.Apply(splited[0]);
        var teaching = splited[0].Filter(indexesToStay);

        var parentGen = new byte[splited[0].Classes.Length];
        foreach (var item in indexesToStay)
        {
            parentGen[item] = 1;
        }

        Evolutionary e = new Evolutionary(splited[0],splited[1],100,parentGen);
        var res = Knn(teaching, splited[1], 3);

        Console.WriteLine(1- teaching.Classes.Length/(float) splited[0].Classes.Length);
        Console.WriteLine(res);


    }
}
