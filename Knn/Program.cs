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

    public static FlattArray<byte> CreateRandomPopulation(int popSize, int genLength)
    {
        FlattArray<byte> population =
            new FlattArray<byte>(popSize, genLength);
        Random r = new Random();

        for (int i = 0; i < population.GetLength(0); i++)
        {
            for (int j = 0; j < population.GetLength(1); j++)
            {
                population[i, j] = r.NextDouble() < 0.5 ? (byte)1 : (byte)0;
            }
        }
        return population;

    }


    public static FlattArray<byte> CreatePopulationBasedOnParent(byte[] parent, int popSize, float flippChance, float bias)
    {

        FlattArray<byte> population =
             new FlattArray<byte>(popSize, parent.Length);
        Random r = new Random();


        for (int i = 0; i < population.GetLength(0); i++)
        {
            for (int j = 0; j < population.GetLength(1); j++)
            {
                float chance = population[i, j] == 1 ? flippChance : flippChance - bias;
                population[i, j] = r.NextDouble() < chance ? (byte)1 : (byte)0;
            }
        }

        return population;

    }



    static void Main(string[] args)
    {




        System.Globalization.CultureInfo customCulture = (System.Globalization.CultureInfo)System.Threading.Thread.CurrentThread.CurrentCulture.Clone();
        customCulture.NumberFormat.NumberDecimalSeparator = ".";
        System.Threading.Thread.CurrentThread.CurrentCulture = customCulture;


        var data = DataSetHelper.ReadMagic();
        DataSetHelper.Normalize(data);
        DataSetHelper.Shuffle(data);
        var splited = DataSetHelper.Split(data, new float[] { 0.75f, 0.25f });

        splited[0].ResetIndeces();


        int popSize = 100;

        using (CudaContext context = new CudaContext())
        {

            DeviceDataSet teaching = new DeviceDataSet(splited[0]);
            DeviceDataSet test = new DeviceDataSet(splited[1]);


            FlattArray<byte> initialPopulation;
            IFitnessFunction fitnessFunc;


            {
                fitnessFunc =
                    new VectorReductionFitness(context, teaching, test, popSize)
                    {
                        Alpha = 0.7f
                    };

                Drop3 drop = new Drop3();
                drop.CasheSize = 5;
                drop.K = 3;

                Profiler.Start("Drop3");
                var indexesToStay = drop.Apply(splited[0], context);
                Profiler.Stop("Drop3");


                byte[] parrent = new byte[splited[0].Vectors.GetLength(0)];
                foreach (var item in indexesToStay)
                {
                    parrent[item] = 1;
                }

                initialPopulation = CreatePopulationBasedOnParent(parrent, popSize, 0.2f, 0.05f);

            }



            //{

            //    fitnessFunc = new DimensionReductionFitness(context, teaching,test,popSize);
            //    initialPopulation = CreateRandomPopulation(popSize, teaching.attributeCount);


            //}

            var d = new Evolutionary2(context, fitnessFunc, initialPopulation);
            for (int i = 0; i < 100; i++)
            {
                Profiler.Start("iteration");
                d.CreateNewPopulation();
                Profiler.Stop("iteration");

            }
            var best = d.FindFitest();
            Console.WriteLine(fitnessFunc.FitnessDetails(best.index));
            Profiler.Print();

        }


    }
}
