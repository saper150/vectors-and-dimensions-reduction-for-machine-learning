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


    //static float Knn(CudaDataSet teaching, CudaDataSet test, int k = 3, int threadsPerBlock = 256)
    //{
    //    CudaContext context = new CudaContext();
    //    var kernel = context.LoadKernel("kernels/kernel.ptx", "knnKernal");
    //    kernel.GridDimensions = test.Vectors.GetLength(0) / threadsPerBlock + 1;
    //    kernel.BlockDimensions = threadsPerBlock;


    //    CudaDeviceVariable<float> teachingDevice = teaching.Vectors.Raw;
    //    CudaDeviceVariable<float> testDevice = test.Vectors.Raw;
    //    CudaDeviceVariable<int> labelDevice = teaching.Classes;
    //    CudaDeviceVariable<int> testLabels = test.Classes;

    //    CudaDeviceVariable<HeapData> heapMemory =
    //        new CudaDeviceVariable<HeapData>(test.Vectors.GetLength(0) * k);

    //    heapMemory.Memset(uint.MaxValue);


    //    kernel.Run(
    //        teachingDevice.DevicePointer,//teaching vectors
    //        teaching.Vectors.GetLength(0),//teaching count
    //        testDevice.DevicePointer,//testVectors
    //        test.Vectors.GetLength(0),//test count
    //        labelDevice.DevicePointer,//labels
    //        test.Vectors.GetLength(1),//vectorLen
    //        k,
    //        heapMemory.DevicePointer
    //    );

    //    FlattArray<HeapData> maxL =
    //        new FlattArray<HeapData>(heapMemory, k);

    //    int failureCount = 0;
    //    for (int i = 0; i < maxL.GetLength(0); i++)
    //    {
    //        var labelCounts = new Dictionary<int, int>();
    //        for (int j = 0; j < maxL.GetLength(1); j++)
    //        {
    //            if (labelCounts.ContainsKey(maxL[i, j].label))
    //            {
    //                labelCounts[maxL[i, j].label]++;
    //            }
    //            else
    //            {
    //                labelCounts[maxL[i, j].label] = 1;
    //            }
    //        }

    //        var guesedLabel = labelCounts.Aggregate((max, current) =>
    //        {
    //            if (current.Value > max.Value)
    //            {
    //                return current;
    //            }
    //            else
    //            {
    //                return max;
    //            }
    //        }).Key;

    //        if (guesedLabel != test.Classes[i])
    //        {
    //            failureCount++;
    //        }
    //    }

    //    context.Dispose();

    //    return (float)(test.Vectors.GetLength(0) - failureCount) / (float)test.Vectors.GetLength(0);

    //}

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


    static void ReduceDimension(CudaDataSet<int> teaching, CudaDataSet<int> test)
    {

        using (CudaContext context = new CudaContext())
        {

            int popSize = 20;
            int genLength = teaching.Vectors.GetLength(1);

            DeviceDataSet<int> deviceTeaching = new DeviceDataSet<int>(teaching);
            DeviceDataSet<int> deviceTest = new DeviceDataSet<int>(test);


            FlattArray<byte> initialPopulation;

            DimensionReductionAccuracy accuracy = new DimensionReductionAccuracy(context, deviceTeaching, deviceTest, popSize)
            {
                K = 3,
                CountToPass = 2

            };

            DimensionReductionFitness fitnessFunc =
                new DimensionReductionFitness(context, accuracy, popSize, genLength)
                {
                    Alpha = 1f
                };


            Console.WriteLine(accuracy.BaseAccuracy());
            Console.WriteLine();
            initialPopulation = CreateRandomPopulation(popSize, deviceTeaching.attributeCount);



            var d = new Evolutionary2(context, fitnessFunc, initialPopulation)
            {
                CrossOverRate = 0.5f,
                MutationRate = 0.02f,
                Elitism = 0.2f
            };

            for (int i = 0; i < 200; i++)
            {
                Profiler.Start("iteration");
                d.CreateNewPopulation();
                Profiler.Stop("iteration");

            }
            var best = d.FindFitest();
            // Console.WriteLine(best.fitness);
            var acc = accuracy.GenAccuracy(best.index);
            var len = fitnessFunc.GenLength(best.index);
            var gen = d.genGen(best.index);
            foreach (var item in gen)
            {
                Console.Write(item + " ");
            }

            Console.WriteLine($"accuracy: {acc / (float)deviceTest.length} length: {len / (float)genLength}");
            Console.WriteLine();
            //  Profiler.Print();

        }



    }
    static void ReduceVectorsRegresion(CudaDataSet<float> teaching, CudaDataSet<float> test)
    {
        using (CudaContext context = new CudaContext())
        {

            int popSize = 100;
            var deviceTeaching = new DeviceDataSet<float>(teaching);
            var deviceTest = new DeviceDataSet<float>(test);

            FlattArray<byte> initialPopulation;

            var acc= new VectorReductionAccuracyRegresion(context, deviceTeaching, deviceTest, popSize)
            {
                K = 3,
                CountToPass = 2
            };

            VectorReductionFitness fitnessFunc =
                new VectorReductionFitness(context, acc, popSize, deviceTeaching.length)
                {
                    Alpha = 0.7f
                };

            initialPopulation = CreateRandomPopulation(popSize,deviceTeaching.length);

            var d = new Evolutionary2(context, fitnessFunc, initialPopulation)
            {
                Elitism = 0.1f,
                MutationRate = 0.05f
            };
            for (int i = 0; i < 10; i++)
            {
                Profiler.Start("iteration");
                d.CreateNewPopulation();
                Profiler.Stop("iteration");

            }
            Console.WriteLine(acc.BaseAccuracy());
            var best = d.FindFitest();
            Console.WriteLine(acc.GenAccuracy(best.index));
            Console.WriteLine(fitnessFunc.GenLength(best.index)/(float)deviceTeaching.length);

            Profiler.Print();

        }

    }



    static void ReduceVectors(CudaDataSet<int> teaching, CudaDataSet<int> test)
    {
        using (CudaContext context = new CudaContext())
        {

            int popSize = 100;
            DeviceDataSet<int> deviceTeaching = new DeviceDataSet<int>(teaching);
            DeviceDataSet<int> deviceTest = new DeviceDataSet<int>(test);

            FlattArray<byte> initialPopulation;

            VectorReductionAccuracy acc = new VectorReductionAccuracy(context, deviceTeaching, deviceTest, popSize) {
                K = 5,
                CountToPass = 3
            };

            VectorReductionFitness fitnessFunc =
                new VectorReductionFitness(context, acc, popSize, deviceTeaching.length)
                {
                    Alpha = 0.7f
                };

            //Drop3 drop = new Drop3();
            //drop.CasheSize = 5;
            //drop.K = 3;

            //Profiler.Start("Drop3");
            //var indexesToStay = drop.Apply(teaching, context);
            //Profiler.Stop("Drop3");


            //byte[] parrent = new byte[teaching.Vectors.GetLength(0)];
            //foreach (var item in indexesToStay)
            //{
            //    parrent[item] = 1;
            //}



            initialPopulation = CreateRandomPopulation(popSize, deviceTeaching.length);
                //CreatePopulationBasedOnParent(parrent, popSize, 0.2f, 0.05f);

            var d = new Evolutionary2(context, fitnessFunc, initialPopulation)
            {
                Elitism = 0.001f,
                MutationRate = 0.001f
            };
            for (int i = 0; i < 30; i++)
            {
                Profiler.Start("iteration");
                d.CreateNewPopulation();
                Profiler.Stop("iteration");

            }
            var best = d.FindFitest();
            Console.WriteLine(acc.GenAccuracy(best.index)/(float) deviceTest.length);
            Console.WriteLine(fitnessFunc.GenLength(best.index));

            Profiler.Print();

        }

    }

    static void Main(string[] args)
    {




        System.Globalization.CultureInfo customCulture = (System.Globalization.CultureInfo)System.Threading.Thread.CurrentThread.CurrentCulture.Clone();
        customCulture.NumberFormat.NumberDecimalSeparator = ".";
        System.Threading.Thread.CurrentThread.CurrentCulture = customCulture;

        {
            var data = DataSetHelper.ReadMagic();
            DataSetHelper.Normalize(data);
            DataSetHelper.Shuffle(data);
            var splited = DataSetHelper.Split(data, new float[] { 0.7f, 0.3f });
            splited[0].ResetIndeces();
            ReduceVectors(splited[0], splited[1]);
        }

        {
            var regresionData = DataSetHelper.ReadHouse();
            DataSetHelper.Normalize(regresionData);
            DataSetHelper.Shuffle(regresionData);
            var regresionSplited = DataSetHelper.Split(regresionData, new float[] { 0.75f, 0.25f });

            //ReduceVectorsRegresion(regresionSplited[0], regresionSplited[1]);
        }
        //ReduceDimension(splited[0],splited[1]);


    }
}
