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
using System.Web.Script.Serialization;


struct HeapData {
    public float val;
    public int label;
}

class Result {
    public float Accurarcy { get; set; }
    public float Compression { get; set; }
    public int[] Vectors { get; set; }
}


class Options {
    public string TrainSetPath { get; set; }
    public string TestSetPath { get; set; }
    public int K { get; set; }
    public float Alpha { get; set; }
    public int PopSize { get; set; }
    public int Iterations { get; set; }
}

class Program {

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

    //static void ReduceVectorsRegresion(CudaDataSet<float> teaching, CudaDataSet<float> test)
    //{
    //    using (CudaContext context = new CudaContext())
    //    {

    //        int popSize = 100;
    //        var deviceTeaching = new DeviceDataSet<float>(teaching);
    //        var deviceTest = new DeviceDataSet<float>(test);

    //        FlattArray<byte> initialPopulation;

    //        var acc= new VectorReductionAccuracyRegresion(context, deviceTeaching, deviceTest, popSize)
    //        {
    //            K = 3,
    //            CountToPass = 2
    //        };

    //        VectorReductionFitness fitnessFunc =
    //            new VectorReductionFitness(context, acc, popSize, deviceTeaching.length)
    //            {
    //                Alpha = 0.7f
    //            };

    //        initialPopulation = CreateRandomPopulation(popSize,deviceTeaching.length);

    //        var d = new Evolutionary2(context, fitnessFunc, initialPopulation)
    //        {
    //            Elitism = 0.1f,
    //            MutationRate = 0.05f
    //        };
    //        for (int i = 0; i < 10; i++)
    //        {
    //            Profiler.Start("iteration");
    //            d.CreateNewPopulation();
    //            Profiler.Stop("iteration");

    //        }
    //        Console.WriteLine(acc.BaseAccuracy());
    //        var best = d.FindFitest();
    //        Console.WriteLine(acc.GenAccuracy(best.index));
    //        Console.WriteLine(fitnessFunc.GenLength(best.index)/(float)deviceTeaching.length);

    //        Profiler.Print();

    //    }

    //}



    static Result ReduceVectors(CudaDataSet<int> teaching, CudaDataSet<int> test, Options options)
    {
        using (CudaContext context = new CudaContext())
        {
            DeviceDataSet<int> deviceTeaching = new DeviceDataSet<int>(teaching);
            DeviceDataSet<int> deviceTest = new DeviceDataSet<int>(test);

            FlattArray<byte> initialPopulation;

            VectorReductionAccuracy acc = new VectorReductionAccuracy(context, deviceTeaching, deviceTeaching, options)
            {
                K = options.K,
                CountToPass = (int)Math.Ceiling((double)options.K / 2)
            };

            VectorReductionFitness fitnessFunc =
                new VectorReductionFitness(context, acc, options, deviceTeaching.length)
                {
                    Alpha = options.Alpha
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



            initialPopulation = CreateRandomPopulation(options.PopSize, deviceTeaching.length);
            ////CreatePopulationBasedOnParent(parrent, popSize, 0.2f, 0.05f);

            var d = new Evolutionary2(context, fitnessFunc, initialPopulation)
            {
                Elitism = 0.001f,
                MutationRate = 0.001f
            };
            for (int i = 0; i < options.Iterations; i++)
            {
                Profiler.Start("iteration");
                d.CreateNewPopulation();
                Profiler.Stop("iteration");
            }

            ////float[] acccc = acc.CalculateAccuracy(Enumerable.Repeat((byte)1, deviceTeaching.length).ToArray(), 1);

            var best = d.FindFitest();
            options.PopSize = 1;
            VectorReductionAccuracy finnalAcc = new VectorReductionAccuracy(context, deviceTeaching, deviceTest, options)
            {
                K = options.K,
                CountToPass = (int)Math.Ceiling((double)options.K / 2)
            };

            finnalAcc.CalculateAccuracy(d.genGen(best.index), 0);
            byte[] gen = d.genGen(best.index);
            List<int> vectorIndexes = new List<int>();
            for (int i = 0; i < gen.Length; i++)
            {
                if (gen[i] != 0)
                {
                    vectorIndexes.Add(i);
                }
            }

            return new Result()
            {
                Vectors = vectorIndexes.ToArray(),
                Accurarcy = finnalAcc.GenAccuracy(0) / deviceTest.length,
                Compression = 1 - vectorIndexes.Count() / (float)deviceTeaching.length
            };
        }
    }

    static void Main(string[] args)
    {
        System.Globalization.CultureInfo customCulture = (System.Globalization.CultureInfo)System.Threading.Thread.CurrentThread.CurrentCulture.Clone();
        customCulture.NumberFormat.NumberDecimalSeparator = ".";
        System.Threading.Thread.CurrentThread.CurrentCulture = customCulture;


        var options = new JavaScriptSerializer().Deserialize<Options>(File.ReadAllText("./config.json"));

        DataSetHelper.CreateTrainingAndTestDataset("dataSets/iris.csv", 0.25f);

        var test = DataSetHelper.LoadDataSet(options.TestSetPath, true);
        var train = DataSetHelper.LoadDataSet(options.TrainSetPath, true);
        var result = ReduceVectors(train, test, options);
        Console.WriteLine(new JavaScriptSerializer().Serialize(result));
        File.WriteAllText("output.json", new JavaScriptSerializer().Serialize(result));
        //Profiler.Print();

    }
}
