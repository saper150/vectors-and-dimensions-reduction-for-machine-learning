using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;


struct subject {
    public float fitness;
    public int accuracy;
    public int length;
}

class Evolutionary : IDisposable
{
    CudaContext context;

    CudaDataSet teaching;
    CudaDataSet test;

    #region deviceMemories

    CudaDeviceVariable<float> deviceTeachingVectors;
    CudaDeviceVariable<int> deviceTeachingClasses;
    CudaDeviceVariable<int> neaboursIndexes;


    CudaDeviceVariable<float> deviceTestVectors;
    CudaDeviceVariable<int> deviceTestClasses;


    CudaDeviceVariable<byte> populationGens;
    CudaDeviceVariable<byte> populationGens2;


    CudaDeviceVariable<int> deviceAccuracy;
    CudaDeviceVariable<int> deviceVectorSizes;
    CudaDeviceVariable<float> deviceFitnes;

    #endregion

    CudaKernel calculateNearestNeabours;
    CudaKernel createInitialPopulationKernel;
    CudaKernel calculateAccuracyKernel;
    CudaKernel calculateGenLengthsKernel;

    CudaKernel performGeneticAlgorythm;

    int threadsPerBlock = 256;
  


    int popSize;
    int k = 7;
    int countToPass = 4;
    float alpha = 0.8f;
    float mutationRate = 0.01f;
    float crosoverRate = 0.7f;


    int genLength;

    Profiler profiler = new Profiler();

    public void AllocateMemory()
    {

        deviceTeachingVectors = teaching.Vectors.Raw;
        deviceTestVectors = test.Vectors.Raw;
        deviceTeachingClasses = teaching.Classes;
        deviceTestClasses = test.Classes;

        neaboursIndexes =
            new CudaDeviceVariable<int>(teaching.Classes.Length * test.Classes.Length);

        populationGens = 
            new CudaDeviceVariable<byte>(popSize*teaching.Vectors.GetLength(0));

        populationGens2 =
            new CudaDeviceVariable<byte>(popSize * teaching.Vectors.GetLength(0));

        deviceAccuracy = new CudaDeviceVariable<int>(popSize);
        deviceVectorSizes = new CudaDeviceVariable<int>(popSize);
        deviceFitnes = new CudaDeviceVariable<float>(popSize);

    }
    public void CreateInitialPopulation(byte[] parrent) {

        if (parrent.Length != teaching.Vectors.GetLength(0))
            throw new Exception("parent is incorrect length");

        using (CudaDeviceVariable<byte> deviceParrent = parrent) {

            createInitialPopulationKernel.Run(
                deviceParrent.DevicePointer,
                populationGens.DevicePointer
                );
            var population = new FlattArray<byte>((byte[])populationGens, genLength).To2d() ;
        }

    }

    public void createRandomPopulation() {
        FlattArray<byte> population = 
            new FlattArray<byte>(popSize, teaching.Classes.Length);
        Random r = new Random();

        for (int i = 0; i < population.GetLength(0); i++)
        {
            for (int j = 0; j < population.GetLength(1); j++)
            {
                population[i, j] = r.NextDouble() > 0.5 ? (byte)1 : (byte)0;
            }
        }
        populationGens = population.Raw;

    }

    public void LoadKernels()
    {


        #region calculateNearestNeaboursKernel
        {
            calculateNearestNeabours =
                context.LoadKernel("kernels/evolutionary.ptx", "calculateNearestNeabours");
            calculateNearestNeabours.GridDimensions =
                test.Vectors.GetLength(0) / threadsPerBlock + 1;
            calculateNearestNeabours.BlockDimensions = threadsPerBlock;
            setConstants(calculateNearestNeabours);
        }
        #endregion

        #region geneticAlgorythmKernel

        {
            performGeneticAlgorythm = context.LoadKernel("kernels/evolutionary.ptx", "genetic");
            performGeneticAlgorythm.GridDimensions = 1;
            performGeneticAlgorythm.BlockDimensions = popSize;
            performGeneticAlgorythm.DynamicSharedMemory =
                (uint)(sizeof(float) * popSize);
            setConstants(performGeneticAlgorythm);
        }
        #endregion

        #region accuracyKernel
        {
            calculateAccuracyKernel = context.LoadKernel("kernels/evolutionary.ptx", "calculateAccuracy");
            setConstants(calculateAccuracyKernel);

            dim3 gridDimension = new dim3()
            {
                x = (uint)(test.Vectors.GetLength(0) / threadsPerBlock + 1),
                y = (uint)popSize,
                z = 1
            };
            calculateAccuracyKernel.GridDimensions = gridDimension;
            calculateAccuracyKernel.BlockDimensions = threadsPerBlock;

        }
        #endregion

        #region genLengthKernel
        {
            calculateGenLengthsKernel =
                context.LoadKernel("kernels/evolutionary.ptx", "countVectors");
            calculateGenLengthsKernel.BlockDimensions = popSize;
            calculateGenLengthsKernel.GridDimensions = 1;

            setConstants(calculateGenLengthsKernel);
        }
        #endregion

        #region initialPopulationkernel
        {
            createInitialPopulationKernel = context.LoadKernel("kernels/evolutionary.ptx", "createInitialPopulation");
            createInitialPopulationKernel.GridDimensions = 1;
            createInitialPopulationKernel.BlockDimensions = popSize;
            setConstants(createInitialPopulationKernel);
        }
        #endregion

    }


    private void setConstants(CudaKernel kernel) {

        kernel.SetConstantVariable("popSize", popSize);
        kernel.SetConstantVariable("genLength", genLength);
        kernel.SetConstantVariable("k", k);
        kernel.SetConstantVariable("countToPass", countToPass);
        kernel.SetConstantVariable("testVectorsCount", test.Classes.Length);
        kernel.SetConstantVariable("teachingVectorsCount", teaching.Classes.Length);
        kernel.SetConstantVariable("attributeCount", teaching.Vectors.GetLength(1));

    }
    public Evolutionary(
        CudaDataSet teaching, 
        CudaDataSet test,
        int popSize,
        byte[] parrent
        )
    {
        context = new CudaContext();

        this.teaching = teaching;
        this.test = test;
        this.popSize = popSize;
        this.genLength = teaching.Classes.Length;

        LoadKernels();
        AllocateMemory();

        profiler.Start("calculate classes");
        CalculateClasses();
        profiler.Stop("calculate classes");

        //CreateInitialPopulation(parrent);
        profiler.Start("create random population");
        createRandomPopulation();
        profiler.Stop("create random population");


        for (int i = 0; i < 100; i++)
        {
            profiler.Start("iteration");
            Genetics();
            profiler.Stop("iteration");

        }
        profiler.Print();
        subject best = FindFitest();
        Console.WriteLine($"best fitness:\t{best.fitness}");
        Console.WriteLine($"best accuracy:\t{best.accuracy}");
        Console.WriteLine($"shortest:\t{best.length}");

    }


    private void CalculateClasses()
    {

        using (CudaDeviceVariable<float> deviceDistancesMemory =
            new CudaDeviceVariable<float>(teaching.Classes.Length * test.Classes.Length))
        {

            calculateNearestNeabours.Run(
                deviceTeachingVectors.DevicePointer,
                deviceTestVectors.DevicePointer,
                deviceDistancesMemory.DevicePointer,
                neaboursIndexes.DevicePointer
                );

        }

    }


    public void CalculateAccuracy()
    {
        context.ClearMemory(deviceAccuracy.DevicePointer, 0, deviceAccuracy.SizeInBytes);

        calculateAccuracyKernel.Run(
            deviceTestClasses.DevicePointer,
            deviceTeachingClasses.DevicePointer,
            populationGens.DevicePointer,
            neaboursIndexes.DevicePointer,
            deviceAccuracy.DevicePointer
            );

    }

    public void CalculateVectorLengths() {

        calculateGenLengthsKernel.Run(
            populationGens.DevicePointer,
            deviceVectorSizes.DevicePointer
            );

    }

    private void Genetics() {

        profiler.Start("calculate gen lengths");
        CalculateVectorLengths();
        profiler.Stop("calculate gen lengths");

        profiler.Start("calculate accuracy");
        CalculateAccuracy();
        profiler.Stop("calculate accuracy");

        profiler.Start("calculate avrageAccuracy");
        float avgAccuracy = Thrust.Avrage(deviceAccuracy);
        profiler.Stop("calculate avrageAccuracy");

        profiler.Start("calculate avrageGenLength");
        float avgVectorLen = Thrust.Avrage(deviceVectorSizes);
        profiler.Stop("calculate avrageGenLength");

        profiler.Start("performin genetic algorythm");
        performGeneticAlgorythm.Run(
            deviceAccuracy.DevicePointer,
            deviceVectorSizes.DevicePointer,
            alpha,
            populationGens.DevicePointer,
            populationGens2.DevicePointer,
            crosoverRate,
            mutationRate,
            avgAccuracy,
            avgVectorLen,
            deviceFitnes.DevicePointer
            );
        profiler.Stop("performin genetic algorythm");


        var tmp = populationGens;
        populationGens = populationGens2;
        populationGens2 = tmp;

        profiler.Start("finding fitest");
        subject best = FindFitest();
        profiler.Stop("finding fitest");


        //Console.WriteLine($"best fitness:\t{best.fitness}");
        //Console.WriteLine($"best accuracy:\t{best.accuracy}");
        //Console.WriteLine($"shortest:\t{best.length}");


        Console.WriteLine();

    }
    [DllImport("kernels/CudaFunctions.dll", EntryPoint = "FindFitest")]
    static extern subject __findFitest(
        ManagedCuda.BasicTypes.SizeT ptr1,
        ManagedCuda.BasicTypes.SizeT ptr2,
        ManagedCuda.BasicTypes.SizeT ptr3,
        int size
        );

    public subject FindFitest()
    {
        return __findFitest(
            deviceFitnes.DevicePointer.Pointer,
            deviceAccuracy.DevicePointer.Pointer,
            deviceVectorSizes.DevicePointer.Pointer,
            deviceVectorSizes.Size
            );
    }


    public void Dispose()
    {
        context.Dispose();
    }
}

