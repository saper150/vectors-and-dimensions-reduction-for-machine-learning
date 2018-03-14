using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;



struct subject
{
    public float fitness;
    public int index;
}


class DeviceDataSet<T> where T :struct
{
    public DeviceDataSet(CudaDataSet<T> data)
    {
        vectors = data.Vectors.Raw;
        classes = data.Classes;
        attributeCount = data.Vectors.GetLength(1);
        length = data.Vectors.GetLength(0);
    }
    public int attributeCount;
    public int length;

    public CudaDeviceVariable<float> vectors;
    public CudaDeviceVariable<T> classes;
}


class Evolutionary2
{
    CudaContext context;


    CudaDeviceVariable<byte> populationGens;
    CudaDeviceVariable<byte> populationGens2;

    CudaDeviceVariable<float> deviceFitnes;
    CudaDeviceVariable<int> fitnessIndeces;

    CudaKernel performGeneticAlgorythm;


    int popSize;

    float _alpha;
    public float Alpha
    {
        get
        {
            return _alpha;
        }
        set
        {
            _alpha = value;
            performGeneticAlgorythm.SetConstantVariable("alpha", _alpha);

        }
    }

    float _mutationRate;
    public float MutationRate
    {
        get
        {
            return _mutationRate;
        }
        set
        {
            _mutationRate = value;
            performGeneticAlgorythm.SetConstantVariable("mutationRate", _mutationRate);

        }
    }
    float _elitism;
    public float Elitism
    {
        get { return _elitism; }
        set
        {
            _elitism = value;

            performGeneticAlgorythm.SetConstantVariable("eliteIndex", (genLength-(int)(genLength * _elitism)));

        }
    }


    float _crossOverRate;
    public float CrossOverRate
    {
        get
        {
            return _crossOverRate;
        }
        set
        {
            _crossOverRate = value;
            performGeneticAlgorythm.SetConstantVariable("crossoverRate", _crossOverRate);

        }
    }


    IFitnessFunction fitnessCalc;

    int genLength;

    public Evolutionary2(
       CudaContext context,
       IFitnessFunction fitnessCalc,
       FlattArray<byte> initialPopulation
       )
    {
        this.context = context;

        this.popSize = initialPopulation.GetLength(0);
        this.genLength = initialPopulation.GetLength(1);


        int alignedPopSizeMemory = (popSize * genLength) + ((popSize * genLength) % (sizeof(int)));

        populationGens =
            new CudaDeviceVariable<byte>(alignedPopSizeMemory);
        populationGens2 =
            new CudaDeviceVariable<byte>(alignedPopSizeMemory);

        context.CopyToDevice(populationGens2.DevicePointer, initialPopulation.Raw);
        //initialPopulation.Raw;

        deviceFitnes = new CudaDeviceVariable<float>(popSize);
        fitnessIndeces = new CudaDeviceVariable<int>(popSize);


        LoadKernels();

        MutationRate = 0.01f;
        CrossOverRate = 0.7f;
        Alpha = 0.7f;
        Elitism = 0.2f;

        this.fitnessCalc = fitnessCalc;

        performGeneticAlgorythm.SetConstantVariable("popSize", popSize);
        performGeneticAlgorythm.SetConstantVariable("genLength", genLength);

    }


    public void CreateNewPopulation() {
        var tmp = populationGens;
        populationGens = populationGens2;
        populationGens2 = tmp;

       // var b = new FlattArray<byte>((byte[])populationGens, genLength).To2d();
        Profiler.Start("Calculate fitness");
        fitnessCalc.CalculateFitness(populationGens, deviceFitnes);
        Profiler.Stop("Calculate fitness");

        Profiler.Start("sqauance");
        Thrust.seaquance(fitnessIndeces);
        Profiler.Stop("sqauance");

        Profiler.Start("sorting fitness");
        Thrust.sort_by_key(deviceFitnes, fitnessIndeces);
        Profiler.Stop("sorting fitness");

        Profiler.Start("performing genetics");
       // var c = new FlattArray<byte>((byte[])populationGens, genLength).To2d();

        performGeneticAlgorythm.Run(
            populationGens.DevicePointer,
            populationGens2.DevicePointer,
            deviceFitnes.DevicePointer,
            fitnessIndeces.DevicePointer
            );
        //var a = new FlattArray<byte>((byte[])populationGens, genLength).To2d() ;

        Profiler.Stop("performing genetics");

    }


    public void LoadKernels()
    {

            performGeneticAlgorythm = context.LoadKernel("kernels/evolutionary2.ptx", "genetic");
            performGeneticAlgorythm.GridDimensions = 1;
            performGeneticAlgorythm.BlockDimensions = popSize;
            performGeneticAlgorythm.DynamicSharedMemory =
                (uint)(sizeof(float) * popSize);

    }


  

   public byte[] genGen(int index)
    {
        byte[] hostpopulation = populationGens;
        byte[] res = new byte[genLength];

        Array.Copy(
            hostpopulation,
            index * genLength,
            res,
            0,
            genLength
            );

        return res;

    }



    [DllImport("kernels/CudaFunctions.dll", EntryPoint = "FindFitest")]
    static extern subject __findFitest(
        ManagedCuda.BasicTypes.SizeT ptr1,
        ManagedCuda.BasicTypes.SizeT ptr2,
        int size
        );

    public subject FindFitest()
    {
        float[] hostFitness = deviceFitnes;
        int[] indeces = fitnessIndeces;

        var a = new FlattArray<byte>((byte[])populationGens, genLength).To2d();

        return __findFitest(
            deviceFitnes.DevicePointer.Pointer,
            fitnessIndeces.DevicePointer.Pointer,
            fitnessIndeces.Size
            );
    }

}



