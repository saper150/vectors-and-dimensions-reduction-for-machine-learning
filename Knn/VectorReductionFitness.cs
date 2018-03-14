using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

class VectorReductionFitness : IFitnessFunction
{

    CountVectorKernel countVectorsKernel;

    int popSize;

    CudaDeviceVariable<int> vectorSizes;

    CudaContext context;

    CudaKernel fitnessKernel;



    float _alpha;
    public float Alpha
    {
        get { return _alpha; }
        set
        {
            _alpha = value;
            fitnessKernel.SetConstantVariable("alpha", _alpha);

        }
    }
    IVectorReductionAccuracy accuracyCalc;
    int teachingCount;

    public VectorReductionFitness(CudaContext context, IVectorReductionAccuracy accuracyCalc, int popSize,int teachingCount)
    {
        this.teachingCount = teachingCount;
        this.accuracyCalc = accuracyCalc;
        this.popSize = popSize;
        this.context = context;
        countVectorsKernel = new CountVectorKernel(context, popSize, teachingCount);
        vectorSizes = new CudaDeviceVariable<int>(popSize);


        fitnessKernel = context.LoadKernel("kernels/VectorReduction.ptx", "fitnessFunction");
        Alpha = 0.7f;
        fitnessKernel.BlockDimensions = popSize;
        fitnessKernel.GridDimensions = 1;


    }



    public void CalculateFitness(CudaDeviceVariable<byte> population, CudaDeviceVariable<float> fitness)
    {
        Profiler.Start("Calculate accuracy");
        var deviceAccuracy = accuracyCalc.CalculateAccuracy(population);
        Profiler.Stop("Calculate accuracy");
        float[] asdf = deviceAccuracy;

        Profiler.Start("Calculate vectorSizes");
        countVectorsKernel.Calculate(population, vectorSizes);
        Profiler.Stop("Calculate vectorSizes");

        int[] v = vectorSizes;
        Profiler.Start("Avrage VectorSizes");
        float avrageVectorSize = Thrust.Avrage(vectorSizes);
        Profiler.Stop("Avrage VectorSizes");

        Profiler.Start("Avrage accuracy");
        float avrageAccuracy = Thrust.Avrage(deviceAccuracy);
        Profiler.Stop("Avrage accuracy");


        Profiler.Start("fittness kernel");
        fitnessKernel.Run(
            deviceAccuracy.DevicePointer,
            avrageAccuracy,
            vectorSizes.DevicePointer,
            avrageVectorSize,
            fitness.DevicePointer
            );
        Profiler.Stop("fittness kernel");

    }

    public int GenLength(int index)
    {
        int[] hostlen = vectorSizes;
        return hostlen[index];

    }
}








//class VectorReductionFitness : IFitnessFunction
//{

//    CountVectorKernel countVectorsKernel;
//    DeviceDataSet<int> teaching;
//    DeviceDataSet<int> test;
//    int popSize;

//    CudaDeviceVariable<int> calculatedNeabours;
//    CudaDeviceVariable<float> deviceAccuracy;
//    CudaDeviceVariable<int> vectorSizes;

//    CudaContext context;

//    CudaKernel accuracyKernel;
//    CudaKernel fitnessKernel;


//    int _threadsPerBlock = 256;
//    public int ThreadsPerBlock
//    {
//        get { return _threadsPerBlock; }
//        set
//        {
//            _threadsPerBlock = value;
//            dim3 gridDimension = new dim3()
//            {
//                x = (uint)(test.length / _threadsPerBlock + 1),
//                y = (uint)popSize,
//                z = 1
//            };
//            accuracyKernel.GridDimensions = gridDimension;
//            accuracyKernel.BlockDimensions = ThreadsPerBlock;

//        }
//    }



//    float _alpha;
//    public float Alpha
//    {
//        get { return _alpha; }
//        set
//        {
//            _alpha = value;
//            fitnessKernel.SetConstantVariable("alpha", _alpha);

//        }
//    }


//    int _k;
//    public int K
//    {
//        get { return _k; }
//        set
//        {
//            _k = value;
//            accuracyKernel.SetConstantVariable("k", _k);
//        }
//    }
//    int _countToPass;
//    public int CountToPass
//    {
//        get { return _countToPass; }
//        set
//        {
//            _countToPass = value;
//            accuracyKernel.SetConstantVariable("countToPass", _countToPass);
//        }
//    }



//    public VectorReductionFitness(CudaContext context, DeviceDataSet<int> teaching, DeviceDataSet<int> test, int popSize)
//    {
//        this.teaching = teaching;
//        this.test = test;
//        this.popSize = popSize;
//        this.context = context;
//        countVectorsKernel = new CountVectorKernel(context, popSize, teaching.length);
//        calculatedNeabours = new CudaDeviceVariable<int>(teaching.length * test.length);
//        deviceAccuracy = new CudaDeviceVariable<float>(popSize);
//        vectorSizes = new CudaDeviceVariable<int>(popSize);

//        Profiler.Start("calculate neabours");
//        CalculateNeabours();
//        Profiler.Stop("calculate neabours");


//        accuracyKernel = context.LoadKernel("kernels/VectorReduction.ptx", "calculateAccuracy");
//        dim3 gridDimension = new dim3()
//        {
//            x = (uint)(test.length / ThreadsPerBlock + 1),
//            y = (uint)popSize,
//            z = 1
//        };
//        accuracyKernel.GridDimensions = gridDimension;
//        accuracyKernel.BlockDimensions = ThreadsPerBlock;

//        accuracyKernel.SetConstantVariable("testVectorsCount", test.length);
//        accuracyKernel.SetConstantVariable("teachingVectorsCount", teaching.length);
//        accuracyKernel.SetConstantVariable("attributeCount", teaching.attributeCount);
//        accuracyKernel.SetConstantVariable("genLength", teaching.length);

//        K = 3;
//        CountToPass = 2;



//        fitnessKernel = context.LoadKernel("kernels/VectorReduction.ptx", "fitnessFunction");
//        Alpha = 0.7f;
//        fitnessKernel.BlockDimensions = ThreadsPerBlock;
//        fitnessKernel.GridDimensions = 1;


//    }


//    private void CalculateNeabours()
//    {
//        var kernel = context.LoadKernel("kernels/VectorReduction.ptx", "calculateNearestNeabours");
//        kernel.GridDimensions = test.length / ThreadsPerBlock + 1;
//        kernel.BlockDimensions = ThreadsPerBlock;

//        kernel.SetConstantVariable("testVectorsCount", test.length);
//        kernel.SetConstantVariable("teachingVectorsCount", teaching.length);
//        kernel.SetConstantVariable("attributeCount", teaching.attributeCount);

//        using (var deviceDistanceMemory =
//            new CudaDeviceVariable<float>(teaching.length * test.length))
//        {

//            kernel.Run(
//                teaching.vectors.DevicePointer,
//                test.vectors.DevicePointer,
//                deviceDistanceMemory.DevicePointer,
//                calculatedNeabours.DevicePointer
//                );
//            Profiler.Start("sort by key multiple");
//            Thrust.sort_by_key_multiple(deviceDistanceMemory, calculatedNeabours, teaching.length, test.length);
//            Profiler.Stop("sort by key multiple");

//        }
//    }



//    public void CalculateFitness(CudaDeviceVariable<byte> population, CudaDeviceVariable<float> fitness)
//    {
//        Profiler.Start("clear accuracy memory");
//        context.ClearMemory(deviceAccuracy.DevicePointer, 0, deviceAccuracy.SizeInBytes);
//        Profiler.Stop("clear accuracy memory");

//        Profiler.Start("Calculate Accuracy");
//        accuracyKernel.Run(
//            test.classes.DevicePointer,
//            teaching.classes.DevicePointer,
//            population.DevicePointer,
//            calculatedNeabours.DevicePointer,
//            deviceAccuracy.DevicePointer
//            );
//        Profiler.Stop("Calculate Accuracy");

//        Profiler.Start("Calculate vectorSizes");
//        countVectorsKernel.Calculate(population, vectorSizes);
//        Profiler.Stop("Calculate vectorSizes");

//        Profiler.Start("Avrage VectorSizes");
//        float avrageVectorSize = Thrust.Avrage(vectorSizes);
//        Profiler.Stop("Avrage VectorSizes");

//        Profiler.Start("Avrage accuracy");
//        float avrageAccuracy = Thrust.Avrage(deviceAccuracy);
//        Profiler.Stop("Avrage accuracy");


//        Profiler.Start("fittness kernel");
//        fitnessKernel.Run(
//            deviceAccuracy.DevicePointer,
//            avrageAccuracy,
//            vectorSizes.DevicePointer,
//            avrageVectorSize,
//            fitness.DevicePointer
//            );
//        Profiler.Stop("fittness kernel");


//    }

//    public string FitnessDetails(int index)
//    {
//        float[] hostAccuracy = deviceAccuracy;
//        int[] hostSizes = vectorSizes;

//        return $"accuracy: {hostAccuracy[index] / (float)test.length} length: {hostSizes[index]}";
//    }

//    public int GenLength(int index)
//    {
//        throw new NotImplementedException();
//    }
//}
