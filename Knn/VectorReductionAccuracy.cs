using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.VectorTypes;


interface IVectorReductionAccuracy
{
    CudaDeviceVariable<float> CalculateAccuracy(CudaDeviceVariable<byte> population,int startIndex = 1);
    float GenAccuracy(int index);
}


static class Neabours {

    public static void CalculateNeabours<T>
        (CudaContext context,
        DeviceDataSet<T> teaching,
        DeviceDataSet<T> test,
        CudaDeviceVariable<int> calculatedNeabours,
        int threadsPerBlock
        ) where T : struct
    {
        var kernel = context.LoadKernel("kernels/VectorReduction.ptx", "calculateNearestNeabours");
        kernel.GridDimensions = test.length / threadsPerBlock + 1;
        kernel.BlockDimensions = threadsPerBlock;

        kernel.SetConstantVariable("testVectorsCount", test.length);
        kernel.SetConstantVariable("teachingVectorsCount", teaching.length);
        kernel.SetConstantVariable("attributeCount", teaching.attributeCount);

        using (var deviceDistanceMemory =
            new CudaDeviceVariable<float>(teaching.length * test.length))
        {

            kernel.Run(
                teaching.vectors.DevicePointer,
                test.vectors.DevicePointer,
                deviceDistanceMemory.DevicePointer,
                calculatedNeabours.DevicePointer
                );
            Thrust.sort_by_key_multiple(deviceDistanceMemory, calculatedNeabours, teaching.length, test.length);

        }
    }



}



class VectorReductionAccuracy : IVectorReductionAccuracy
{

    DeviceDataSet<int> teaching;
    DeviceDataSet<int> test;
    int popSize;

    CudaDeviceVariable<int> calculatedNeabours;
    CudaDeviceVariable<float> deviceAccuracy;

    CudaContext context;

    CudaKernel accuracyKernel;


    int _threadsPerBlock = 256;
    public int ThreadsPerBlock
    {
        get { return _threadsPerBlock; }
        set
        {
            _threadsPerBlock = value;
            dim3 gridDimension = new dim3()
            {
                x = (uint)(test.length / _threadsPerBlock + 1),
                y = (uint)popSize,
                z = 1
            };
            accuracyKernel.GridDimensions = gridDimension;
            accuracyKernel.BlockDimensions = ThreadsPerBlock;

        }
    }

    int _k;
    public int K
    {
        get { return _k; }
        set
        {
            _k = value;
            accuracyKernel.SetConstantVariable("k", _k);
        }
    }
    int _countToPass;
    public int CountToPass
    {
        get { return _countToPass; }
        set
        {
            _countToPass = value;
            accuracyKernel.SetConstantVariable("countToPass", _countToPass);
        }
    }



    public VectorReductionAccuracy(CudaContext context, DeviceDataSet<int> teaching, DeviceDataSet<int> test, Options options )
    {
        this.teaching = teaching;
        this.test = test;
        this.popSize = options.PopSize;
        this.context = context;

        calculatedNeabours = new CudaDeviceVariable<int>(teaching.length * test.length);
        deviceAccuracy = new CudaDeviceVariable<float>(popSize);

        Profiler.Start("calculate neabours");
        Neabours.CalculateNeabours(context, teaching, test, calculatedNeabours, ThreadsPerBlock);
        Profiler.Stop("calculate neabours");


        accuracyKernel = context.LoadKernel("kernels/VectorReduction.ptx", "calculateAccuracy");
        dim3 gridDimension = new dim3()
        {
            x = (uint)(test.length / ThreadsPerBlock + 1),
            y = (uint)popSize,
            z = 1
        };
        accuracyKernel.GridDimensions = gridDimension;
        accuracyKernel.BlockDimensions = ThreadsPerBlock;

        accuracyKernel.SetConstantVariable("testVectorsCount", test.length);
        accuracyKernel.SetConstantVariable("teachingVectorsCount", teaching.length);
        accuracyKernel.SetConstantVariable("attributeCount", teaching.attributeCount);
        accuracyKernel.SetConstantVariable("genLength", teaching.length);

        K = options.K;

    }

    public float GenAccuracy(int index)
    {
        float[] host = deviceAccuracy;
        return host[index];

    }

    public CudaDeviceVariable<float> CalculateAccuracy(CudaDeviceVariable<byte> population, int startIndex = 1)
    {
        byte[] pop = population;
        Profiler.Start("clear accuracy memory");
        context.ClearMemory(deviceAccuracy.DevicePointer, 0, deviceAccuracy.SizeInBytes);
        Profiler.Stop("clear accuracy memory");
        Profiler.Start("accuracy Kernel");
        accuracyKernel.Run(
            test.classes.DevicePointer,
            teaching.classes.DevicePointer,
            population.DevicePointer,
            calculatedNeabours.DevicePointer,
            deviceAccuracy.DevicePointer,
            startIndex
            );
        Profiler.Stop("accuracy Kernel");
        return deviceAccuracy;
    }
}





class VectorReductionAccuracyRegresion : IVectorReductionAccuracy
{

    DeviceDataSet<float> teaching;
    DeviceDataSet<float> test;
    int popSize;

    CudaDeviceVariable<int> calculatedNeabours;
    CudaDeviceVariable<float> deviceAccuracy;

    CudaContext context;

    CudaKernel accuracyKernel;
    CudaKernel RMSEKernel;

    int _threadsPerBlock = 256;
    public int ThreadsPerBlock
    {
        get { return _threadsPerBlock; }
        set
        {
            _threadsPerBlock = value;
            dim3 gridDimension = new dim3()
            {
                x = (uint)(test.length / _threadsPerBlock + 1),
                y = (uint)popSize,
                z = 1
            };
            accuracyKernel.GridDimensions = gridDimension;
            accuracyKernel.BlockDimensions = ThreadsPerBlock;

        }
    }

    int _k;
    public int K
    {
        get { return _k; }
        set
        {
            _k = value;
            accuracyKernel.SetConstantVariable("k", _k);
        }
    }
    int _countToPass;
    public int CountToPass
    {
        get { return _countToPass; }
        set
        {
            _countToPass = value;
            accuracyKernel.SetConstantVariable("countToPass", _countToPass);
        }
    }

    

    public VectorReductionAccuracyRegresion(CudaContext context, DeviceDataSet<float> teaching, DeviceDataSet<float> test, int popSize)
    {
        this.teaching = teaching;
        this.test = test;
        this.popSize = popSize;
        this.context = context;

        calculatedNeabours = new CudaDeviceVariable<int>(teaching.length * test.length);
        deviceAccuracy = new CudaDeviceVariable<float>(popSize);

        Profiler.Start("calculate neabours");
        Neabours.CalculateNeabours(context, teaching, test, calculatedNeabours, ThreadsPerBlock);
        Profiler.Stop("calculate neabours");


        accuracyKernel = context.LoadKernel("kernels/VectorReduction.ptx", "calculateAccuracy");
        dim3 gridDimension = new dim3()
        {
            x = (uint)(test.length / ThreadsPerBlock + 1),
            y = (uint)popSize,
            z = 1
        };
        accuracyKernel.GridDimensions = gridDimension;
        accuracyKernel.BlockDimensions = ThreadsPerBlock;

        accuracyKernel.SetConstantVariable("testVectorsCount", test.length);
        accuracyKernel.SetConstantVariable("teachingVectorsCount", teaching.length);
        accuracyKernel.SetConstantVariable("attributeCount", teaching.attributeCount);
        accuracyKernel.SetConstantVariable("genLength", teaching.length);

        K = 3;
        CountToPass = 2;

        RMSEKernel = context.LoadKernel("kernels/VectorReduction.ptx", "RMSE");
        RMSEKernel.GridDimensions = 1;
        RMSEKernel.BlockDimensions = popSize;
        RMSEKernel.SetConstantVariable("testVectorsCount", test.length);

    }


    public float GenAccuracy(int index)
    {
        float[] host = deviceAccuracy;
        return host[index];

    }

    public float BaseAccuracy()
    {
        var baseKernel = context.LoadKernel("kernels/VectorReduction.ptx", "calculateAccuracy");
        dim3 gridDimension = new dim3()
        {
            x = (uint)(test.length / ThreadsPerBlock + 1),
            y = (uint)1,
            z = 1
        };
        baseKernel.GridDimensions = gridDimension;
        baseKernel.BlockDimensions = ThreadsPerBlock;

        baseKernel.SetConstantVariable("testVectorsCount", test.length);
        baseKernel.SetConstantVariable("teachingVectorsCount", teaching.length);
        baseKernel.SetConstantVariable("attributeCount", teaching.attributeCount);
        baseKernel.SetConstantVariable("genLength", teaching.length);

        var BaseRMSEKernel = context.LoadKernel("kernels/VectorReduction.ptx", "RMSE");
        BaseRMSEKernel.GridDimensions = 1;
        BaseRMSEKernel.BlockDimensions = 1;
        BaseRMSEKernel.SetConstantVariable("testVectorsCount", test.length);

        byte[] gen = new byte[teaching.length];
        for (int i = 0; i < gen.Length; i++)
        {
            gen[i] = 1;
        }

        using (CudaDeviceVariable<byte> deviceGen = gen)
        using (CudaDeviceVariable<float> baseAccuracy = new CudaDeviceVariable<float>(1))
        {
            accuracyKernel.Run(
                test.classes.DevicePointer,
                teaching.classes.DevicePointer,
                deviceGen.DevicePointer,
                calculatedNeabours.DevicePointer,
                deviceAccuracy.DevicePointer
                );

            BaseRMSEKernel.Run(baseAccuracy.DevicePointer);

            float[] host = baseAccuracy;
            return host[0];

        }



    }



    public CudaDeviceVariable<float> CalculateAccuracy(CudaDeviceVariable<byte> population,int startIndex = 1)
    {
        Profiler.Start("clear accuracy memory");
        context.ClearMemory(deviceAccuracy.DevicePointer, 0, deviceAccuracy.SizeInBytes);
        Profiler.Stop("clear accuracy memory");

        Profiler.Start("accuracy kernel");
        accuracyKernel.Run(
            test.classes.DevicePointer,
            teaching.classes.DevicePointer,
            population.DevicePointer,
            calculatedNeabours.DevicePointer,
            deviceAccuracy.DevicePointer
            );
        Profiler.Stop("accuracy kernel");

        Profiler.Start("RSEM");
        RMSEKernel.Run(deviceAccuracy.DevicePointer);
        Profiler.Stop("RSEM");

        return deviceAccuracy;
    }
}