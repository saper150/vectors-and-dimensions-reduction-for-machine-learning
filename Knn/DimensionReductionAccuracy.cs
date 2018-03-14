using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;





interface IDimensionAccuracy {
    CudaDeviceVariable<float> CalculateAccuracy(CudaDeviceVariable<byte> population,
        CudaDeviceVariable<int> indeces,
        CudaDeviceVariable<int> sizes
        );

    float GenAccuracy(int index);

}

class DimensionReductionAccuracy : IDimensionAccuracy
{

    DeviceDataSet<int> teaching;
    DeviceDataSet<int> test;
    CudaContext context;

    CudaKernel accuracyKernel;

    CudaKernel saveCasheKernel;
    CudaKernel readCasheKernel;

    CudaDeviceVariable<byte> isInCashe;

    CudaDeviceVariable<Node> casheTreeRoot;

    CudaDeviceVariable<HeapData> heapMemory;

    CudaDeviceVariable<float> accuracy;

    int _k;
    public int K
    {
        get
        {
            return _k;
        }
        set
        {
            _k = value;
            accuracyKernel.SetConstantVariable("k", _k);
            heapMemory?.Dispose();
            heapMemory = new CudaDeviceVariable<HeapData>(_k * test.length * popSize);
        }
    }

    int _countToPass;
    public int CountToPass
    {
        get
        {
            return _countToPass;
        }
        set
        {
            _countToPass = value;
            accuracyKernel.SetConstantVariable("countToPass", _countToPass);
        }
    }


    int _threadsPerBlock = 256;
    public int ThreadsPerBlock
    {
        get { return _threadsPerBlock; }
        set
        {
            _threadsPerBlock = value;
            accuracyKernel.BlockDimensions = _threadsPerBlock;
        }
    }


    int popSize;
    public DimensionReductionAccuracy(
        CudaContext context,
        DeviceDataSet<int> teaching,
        DeviceDataSet<int> test,
        int popSize
        )
    {

        this.popSize = popSize;
        this.teaching = teaching;
        this.test = test;
        this.context = context;

        accuracyKernel = context.LoadKernel
            (
            "kernels/dimensionsReductions.ptx",
            "geneticKnn"
            );

        accuracyKernel.GridDimensions = new dim3()
        {
            x = (uint)(test.vectors.Size / ThreadsPerBlock) + 1,
            y = (uint)popSize,
            z = 1
        };
        accuracyKernel.BlockDimensions = ThreadsPerBlock;

        K = 3;
        CountToPass = 2;
        accuracyKernel.SetConstantVariable("atributeCount", test.attributeCount);
        accuracyKernel.SetConstantVariable("teachingVectorsCount", teaching.length);
        accuracyKernel.SetConstantVariable("testVectorsCount", test.length);
        accuracyKernel.SetConstantVariable("popSize", popSize);
        accuracyKernel.DynamicSharedMemory = (uint)(test.attributeCount * sizeof(float));


        saveCasheKernel = context.LoadKernel(
            "kernels/dimensionsReductions.ptx",
            "saveToCashe"
            );
        saveCasheKernel.GridDimensions = (popSize * 32) / ThreadsPerBlock + 1;
        saveCasheKernel.BlockDimensions = ThreadsPerBlock;
        saveCasheKernel.SetConstantVariable("atributeCount", teaching.attributeCount);
        saveCasheKernel.SetConstantVariable("popSize", teaching.attributeCount);


        readCasheKernel = context.LoadKernel(
            "kernels/dimensionsReductions.ptx",
            "readCashe"
            );
        readCasheKernel.GridDimensions = 1;
        readCasheKernel.BlockDimensions = popSize;
        readCasheKernel.SetConstantVariable("atributeCount", teaching.attributeCount);


        casheTreeRoot = new Node()
        {
            mutex = 0,
            one = (IntPtr)0,
            zero = (IntPtr)0,
        };

        isInCashe = new CudaDeviceVariable<byte>(popSize);
        accuracy = new CudaDeviceVariable<float>(popSize);
    }


    public float BaseAccuracy()
    {

        var kernel = context.LoadKernel
        (
        "kernels/dimensionsReductions.ptx",
        "geneticKnn"
        );

        kernel.GridDimensions = new dim3()
        {
            x = (uint)(test.vectors.Size / ThreadsPerBlock) + 1,
            y = 1,
            z = 1
        };
        kernel.BlockDimensions = ThreadsPerBlock;

        kernel.SetConstantVariable("atributeCount", test.attributeCount);
        kernel.SetConstantVariable("teachingVectorsCount", teaching.length);
        kernel.SetConstantVariable("testVectorsCount", test.length);
        kernel.SetConstantVariable("popSize", 1);
        kernel.SetConstantVariable("k", K);
        kernel.SetConstantVariable("countToPass", CountToPass);

        kernel.DynamicSharedMemory = (uint)(test.attributeCount * sizeof(float));

        var vectorSizes = new int[1];
        vectorSizes[0] = test.attributeCount;

        var indeces = Enumerable.Range(0, test.attributeCount).ToArray();
        var acc = new float[] { 0f };
        var inCashe = new byte[] { 0 };

        using (CudaDeviceVariable<int> deviceIndeces = indeces)
        using (CudaDeviceVariable<int> deviceVectorSizesLocal = vectorSizes)
        using (CudaDeviceVariable<float> accuracy = acc)
        using (var heapMem = new CudaDeviceVariable<HeapData>(K))
        using (CudaDeviceVariable<byte> deviceIsInCashe = inCashe)
        {
            kernel.Run(
                test.vectors.DevicePointer,
                test.classes.DevicePointer,
                teaching.vectors.DevicePointer,
                teaching.classes.DevicePointer,
                deviceVectorSizesLocal.DevicePointer,
                deviceIndeces.DevicePointer,
                deviceIsInCashe.DevicePointer,
                heapMem.DevicePointer,
                accuracy.DevicePointer
            );

            float[] res = accuracy;
            return res[0] / test.length;
        }

    }
    public CudaDeviceVariable<float> CalculateAccuracy(
        CudaDeviceVariable<byte> population,
        CudaDeviceVariable<int> indeces,
        CudaDeviceVariable<int> sizes
        )
    {
        Profiler.Start("readCashe");
        readCasheKernel.Run(
            population.DevicePointer,
            accuracy.DevicePointer,
            isInCashe.DevicePointer,
            casheTreeRoot.DevicePointer
            );
        Profiler.Stop("readCashe");

        Profiler.Start("accuracy kernel");
        accuracyKernel.Run(
            test.vectors.DevicePointer,
            test.classes.DevicePointer,
            teaching.vectors.DevicePointer,
            teaching.classes.DevicePointer,
            sizes.DevicePointer,
            indeces.DevicePointer,
            isInCashe.DevicePointer,
            heapMemory.DevicePointer,
            accuracy.DevicePointer
            );
        Profiler.Stop("accuracy kernel");

        Profiler.Start("saveCashe");
        saveCasheKernel.Run(
            population.DevicePointer,
            accuracy.DevicePointer,
            casheTreeRoot.DevicePointer
            );
        Profiler.Stop("saveCashe");

        return accuracy;

    }

    public float GenAccuracy(int index)
    {
        float[] acc = accuracy;
        return acc[index];
    }
}
