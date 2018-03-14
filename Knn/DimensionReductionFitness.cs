using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;




struct FitnessDetails {
    public float accuracy;
    public int size;

}

interface IFitnessFunction
{
    void CalculateFitness(CudaDeviceVariable<byte> population, CudaDeviceVariable<float> fitness);
    int GenLength(int index);
}


struct Node {
    public int mutex;
    public IntPtr one;
    public IntPtr zero; 
}

//class DimensionReductionFitness : IFitnessFunction
//{

//    DeviceDataSet<int> teaching;
//    DeviceDataSet<int> test;
//    CudaContext context;

//    CudaKernel accuracyKernel;
//    CudaKernel fitnessKernel;

//    CudaKernel sizeAndIndecesKernel;


//    CudaKernel saveCasheKernel;
//    CudaKernel readCasheKernel;

//    CudaDeviceVariable<int> deviceVectorSizes;
//    CudaDeviceVariable<int> populationIndeces;
//    CudaDeviceVariable<byte> isInCashe;

//    CudaDeviceVariable<float> deviceAccuracy;

//    CudaDeviceVariable<Node> casheTreeRoot;


//    CudaDeviceVariable<HeapData> heapMemory;

//    int _k;
//    public int K
//    {
//        get
//        {
//            return _k;
//        }
//        set
//        {
//            _k = value;
//            accuracyKernel.SetConstantVariable("k", _k);
//            heapMemory?.Dispose();
//            heapMemory = new CudaDeviceVariable<HeapData>(_k * test.length * popSize);
//        }
//    }

//    int _countToPass;
//    public int CountToPass
//    {
//        get
//        {
//            return _countToPass;
//        }
//        set
//        {
//            _countToPass = value;
//            accuracyKernel.SetConstantVariable("countToPass", _countToPass);
//        }
//    }


//    int _threadsPerBlock = 256;
//    public int ThreadsPerBlock
//    {
//        get { return _threadsPerBlock; }
//        set
//        {
//            _threadsPerBlock = value;
//            accuracyKernel.BlockDimensions = _threadsPerBlock;
//        }
//    }

//    float _alpha;
//    public float Alpha { get { return _alpha; } set {
//            _alpha = value;
//            fitnessKernel.SetConstantVariable("alpha", _alpha);

//        } }
//    int popSize;
//    public DimensionReductionFitness(
//        CudaContext context,
//        DeviceDataSet<int> teaching,
//        DeviceDataSet<int> test,
//        int popSize
//        )
//    {

//        this.popSize = popSize;

//        this.teaching = teaching;
//        this.test = test;
//        this.context = context;

//        accuracyKernel = context.LoadKernel
//            (
//            "kernels/dimensionsReductions.ptx",
//            "geneticKnn"
//            );

//        accuracyKernel.GridDimensions = new dim3()
//        {
//            x = (uint)(test.vectors.Size / ThreadsPerBlock) + 1,
//            y = (uint)popSize,
//            z = 1
//        };
//        accuracyKernel.BlockDimensions = ThreadsPerBlock;
//        deviceAccuracy = new CudaDeviceVariable<float>(popSize);

//        K = 3;
//        CountToPass = 2;
//        accuracyKernel.SetConstantVariable("atributeCount", test.attributeCount);
//        accuracyKernel.SetConstantVariable("teachingVectorsCount", teaching.length);
//        accuracyKernel.SetConstantVariable("testVectorsCount", test.length);
//        accuracyKernel.SetConstantVariable("popSize", popSize);
//        accuracyKernel.DynamicSharedMemory = (uint)(test.attributeCount * sizeof(float));


//        deviceVectorSizes = new CudaDeviceVariable<int>(popSize);


//        fitnessKernel = context.LoadKernel(
//            "kernels/dimensionsReductions.ptx",
//            "fitnessFunction"
//            );
//        fitnessKernel.GridDimensions = 1;
//        fitnessKernel.BlockDimensions = popSize;
//        Alpha = 0.7f;

//        //countVectorsKernel = new CountVectorKernel(context, popSize, teaching.attributeCount);


//        saveCasheKernel= context.LoadKernel(
//            "kernels/dimensionsReductions.ptx",
//            "saveToCashe"
//            );
//        saveCasheKernel.GridDimensions = (popSize*32)/ThreadsPerBlock+1;
//        saveCasheKernel.BlockDimensions = ThreadsPerBlock;
//        saveCasheKernel.SetConstantVariable("atributeCount", teaching.attributeCount);
//        saveCasheKernel.SetConstantVariable("popSize", teaching.attributeCount);


//        readCasheKernel = context.LoadKernel(
//            "kernels/dimensionsReductions.ptx",
//            "readCashe"
//            );
//        readCasheKernel.GridDimensions = 1;
//        readCasheKernel.BlockDimensions = popSize;
//        readCasheKernel.SetConstantVariable("atributeCount", teaching.attributeCount);


//        casheTreeRoot = new Node() {
//            mutex = 0,
//            one = (IntPtr)0,
//            zero = (IntPtr)0,
//        };


//        sizeAndIndecesKernel = context.LoadKernel("kernels/Common.ptx", "countVectorsIndeces");
//        sizeAndIndecesKernel.SetConstantVariable("genLength", teaching.attributeCount);
//        sizeAndIndecesKernel.GridDimensions = 1;
//        sizeAndIndecesKernel.BlockDimensions = popSize;
//        populationIndeces = new CudaDeviceVariable<int>(teaching.attributeCount * popSize);

//        isInCashe = new CudaDeviceVariable<byte>(popSize);

//    }


//    public float BaseAccuracy() {

//        var kernel = context.LoadKernel
//        (
//        "kernels/dimensionsReductions.ptx",
//        "geneticKnn"
//        );

//        kernel.GridDimensions = new dim3()
//        {
//            x = (uint)(test.vectors.Size / ThreadsPerBlock) + 1,
//            y = 1,
//            z = 1
//        };
//        kernel.BlockDimensions = ThreadsPerBlock;

//        kernel.SetConstantVariable("atributeCount", test.attributeCount);
//        kernel.SetConstantVariable("teachingVectorsCount", teaching.length);
//        kernel.SetConstantVariable("testVectorsCount", test.length);
//        kernel.SetConstantVariable("popSize", 1);
//        kernel.SetConstantVariable("k", K);
//        kernel.SetConstantVariable("countToPass", CountToPass);

//        kernel.DynamicSharedMemory = (uint)(test.attributeCount * sizeof(float));

//        var vectorSizes = new int[1];
//        vectorSizes[0] = test.attributeCount;

//        var indeces = Enumerable.Range(0, test.attributeCount).ToArray();
//        var acc = new float[] {0f };
//        var inCashe = new byte[] { 0 };

//        using (CudaDeviceVariable<int> deviceIndeces = indeces)
//        using (CudaDeviceVariable<int> deviceVectorSizesLocal = vectorSizes)
//        using (CudaDeviceVariable<float> accuracy = acc)
//        using (var heapMem = new CudaDeviceVariable<HeapData>(K))
//        using (CudaDeviceVariable<byte> deviceIsInCashe = inCashe)
//        {
//            kernel.Run(
//                test.vectors.DevicePointer,
//                test.classes.DevicePointer,
//                teaching.vectors.DevicePointer,
//                teaching.classes.DevicePointer,
//                deviceVectorSizesLocal.DevicePointer,
//                deviceIndeces.DevicePointer,
//                deviceIsInCashe.DevicePointer,
//                heapMem.DevicePointer,
//                accuracy.DevicePointer
//            );

//            float[] res = accuracy;
//            return res[0] / test.length;
//        }

//    }

//    public void CalculateFitness(CudaDeviceVariable<byte> population, CudaDeviceVariable<float> fitness)
//    {
//        Profiler.Start("vector sizes");
//        //countVectorsKernel.Calculate(population, deviceVectorSizes);
//        Profiler.Stop("vector sizes");

//        Profiler.Start("size and indeces");
//        sizeAndIndecesKernel.Run(
//            population.DevicePointer,
//            populationIndeces.DevicePointer,
//            deviceVectorSizes.DevicePointer
//            );

//        Profiler.Stop("size and indeces");

//        Profiler.Start("readCashe");
//        readCasheKernel.Run(
//            population.DevicePointer,
//            deviceAccuracy.DevicePointer,
//            isInCashe.DevicePointer,
//            casheTreeRoot.DevicePointer
//            );
//        Profiler.Stop("readCashe");

//        Profiler.Start("accuracy kernel");
//        accuracyKernel.Run(
//            test.vectors.DevicePointer,
//            test.classes.DevicePointer,
//            teaching.vectors.DevicePointer,
//            teaching.classes.DevicePointer,
//            deviceVectorSizes.DevicePointer,
//            populationIndeces.DevicePointer,
//            isInCashe.DevicePointer,
//            heapMemory.DevicePointer,
//            deviceAccuracy.DevicePointer
//            );
//        Profiler.Stop("accuracy kernel");

//        Profiler.Start("saveCashe");
//        saveCasheKernel.Run(
//            population.DevicePointer,
//            deviceAccuracy.DevicePointer,
//            casheTreeRoot.DevicePointer
//            );
//        Profiler.Stop("saveCashe");


//        Profiler.Start("Avrage Accuracy");
//        float avrageAccuracy = Thrust.Avrage(deviceAccuracy);
//        Profiler.Stop("Avrage Accuracy");

//        Profiler.Start("Avrage VectorSize");
//        float avrageVectorSize = Thrust.Avrage(deviceVectorSizes);
//        Profiler.Stop("Avrage VectorSize");

//        Profiler.Start("fitness kernel");
//        fitnessKernel.Run(
//            deviceAccuracy.DevicePointer,
//            avrageAccuracy,
//            deviceVectorSizes.DevicePointer,
//            avrageVectorSize,
//            fitness.DevicePointer
//            );
//        Profiler.Stop("fitness kernel");
//    }

//    public string FitnessDetails(int index)
//    {
//        float[] hostAccuracy = deviceAccuracy;
//        int[] hostLength = deviceVectorSizes;

//        return $"accuracy: {hostAccuracy[index]/test.length} length: {hostLength[index]/(float)test.attributeCount}";

//    }
//}



class DimensionReductionFitness : IFitnessFunction
{
    CudaContext context;

    CudaKernel fitnessKernel;

    CudaKernel sizeAndIndecesKernel;


    CudaDeviceVariable<int> deviceVectorSizes;
    CudaDeviceVariable<int> populationIndeces;

    IDimensionAccuracy accuracyFunc;

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
    int popSize;
    public DimensionReductionFitness(
        CudaContext context,
        IDimensionAccuracy accuracyFunc,
        int popSize,
        int genLength
        )
    {

        this.accuracyFunc = accuracyFunc;
        this.popSize = popSize;
        this.context = context;

        deviceVectorSizes = new CudaDeviceVariable<int>(popSize);


        fitnessKernel = context.LoadKernel(
            "kernels/dimensionsReductions.ptx",
            "fitnessFunction"
            );
        fitnessKernel.GridDimensions = 1;
        fitnessKernel.BlockDimensions = popSize;
        Alpha = 0.7f;

        sizeAndIndecesKernel = context.LoadKernel("kernels/Common.ptx", "countVectorsIndeces");
        sizeAndIndecesKernel.SetConstantVariable("genLength", genLength);
        sizeAndIndecesKernel.GridDimensions = 1;
        sizeAndIndecesKernel.BlockDimensions = popSize;
        populationIndeces = new CudaDeviceVariable<int>(genLength * popSize);

    }


    public void CalculateFitness(CudaDeviceVariable<byte> population, CudaDeviceVariable<float> fitness)
    {
        Profiler.Start("vector sizes");
        //countVectorsKernel.Calculate(population, deviceVectorSizes);
        Profiler.Stop("vector sizes");

        Profiler.Start("size and indeces");
        sizeAndIndecesKernel.Run(
            population.DevicePointer,
            populationIndeces.DevicePointer,
            deviceVectorSizes.DevicePointer
            );

        Profiler.Stop("size and indeces");

        var accuracy =  accuracyFunc.CalculateAccuracy(population, populationIndeces, deviceVectorSizes);


        Profiler.Start("Avrage Accuracy");
        float avrageAccuracy = Thrust.Avrage(accuracy);
        Profiler.Stop("Avrage Accuracy");

        Profiler.Start("Avrage VectorSize");
        float avrageVectorSize = Thrust.Avrage(deviceVectorSizes);
        Profiler.Stop("Avrage VectorSize");

        Profiler.Start("fitness kernel");
        fitnessKernel.Run(
            accuracy.DevicePointer,
            avrageAccuracy,
            deviceVectorSizes.DevicePointer,
            avrageVectorSize,
            fitness.DevicePointer
            );
        Profiler.Stop("fitness kernel");
    }


    public int GenLength(int index)
    {
        int[] len = deviceVectorSizes;
        return len[index];
    }

}






