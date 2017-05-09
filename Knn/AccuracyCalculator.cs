using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


interface IFitnessFunction
{
    void CalculateFitness(CudaDeviceVariable<byte> population, CudaDeviceVariable<float> fitness);

}




class DimensionReductionFitness : IFitnessFunction
{

    DeviceDataSet teaching;
    DeviceDataSet test;
    CudaContext context;

    CudaKernel accuracyKernel;
    CudaKernel countVectorsKernel;
    CudaKernel fitnessKernel;

    CudaDeviceVariable<int> deviceVectorSizes;
    CudaDeviceVariable<float> deviceAccuracy;

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

    float _alpha;
    public float Alpha { get { return _alpha; } set {
            _alpha = value;
            fitnessKernel.SetConstantVariable("alpha", _alpha);

        } }

    public DimensionReductionFitness(
        CudaContext context,
        DeviceDataSet teaching,
        DeviceDataSet test,
        int popSize
        )
    {

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
        deviceAccuracy = new CudaDeviceVariable<float>(popSize);

        K = 3;
        CountToPass = 2;
        accuracyKernel.SetConstantVariable("atributeCount", test.attributeCount);
        accuracyKernel.SetConstantVariable("teachingVectorsCount", teaching.length);
        accuracyKernel.SetConstantVariable("testVectorsCount", test.length);
        accuracyKernel.SetConstantVariable("popSize", popSize);



        countVectorsKernel = context.LoadKernel
            (
            "kernels/Common.ptx",
            "countVectors"
            );

        countVectorsKernel.SetConstantVariable("genLength", teaching.attributeCount);
        countVectorsKernel.BlockDimensions = popSize;
        countVectorsKernel.GridDimensions = 1;

        deviceVectorSizes = new CudaDeviceVariable<int>(popSize);


        fitnessKernel = context.LoadKernel(
            "kernels/dimensionsReductions.ptx",
            "fitnessFunction"
            );
        fitnessKernel.GridDimensions = 1;
        fitnessKernel.BlockDimensions = popSize;
        fitnessKernel.SetConstantVariable("alpha", _alpha);



    }


    public void CalculateFitness(CudaDeviceVariable<byte> population, CudaDeviceVariable<float> fitness)
    {

        countVectorsKernel.Run(
            population.DevicePointer,
            deviceVectorSizes.DevicePointer
        );
        int[] t = deviceVectorSizes;

        context.ClearMemory(deviceAccuracy.DevicePointer, 0, deviceAccuracy.SizeInBytes);
        accuracyKernel.Run(
            test.vectors.DevicePointer,
            test.classes.DevicePointer,
            teaching.vectors.DevicePointer,
            teaching.classes.DevicePointer,
            population.DevicePointer,
            deviceAccuracy.DevicePointer
            );
        float[] a = deviceAccuracy;


        float avrageAccuracy = Thrust.Avrage(deviceAccuracy);
        float avrageVectorSize = Thrust.Avrage(deviceVectorSizes);

        fitnessKernel.Run(
            deviceAccuracy.DevicePointer,
            avrageAccuracy,
            deviceVectorSizes.DevicePointer,
            avrageVectorSize,
            fitness.DevicePointer
            );
        float[] f = fitness;

    }
}
