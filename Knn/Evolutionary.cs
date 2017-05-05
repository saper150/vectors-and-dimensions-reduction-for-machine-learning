using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


class Evolutionary : IDisposable
{
    CudaContext context;

    CudaDataSet teaching;
    CudaDataSet test;

    CudaDeviceVariable<float> deviceTeachingVectors;
    CudaDeviceVariable<int> deviceTeachingClasses;

    CudaDeviceVariable<float> deviceTestVectors;
    CudaDeviceVariable<int> deviceTestClasses;

    CudaDeviceVariable<int> deviceClasses;

    CudaDeviceVariable<byte> populationGens;
    CudaDeviceVariable<byte> populationGens2;


    CudaDeviceVariable<int> deviceAccuracy;
    CudaDeviceVariable<int> deviceVectorSizes;
    CudaDeviceVariable<float> deviceFitnes;

    int popSize;

    int k = 3;
    int countToPass = 2;
    float alpha = 0.7f;
    public void AllocateMemory()
    {

        deviceTeachingVectors = teaching.Vectors.Raw;
        deviceTestVectors = test.Vectors.Raw;
        deviceTeachingClasses = teaching.Classes;
        deviceTestClasses = test.Classes;

        deviceClasses = 
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

        var kernel = context.LoadKernel("kernels/evolutionary.ptx", "createInitialPopulation");
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = popSize;
        using (CudaDeviceVariable<byte> deviceParrent = parrent) {

            kernel.Run(
                deviceParrent.DevicePointer,
                parrent.Length,
                populationGens.DevicePointer,
                popSize
                );

            //var hostGens = new FlattArray<byte>((byte[])populationGens,parrent.Length).To2d();


            Console.WriteLine();
        }

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
        AllocateMemory();

        CalculateClasses();
        CreateInitialPopulation(parrent);

        CalculateFitnes();

    }

    private void CalculateClasses()
    {

        const int threadsPerBlock = 256;
        var kernel = context.LoadKernel("kernels/evolutionary.ptx", "calculateDistances");

        kernel.GridDimensions = test.Vectors.GetLength(0) / threadsPerBlock + 1;
        kernel.BlockDimensions = threadsPerBlock;


        using (CudaDeviceVariable<float> deviceDistancesMemory =
            new CudaDeviceVariable<float>(teaching.Classes.Length * test.Classes.Length))
        {

            kernel.Run(
                deviceTeachingVectors.DevicePointer,
                teaching.Vectors.GetLength(0),
                deviceTestVectors.DevicePointer,
                test.Vectors.GetLength(0),
                deviceTeachingClasses.DevicePointer,
                teaching.Vectors.GetLength(1),
                deviceDistancesMemory.DevicePointer,
                deviceClasses.DevicePointer
                );

            FlattArray<float> distanceResult =
                new FlattArray<float>(deviceDistancesMemory, teaching.Classes.Length);

        }

    }



    public void CalculateAccuracy()
    {
        const int threadsPerBlock = 256;
        var kernel = context.LoadKernel("kernels/evolutionary.ptx", "findNewNeabours");

        dim3 gridDimension = new dim3() {
            x = (uint)(test.Vectors.GetLength(0)/threadsPerBlock + 1),
            y = (uint)popSize,
            z = 1
        };

        kernel.GridDimensions = gridDimension;
        kernel.BlockDimensions = threadsPerBlock;

        kernel.Run(
            deviceTestClasses.DevicePointer,
            test.Vectors.GetLength(0),
            deviceClasses.DevicePointer,
            teaching.Vectors.GetLength(0),
            populationGens.DevicePointer,
            popSize,
            k,
            countToPass,
            deviceAccuracy.DevicePointer
            );

        int[] hostR = deviceAccuracy;

    }

    public void CalculateVectorLengths() {
        var kernel = 
            context.LoadKernel("kernels/evolutionary.ptx", "countVectors");

        kernel.BlockDimensions = popSize;
        kernel.Run(
            populationGens.DevicePointer,
            teaching.Vectors.GetLength(0),
            deviceVectorSizes.DevicePointer
            );

        int[] a = deviceVectorSizes;
  
    }

    private void CalculateFitnes() {
        CalculateVectorLengths();
        CalculateAccuracy();

        var kernel =
                context.LoadKernel("kernels/evolutionary.ptx", "calculateFitnes");

        kernel.BlockDimensions = popSize;
        kernel.Run(
            deviceAccuracy.DevicePointer,
            deviceVectorSizes.DevicePointer,
            alpha,
            deviceFitnes.DevicePointer
            );

        float[] a = deviceFitnes;

    }


    public void Dispose()
    {
        context.Dispose();
    }
}

