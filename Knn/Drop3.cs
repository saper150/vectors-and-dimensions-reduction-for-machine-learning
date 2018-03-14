using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;




class Comparer : IComparer<Data>
{
    public int Compare(Data x, Data y)
    {
        return x.val < y.val ? 1 : 0;
    }
}



class NeighborFinder : IDisposable
{


    CudaContext context;
    CudaKernel kernel;
    CudaDeviceVariable<float> deviceVectors;
    CudaDeviceVariable<float> deviceResult;
    CudaDeviceVariable<byte> deviceIsInDataSet;


    Data[] heap;
    int vectorCount;
    int attrCount;
    float[] results;
    public NeighborFinder(CudaContext context, FlattArray<float> vectors,int countToFind)
    {

        heap = new Data[countToFind];
        this.vectorCount = vectors.GetLength(0);
        this.attrCount= vectors.GetLength(1);
        this.context = context;

        kernel = context.LoadKernel("kernels/drop3.ptx", "calculateDistances");
        kernel.GridDimensions = vectors.GetLength(0) / 256 + 1;
        kernel.BlockDimensions = 256;
        results = new float[vectors.GetLength(0)];

        deviceVectors = vectors.Raw;
        deviceResult = new CudaDeviceVariable<float>(vectors.GetLength(0));
        deviceIsInDataSet = new CudaDeviceVariable<byte>(vectors.GetLength(0));

    }

    public int[] Find(byte[] vectorsInDataset,int vectorToExamine,out int vectorsFound) {

        context.CopyToDevice(
            deviceIsInDataSet.DevicePointer, vectorsInDataset);

        kernel.Run(
            deviceVectors.DevicePointer,
            vectorCount,
            vectorToExamine,
            attrCount,
            deviceIsInDataSet.DevicePointer,
            deviceResult.DevicePointer
            );

        float[] hostResult = deviceResult;

        for (int i = 0; i < heap.Length; i++)
        {
            heap[i].val = float.MaxValue;
        }
        vectorsFound = 0;
        for (int i = 0; i < hostResult.Length; i++)
        {
            if (vectorsInDataset[i]==1 && hostResult[i] < heap[0].val && i != vectorToExamine)
            {
                vectorsFound++;
                heap[0].val = hostResult[i];
                heap[0].index = i;
                Utils.hipify(heap);
            }
        }
        if (vectorCount>heap.Length)
        {
            vectorsFound = heap.Length;
        }
        Array.Sort(heap, new Comparer());

        int[] result = new int[heap.Length];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = heap[i].index;
        }
        return result;

    }

    public void Dispose()
    {
        deviceVectors.Dispose();
        deviceResult.Dispose();
        deviceIsInDataSet.Dispose();
    }
}






public class Drop3
{
    public int K { get; set; }
    public int ThreadsPerBlock { get; set; }
    public int CasheSize { get; set; }
    public int EnnCountToPass { get; set; }


    public Drop3()
    {
        K = 3;
        ThreadsPerBlock = 256;
        CasheSize = 1230;
        EnnCountToPass = 2;
    }

    public int[] Apply(CudaDataSet<int> data,CudaContext context)
    {
        Profiler.Start("Enn");
        var ennFiltered = Enn(data,context);
        Profiler.Stop("Enn");

        Profiler.Start("drop3 rest");
        var rest = ApplyRest(context, ennFiltered);
        Profiler.Stop("drop3 rest");

        return rest;
    }

    public static void SortDataDesc(HostDataset host,int[][] nearestNeabours,float[] sortBy)
    {
        Sorting.SortDesc(host, sortBy);
        Sorting.sortAndRemapDesc(nearestNeabours, sortBy);
    }

    private int[] ApplyRest(CudaContext context,CudaDataSet<int> data)
    {


        int vectorsCount = data.Vectors.GetLength(0);
        int attributeCount = data.Vectors.GetLength(1);

        var kernel = context.LoadKernel("kernels/drop3.ptx", "findNeighbours");
        kernel.GridDimensions = data.Vectors.GetLength(0) / ThreadsPerBlock + 1;
        kernel.BlockDimensions = ThreadsPerBlock;

        using (CudaDeviceVariable<int> d_classes = data.Classes)
        using (CudaDeviceVariable<float> vectors = data.Vectors.Raw)
        using (var heapMemory = new CudaDeviceVariable<HeapData>(data.Vectors.GetLength(0) * CasheSize))
        using (var nearestEnemyDistances = new CudaDeviceVariable<float>(data.Vectors.GetLength(0)))
        {

            kernel.Run(
                vectors.DevicePointer,
                data.Vectors.GetLength(0),
                data.Vectors.GetLength(1),
                CasheSize,
                d_classes.DevicePointer,
                heapMemory.DevicePointer,
                nearestEnemyDistances.DevicePointer
                );



            float[] hostNearestEnemy = nearestEnemyDistances;
            float[][] hostVectors = data.Vectors.To2d();

            var Neighbors = new FlattArray<HeapData>(heapMemory, CasheSize);
            var nearestNeighbors = new int[vectorsCount][];
            for (int i = 0; i < vectorsCount; i++)
            {
                nearestNeighbors[i] = new int[CasheSize];

                for (int j = 0; j < CasheSize; j++)
                {
                    nearestNeighbors[i][j] = Neighbors[i, j].label;
                }
            }

            HostDataset host = data.ToHostDataSet();
            SortDataDesc(host, nearestNeighbors, hostNearestEnemy);


            return proccesData(context, host, nearestNeighbors);

        }

    }

    int[] proccesData(CudaContext context, HostDataset host,int[][] nearestNeighbors)
    {

        int len = host.Vectors.Length;
        int initialCasheSize = nearestNeighbors[0].Length-K;
        List<int>[] associates = new List<int>[len];
        for (int i = 0; i < len; i++)
        {
            associates[i] = new List<int>();
        }


        byte[] isInDaaset = new byte[len];
        int[] nextNearestNabour = new int[len];
        int[] nearestNaboursSizes = new int[len];

        for (int i = 0; i < isInDaaset.Length; i++)
        {
            isInDaaset[i] = 1;
            nextNearestNabour[i] = K + 1;
            nearestNaboursSizes[i] = initialCasheSize+K;
        }
        NeighborFinder finder = new NeighborFinder(context,new FlattArray<float>(host.Vectors), initialCasheSize/2);

        Func<int, int, int> isClasifiedCorectly =
            (int vectorToCheck, int removed) =>
            {
                int correctCount = 0;
                int myClass = host.Classes[vectorToCheck];
                for (int i = 0; i < K; i++)
                {
                    if (nearestNeighbors[vectorToCheck][i] != removed
                        && host.Classes[nearestNeighbors[vectorToCheck][i]] == myClass)
                    {
                        correctCount++;
                    }
                }

                if (correctCount >= K / 2)
                {
                    return 1;
                }
                else return 0;
            };


        Action<int, int> findNewNearestNeabout =
            (int vector, int removed) =>
            {
                int toRemove = -1;
                for (int i = 0; i < K; i++)
                {
                    if (removed == nearestNeighbors[vector][i])
                    {
                        toRemove = i;
                        break;
                    }
                }

                for (int i = nextNearestNabour[vector]; i < nearestNaboursSizes[vector]; i++)
                {
                    if (isInDaaset[nearestNeighbors[vector][i]]==1)
                    {
                        nearestNeighbors[vector][toRemove] = nearestNeighbors[vector][i];
                        nextNearestNabour[vector] = i + 1;
                        associates[nearestNeighbors[vector][i]].Add(vector);
                        return;

                    }
                }

                int vectorsFound;
                var neabours = finder.Find(isInDaaset, vector, out vectorsFound);
                nearestNeighbors[vector][toRemove] = neabours[0];
                associates[neabours[0]].Add(vector);
                nextNearestNabour[vector] = K + 1;
                nearestNaboursSizes[vector] = K + vectorsFound;
                Array.Copy(
                    neabours,
                    1,
                    nearestNeighbors[vector],
                    K + 1,
                    neabours.Length-1
                    );


            };



        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < K; j++)
            {
                associates[nearestNeighbors[i][j]].Add(i);
            }
        }


        for (int i = 0; i < len; i++)
        {
            int correctCount = 0;
            for (int j = 0; j < associates[i].Count; j++)
            {
                correctCount += isClasifiedCorectly(associates[i][j], i);
            }
            if (correctCount >= (associates[i].Count - correctCount))
            {
                isInDaaset[i] = 0;
                for (int j = 0; j < associates[i].Count; j++)
                {
                    findNewNearestNeabout(associates[i][j], i);

                }
            }
        }

        List<int> indecesInDataSet = new List<int>();

        for (int i = 0; i < isInDaaset.Length; i++)
        {
            if (isInDaaset[i] == 1) {
                indecesInDataSet.Add(host.OrginalIndeces[i]);
            }
        }


        finder.Dispose();

        return indecesInDataSet.ToArray();

    }


    CudaDataSet<int> Enn(CudaDataSet<int> data,CudaContext context)
    {

        var kernel = context.LoadKernel("kernels/kernel.ptx", "enn");
        kernel.GridDimensions = data.Vectors.GetLength(0) / ThreadsPerBlock + 1;
        kernel.BlockDimensions = ThreadsPerBlock;


        using (CudaDeviceVariable<float> vectors = data.Vectors.Raw)
        using (CudaDeviceVariable<int> classes = data.Classes)
        using (var heapMemory = new CudaDeviceVariable<HeapData>(data.Vectors.GetLength(0) * K))
        using (var result = new CudaDeviceVariable<byte>(data.Vectors.GetLength(0)))
        {

            kernel.Run(
                vectors.DevicePointer,
                data.Vectors.GetLength(0),
                data.Vectors.GetLength(1),
                classes.DevicePointer,
                K,
                EnnCountToPass,
                heapMemory.DevicePointer,
                result.DevicePointer
                );

            byte[] hostResult = result;
            List<int> indeces = new List<int>();
            for (int i = 0; i < hostResult.Length; i++)
            {
                if (hostResult[i] == 1) {
                    indeces.Add(i);
                }
            }

            return data.Filter(hostResult.createIndexesToStay());

        }




    }


}

