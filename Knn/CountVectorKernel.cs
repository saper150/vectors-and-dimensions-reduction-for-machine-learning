using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;




class CountVectorKernel
{
    CudaContext context;
    CudaKernel kernel;

    int _vectorCount;

    public int VectorCount
    {
        get { return _vectorCount; }
        set
        {
            _vectorCount = value;
            kernel.BlockDimensions = _vectorCount;
        }
    }

    int _genLength;
    public int GenLength
    {
        get { return _genLength; }
        set
        {
            _genLength = value;
            kernel.SetConstantVariable("genLength", _genLength);
        }
    }


    public CountVectorKernel(CudaContext context, int vectorCount, int genLength)
    {
        this.context = context;
        kernel = context.LoadKernel("kernels/evolutionary2.ptx", "countVectors");
        VectorCount = vectorCount;
        GenLength = genLength;
        kernel.GridDimensions = 1;
    }


    public void Calculate(CudaDeviceVariable<byte> toCalc, CudaDeviceVariable<int> result)
    {
        kernel.Run(toCalc.DevicePointer, result.DevicePointer);

    }

}