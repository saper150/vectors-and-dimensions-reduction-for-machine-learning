using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

public static class Thrust
{
    [DllImport("kernels/CudaFunctions.dll", EntryPoint = "Sum")]
    static extern int Sum(ManagedCuda.BasicTypes.SizeT ptr, int size);



    [DllImport("kernels/CudaFunctions.dll", EntryPoint = "Avragei")]
    static extern float Avragei(ManagedCuda.BasicTypes.SizeT ptr, int size);
    public static float Avrage(CudaDeviceVariable<int> a) {
        return Avragei(a.DevicePointer.Pointer, a.Size);
    }

    [DllImport("kernels/CudaFunctions.dll", EntryPoint = "Avragef")]
    static extern float Avragef(ManagedCuda.BasicTypes.SizeT ptr, int size);
    public static float Avrage(CudaDeviceVariable<float> a)
    {
        return Avragef(a.DevicePointer.Pointer, a.Size);
    }



    [DllImport("kernels/CudaFunctions.dll", EntryPoint = "Maxf")]
    static extern float Maxf(ManagedCuda.BasicTypes.SizeT ptr, int size);

    [DllImport("kernels/CudaFunctions.dll", EntryPoint = "Maxi")]
    static extern int Maxi(ManagedCuda.BasicTypes.SizeT ptr, int size);


    public static float Max(CudaDeviceVariable<float> a)=>
        Maxf(a.DevicePointer.Pointer, a.Size);

    public static int Max(CudaDeviceVariable<int> a) 
        => Maxi(a.DevicePointer.Pointer, a.Size);


    [DllImport("kernels/CudaFunctions.dll", EntryPoint = "Minf")]
    static extern float Minf(ManagedCuda.BasicTypes.SizeT ptr, int size);

    [DllImport("kernels/CudaFunctions.dll", EntryPoint = "Mini")]
    static extern int Mini(ManagedCuda.BasicTypes.SizeT ptr, int size);

    public static float Min(CudaDeviceVariable<float> a) =>
        Minf(a.DevicePointer.Pointer, a.Size);

    public static int Min(CudaDeviceVariable<int> a)
        => Mini(a.DevicePointer.Pointer, a.Size);


    [DllImport("kernels/CudaFunctions.dll", EntryPoint = "sort_by_key")]
    static extern void sort_by_key(ManagedCuda.BasicTypes.SizeT ptr, ManagedCuda.BasicTypes.SizeT pt2, int size);

    public static void sort_by_key(CudaDeviceVariable<float> keys, CudaDeviceVariable<int> values) {
        sort_by_key(keys.DevicePointer.Pointer, values.DevicePointer.Pointer, keys.Size);
    }

    [DllImport("kernels/CudaFunctions.dll", EntryPoint = "sort_by_keyDesc")]
    static extern void sort_by_keyDesc(ManagedCuda.BasicTypes.SizeT ptr, ManagedCuda.BasicTypes.SizeT pt2, int size);

    public static void sort_by_keyDesc(CudaDeviceVariable<float> keys, CudaDeviceVariable<int> values)
    {
        sort_by_key(keys.DevicePointer.Pointer, values.DevicePointer.Pointer, keys.Size);
    }


    [DllImport("kernels/CudaFunctions.dll", EntryPoint = "sequence")]
    static extern void sequence(ManagedCuda.BasicTypes.SizeT ptr, int size);

    public static void seaquance(CudaDeviceVariable<int> values) {
        sequence(values.DevicePointer.Pointer, values.Size);
    }

}

