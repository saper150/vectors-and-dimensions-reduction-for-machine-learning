using ManagedCuda;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

[TestFixture]
class ThrustTests
{


    [Test]

    public void MiniTest() {

        var context = new CudaContext();
        var data = new int[] {2,5,8,9,6,3,5,4,0,8,8 };
        var min = Thrust.Min(data);
        Assert.AreEqual(0, min);
    }

    [Test]
    public void MinfTest()
    {
        var context = new CudaContext();
        var data = new float[] { 2, 5, 8, 9, 6, 3, 5, 4, 0, 8, 8 };
        float min = Thrust.Min(data);
        Assert.AreEqual(0, min);
    }

    [Test]
    public void MaxfTest()
    {
        var context = new CudaContext();
        var data = new float[] { 2, 5, 8, 9, 6, 3, 5, 4, 0, 8, 8 };
        float max = Thrust.Max(data);
        Assert.AreEqual(9, max);
    }


    [Test]
    public void MaxiTest()
    {
        using (var context = new CudaContext())
        {
            var data = new int[] { 2, 5, 8, 8, 6, 3, 5, 4, 0, 8, 8 };
            int max = Thrust.Max(data);
            Assert.AreEqual(8, max);
        }
    }

    [Test]
    public void AvrageIntegerTest()
    {
        using (var context = new CudaContext())
        {
            var data = new int[] { 2, 5, 8, 9, 6, 3, 5, 4, 0, 8, 8 };
            float avg = Thrust.Avrage(data);
            Assert.AreEqual((float)data.Average(), avg);
        }
    }

    [Test]
    public void AvrageFloatTest()
    {
        using (var context = new CudaContext())
        {
            var data = new float[] { 2, 5, 8, 9, 6, 3, 5, 4, 0, 8, 8 };
            float avg = Thrust.Avrage(data);
            Assert.AreEqual((float)data.Average(), avg);
        }
    }

    [Test]
    public void sequenceTest()
    {
        using (var context = new CudaContext())
        {
            CudaDeviceVariable<int> t = new CudaDeviceVariable<int>(50);
            Thrust.seaquance(t);

            int[] hostSeaquance = t;

            for (int i = 0; i < hostSeaquance.Length; i++)
            {
                Assert.AreEqual(i, hostSeaquance[i]);
            }

        }
    }



    [Test]
    public void sort_by_keyTest()
    {
        using (var context = new CudaContext())
        {

            var hostKeys = new float[] {2,8,4,2,5,8,4,2,8,5,4,56,145,4,65 };
            var hostValues = new int[] { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 };

            CudaDeviceVariable<int> deviceValues = hostValues;
            CudaDeviceVariable<float> deviceKeys = hostKeys;
            Thrust.sort_by_key(deviceKeys, deviceValues);

            Array.Sort(hostKeys, hostValues);

            Assert.True(Enumerable.SequenceEqual(hostKeys, (float[])deviceKeys));
            Assert.True(Enumerable.SequenceEqual(hostValues, (int[])deviceValues));


        }
    }












}

