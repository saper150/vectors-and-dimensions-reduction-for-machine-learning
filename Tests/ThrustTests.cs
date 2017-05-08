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
    public void AvrageTest()
    {
        using (var context = new CudaContext())
        {
            var data = new int[] { 2, 5, 8, 9, 6, 3, 5, 4, 0, 8, 8 };
            float max = Thrust.Avrage(data);
            Assert.AreEqual((float)data.Average(), max);
        }
    }

}

