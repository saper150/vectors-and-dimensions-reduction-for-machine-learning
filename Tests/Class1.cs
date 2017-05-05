using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;

namespace Nunit
{
    [TestFixture]
    public class Class1
    {

        [Test]
        public void FlatArrayTest()
        {
            var a = new float[,] {
                {1,2,3 },
                {4,5,6 },
                {7,8,9 },
                {10,11,12 }
            };
            var flat = new FlattArray<float>(a);

            Assert.AreEqual(a[1, 1], flat.Get(1, 1));
            Assert.AreEqual(a[0, 2], flat.Get(0, 2));
            Assert.AreEqual(a[2, 1], flat.Get(2, 1));


            var b = new float[] {
                1,2,3,
                4,5,6,
                7,8,9,
                1,2,5
            };
            var flat2 = new FlattArray<float>(b,3);
            Assert.AreEqual(a.GetLength(0),4);
            Assert.AreEqual(a.GetLength(1), 3);

            Assert.AreEqual(flat2[0,0], b[0]);
            Assert.AreEqual(flat2[0, 1], b[1]);
            Assert.AreEqual(flat2[3, 2], b[11]);

        }
        [Test]
        public void FlatArrayJaggeredConstructorTest() {
            float[][] data = new float[][] {
                new float[] {2,5,4,4 },
                new float[] {-5,84,8,8 },
                new float[] {-644,12,8,9 },
                new float[] {156,64,8,54 },
                new float[] {2,456,8,4 },
            };

            FlattArray<float> flat = new FlattArray<float>(data);

            for (int i = 0; i < data.Length; i++)
            {
                for (int j = 0; j < data[i].Length; j++)
                {
                    Assert.AreEqual(flat[i,j],data[i][j]);
                }
            }
            

        }


        [Test]
        public void FlatArrayFilterTest() {
            var a = new float[,] {
                {1,2,3 },
                {4,5,6 },
                {7,8,9 },
                {10,11,12 }
            };
            var flat = new FlattArray<float>(a);
            {
                var filtered = flat.Filter(new int[] { });
                Assert.AreEqual(0, filtered.GetLength(0));
                Assert.AreEqual(3, filtered.GetLength(1));
            }

            {
                var filtered = flat.Filter(new int[] { 0, 3 });
                Assert.AreEqual(2, filtered.GetLength(0));
                Assert.AreEqual(3, filtered.GetLength(1));

                Assert.AreEqual(flat[0,0],filtered[0,0]);
                Assert.AreNotEqual(flat[1, 0], filtered[1, 0]);

            }


        }

        [Test]
        public void SortTest() {
            var data = new float[,] {
                {0,1,2 },
                {3,4,5 },
                {6,7,8 },
                {9,10,11 },
            };
            var classes = new int[] {1,2,3,4 };

            var sortBy = new float[] { 2, 0, 1, 3 };
            var flat = new FlattArray<float>(data);
            var cuda = new CudaDataSet()
            {
                Classes = classes,
                Vectors = flat
            };

            var sorted = Sorting.Sort(cuda, sortBy);

            Assert.AreEqual(sorted.Vectors[0, 0], cuda.Vectors[1, 0]);
            Assert.AreEqual(sorted.Classes[0], cuda.Classes[1]);

            Assert.AreEqual(sorted.Vectors[1, 0], cuda.Vectors[2, 0]);
            Assert.AreEqual(sorted.Classes[1], cuda.Classes[2]);

            Assert.AreEqual(sorted.Vectors[2, 0], cuda.Vectors[0, 0]);
            Assert.AreEqual(sorted.Classes[2], cuda.Classes[0]);



        }

        [Test]
        public void FlatArrayTo2dTest() {
            var a = new float[,] {
                {1,2,3 },
                {4,5,6 },
                {7,8,9 },
                {10,11,12 }
            };
            var flat = new FlattArray<float>(a);

            var twoD = flat.To2d();

            for (int i = 0; i < flat.GetLength(0); i++)
            {
                for (int j = 0; j < flat.GetLength(1); j++)
                {
                    Assert.AreEqual(flat[i,j],twoD[i][j]);
                }
            }

        }

        [Test]
        public void SortDataTest()
        {
            var vectors = new float[][] {
                new float[] {1,2,3 },
                new float[] {3,2,4 },
                new float[] {7,8,9 },
                new float[] {6,5,8 }
            };
            var classes = new int[] { 1, 1, 2, 2 };
            var nearestNaboures = new int[][] {
                new int []{1,2 },
                new int []{3,0 },
                new int []{0,1 },
                new int []{0,2 },
            };
            var sortBy = new float[] { 3, 4, 2, 1 };

            var originaIndeces = new int[] {0,1,2,3 };

            HostDataset s = new HostDataset() {
                Vectors = vectors,
                Classes = classes,
                OrginalIndeces = originaIndeces
            };
            Drop3.SortDataDesc(s, nearestNaboures, sortBy);

            Assert.AreEqual(1, s.OrginalIndeces[0]);
            Assert.AreEqual(0 , s.OrginalIndeces[1]);
            Assert.AreEqual(2, s.OrginalIndeces[2]);
            Assert.AreEqual(3, s.OrginalIndeces[3]);

            Assert.AreEqual(vectors[0], s.Vectors[1]);

            Assert.AreEqual(0, nearestNaboures[1][0]);
            Assert.AreEqual(2, nearestNaboures[1][1]);

        }

        [Test]
        public void findKminimumTest() {
            float[] data = new float[] {1,2,3,4,-6,6,7,8,-1 };

            var res = Utils.findKminimum(3, data);

           Assert.AreEqual(3,res.Length);
            Assert.IsTrue(
                res.Where(x=>x.index == 0 && x.val == 1).Any()
                );
            Assert.IsTrue(
                res.Contains(
                    new global::Data()
                    {
                        val = -1,
                        index = 8
                    })
                );

            Assert.IsTrue(
                res.Contains(
                    new global::Data()
                    {
                        val = -6,
                        index = 4
                    })
                );




        }


        [Test]
        public void SwapTest(){
            var a = new int[] {
                1,2, 3,
                4,5,6,
                7,8,9
            };
            a.Swap(0, 1);
            Assert.AreEqual(a[0],2);
            Assert.AreEqual(a[1], 1);

            var b = new int[] {
                1,2, 3,
                4,5,6,
                7,8,9
            };

            var flat = new FlattArray<int>(b,3);
            flat.Swap(0, 1);
            Assert.AreEqual(4, flat[0, 0]);
            Assert.AreEqual(5, flat[0, 1]);
            Assert.AreEqual(6, flat[0, 2]);

            Assert.AreEqual( 1, flat[1, 0]);
            Assert.AreEqual(2, flat[1, 1]);
            Assert.AreEqual(3, flat[1, 2]);


        }


    }



}