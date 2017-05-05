using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

class Genetic
{

    public int PopSize { get; set; }
    public float Mutations { get; set; }

    static Random rand = new Random();
    static byte[] RandomGen(int length) {
        byte[] gen = new byte[length];

        for (int i = 0; i < length; i++)
        {
            gen[i] = rand.NextDouble() > 5.0 ? (byte)1 : (byte)0;
        }
        return gen;

    }

    byte[] Solve(int length,Func<byte[], float> fitness) {
        return Solve(RandomGen(length),fitness);
    }

    byte[][] CreatePopulation(byte[] initialGen)
    {
        byte[][] population = new byte[PopSize][];

        for (int i = 0; i < population.Length; i++)
        {
            population[i] = new byte[initialGen.Length];
            Array.Copy(initialGen, population[i], initialGen.Length);
            for (int j = 0; j < initialGen.Length; j++)
            {
                if (rand.NextDouble() < 0.2) {
                    population[i][j] = population[i][j] == 1 ? (byte)0 : (byte)1;
                }
            }
        }

        return population;
    }
    byte[] Solve(byte[] initialGen,Func<byte[],float> fitness)
    {




        return null;
    }


}

