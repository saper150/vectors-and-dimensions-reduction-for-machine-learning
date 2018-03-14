using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


public static class Profiler
{
    class Watch
    {
        public Stopwatch w = new Stopwatch();
        public List<TimeSpan> notedTimes = new List<TimeSpan>();
    }

    static Dictionary<string, Watch> watches = new Dictionary<string, Watch>();

    public static void Start(string watch) {
        if (!watches.ContainsKey(watch))
        {
            watches[watch] = new Watch();
        }
        watches[watch].w.Start();
    }

    public static void Stop(string watch) {
        var w = watches[watch];
        w.notedTimes.Add(w.w.Elapsed);
        w.w.Reset();
    }

    public static void Print() {

        foreach (var item in watches)
        {
            var times = item.Value.notedTimes;

            var sorted = times.OrderBy(x => x.TotalMilliseconds).ToArray();
            var mediana = sorted[sorted.Length/2];

            Console.WriteLine("-------------");
            Console.WriteLine($"name:\t{item.Key}");
            Console.WriteLine($"avrageTime:\t{times.Average(x=>x.TotalMilliseconds)}");
            Console.WriteLine($"mediana:\t{mediana}");
            Console.WriteLine($"max time:\t{times.Max(x=>x.TotalMilliseconds)}");
            Console.WriteLine($"min time:\t{times.Min(x=>x.TotalMilliseconds)}");
            Console.WriteLine($"executedTimes:\t{times.Count()}");
            Console.WriteLine("-------------");

        }



    }



}

