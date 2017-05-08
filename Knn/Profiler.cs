using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


public class Profiler
{
    class Watch
    {
        public Stopwatch w = new Stopwatch();
        public List<TimeSpan> notedTimes = new List<TimeSpan>();
    }

    Dictionary<string, Watch> watches = new Dictionary<string, Watch>();

    public void Start(string watch) {
        if (!watches.ContainsKey(watch))
        {
            watches[watch] = new Watch();
        }
        watches[watch].w.Start();
    }

    public void Stop(string watch) {
        var w = watches[watch];
        w.notedTimes.Add(w.w.Elapsed);
        w.w.Reset();
    }

    public void Print() {

        foreach (var item in watches)
        {
            var times = item.Value.notedTimes;
            Console.WriteLine("-------------");
            Console.WriteLine($"name:\t{item.Key}");
            Console.WriteLine($"avrageTime:\t{times.Average(x=>x.Milliseconds)}");
            Console.WriteLine($"max time:\t{times.Max()}");
            Console.WriteLine($"min time:\t{times.Min()}");
            Console.WriteLine($"executedTimes:\t{times.Count()}");
            Console.WriteLine("-------------");

        }



    }



}

