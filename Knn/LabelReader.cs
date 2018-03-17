using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


public enum Type {
    param,
    label,
    skip
}


public class LabelReader {

    public char Separator { get; set; }
    public bool Header { get; set; }
    Dictionary<string, int> labelsDictionary = new Dictionary<string, int>();
    public Dictionary<int, string> labelIdMap = new Dictionary<int, string>();

    List<int> ClassList = new List<int>();
    public int rowCount { get; set; }


    public CudaDataSet<int> DataSet
    {
        get
        {
            var set = new CudaDataSet<int>
            {
                Vectors = new FlattArray<float>(
                    f.ToArray(),
                    description.Where(x => x == Type.param).Count()),
                Classes = ClassList.ToArray()
            };
            set.orginalIndeces = DataSetHelper.CreateIndeces(set);
            return set;
        }
    }


    public bool PreserveClasses { get; set; } = false;
    int dictionaryIndex = 0;

    Type[] description;
    int descriptionLen;
    int paramCount;

    public LabelReader(Type[] description)
    {
        Separator = ',';
        Header = false;
        if (description.Where(x => x == Type.label).Count() != 1)
        {
            throw new Exception("Label count is not 1");
        }
        this.description = description;
        descriptionLen = description.Length;

        paramCount = description.Where(x => x == Type.param).Count();

    }

    List<float> f = new List<float>();
    void ReadRow(string row)
    {
        var splited = row.Split(Separator).ToList();

        for (int i = 0; i < descriptionLen; i++)
        {
            if (description[i] == Type.param)
            {
                f.Add(float.Parse(splited[i]));

            }
            else if (description[i] == Type.label)
            {

                if (PreserveClasses)
                {
                    ClassList.Add(int.Parse(splited[i]));
                }
                else if (!labelsDictionary.ContainsKey(splited[i]))
                {
                    labelsDictionary[splited[i]] = dictionaryIndex;
                    ClassList.Add(dictionaryIndex);
                    labelIdMap[dictionaryIndex] = splited[i];
                    dictionaryIndex++;
                }
                else
                {
                    ClassList.Add(labelsDictionary[splited[i]]);
                }

            }

        }

    }


    public void ReadFile(string fileName)
    {
        using (var reader = new StreamReader(fileName))
        {
            if (Header)
            {
                reader.ReadLine();
            }

            while (!reader.EndOfStream)
            {
                rowCount++;
                ReadRow(reader.ReadLine());
            }

        }
    }

}



public class RegresionReader {

    public char Separator { get; set; }
    public bool Header { get; set; }


    List<float> ValuesList = new List<float>();
    public int rowCount { get; set; }


    public CudaDataSet<float> DataSet
    {
        get
        {
            var set = new CudaDataSet<float>
            {
                Vectors = new FlattArray<float>(
                    f.ToArray(),
                    description.Where(x => x == Type.param).Count()),
                Classes = ValuesList.ToArray()
            };
            set.orginalIndeces = DataSetHelper.CreateIndeces(set);
            return set;
        }
    }

    int dictionaryIndex = 0;

    Type[] description;
    int descriptionLen;
    int paramCount;

    public RegresionReader(Type[] description)
    {
        Separator = ',';
        Header = false;
        if (description.Where(x => x == Type.label).Count() != 1)
        {
            throw new Exception("Label count is not 1");
        }
        this.description = description;
        descriptionLen = description.Length;

        paramCount = description.Where(x => x == Type.param).Count();

    }

    List<float> f = new List<float>();
    void ReadRow(string row)
    {
        var splited = row.Split(Separator).ToList();

        for (int i = 0; i < descriptionLen; i++)
        {
            if (description[i] == Type.param)
            {
                f.Add(float.Parse(splited[i]));

            }
            else if (description[i] == Type.label)
            {
                ValuesList.Add(float.Parse(splited[i]));
            }

        }

    }

    public void ReadFile(string fileName)
    {
        using (var reader = new StreamReader(fileName))
        {
            if (Header)
            {
                reader.ReadLine();
            }

            while (!reader.EndOfStream)
            {
                rowCount++;
                ReadRow(reader.ReadLine());
            }

        }
    }

}

