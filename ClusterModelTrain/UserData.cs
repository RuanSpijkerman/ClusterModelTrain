using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace ClusterModelTrain
{
    public class UserData
    {
        [LoadColumn(0)]
        public float Utilities;
        [LoadColumn(1)]
        public float Transport;
        [LoadColumn(2)]
        public float Rent;
        [LoadColumn(3)]
        public float Loans;
        [LoadColumn(4)]
        public float Saving;
        [LoadColumn(5)]
        public float Insurance;
        [LoadColumn(6)]
        public float Education;
        [LoadColumn(7)]
        public float Communication;
        [LoadColumn(8)]
        public float Takeout;
        [LoadColumn(9)]
        public float Groceries;
        [LoadColumn(10)]
        public float Alcohol;
        [LoadColumn(11)]
        public float Entertainment;
        [LoadColumn(12)]
        public float Personal;
        [LoadColumn(13)]
        public float Clothing;
        [LoadColumn(14)]
        public float Cash;
        [LoadColumn(15)]
        public float Other;
    }

    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[] Distances;
    }
}
