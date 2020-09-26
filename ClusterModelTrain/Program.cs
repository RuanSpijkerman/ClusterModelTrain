using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.Collections.Generic;
using System.Linq;

namespace ClusterModelTrain
{
    class Program
    {
        static readonly string _dataPath1 = "D:\\Universiteit\\Discovery\\Training\\ClusterModelTrain\\ClusterModelTrain\\Data\\Group5.csv";


        static readonly string _modelPath = "D:\\Universiteit\\Discovery\\Training\\ClusterModelTrain\\ClusterModelTrain\\Data\\IncomeCluster5.zip";

        public static void TrainOption2()
        {

            var mlContext = new MLContext(seed: 0);
            IDataView trainingData = mlContext.Data.LoadFromTextFile<UserData>(_dataPath1, hasHeader: true, separatorChar: ',');

            var options = new KMeansTrainer.Options
            {
                NumberOfClusters = 4,
                NumberOfThreads = 1,
                FeatureColumnName = "Utilities"
            };

            var pipeline = mlContext.Clustering.Trainers.KMeans(options);


            var model = pipeline.Fit(trainingData);

            VBuffer<float>[] centroids = default;

            var modelParams = model.Model;
            modelParams.GetClusterCentroids(ref centroids, out int k);
            Console.WriteLine(
                $"The first 3 coordinates of the first centroid are: " +
                string.Join(", ", centroids[0].GetValues().ToArray().Take(3)));


        }
     



        public void TrainOption1() {
            var mlContext = new MLContext(seed: 0);

            IDataView dataView = mlContext.Data.LoadFromTextFile<UserData>(_dataPath1, hasHeader: false, separatorChar: ',');

            
            string featuresColumnName = "Features";


            var pipeline = mlContext.Transforms
                .Concatenate(featuresColumnName, "Utilities", "Transport", "Rent", "Loans", "Saving", "Insurance", "Education", "Communication", "Takeout", "Groceries", "Alcohol", "Entertainment", "Personal", "Clothing", "Cash", "Other")
                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 4));

            var model = pipeline.Fit(dataView);

            /*
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }
            */
            var prediction = predictor.Predict(new UserData());
        }

        static void Main(string[] args)
        {

           TrainOpt


        }
            
        }
}
