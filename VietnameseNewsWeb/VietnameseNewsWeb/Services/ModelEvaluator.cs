using System;
using System.Collections.Generic;
using System.Linq;
using VietnameseNewsWeb.Models;

namespace VietnameseNewsWeb.Services
{
    /// <summary>
    /// Service để đánh giá hiệu suất của model
    /// </summary>
    public class ModelEvaluator
    {
        /// <summary>
        /// Đánh giá model trên tập test
        /// </summary>
        public EvaluationResult Evaluate(List<NewsArticle> testData, NaiveBayesClassifier classifier)
        {
            if (testData == null || testData.Count == 0)
            {
                throw new ArgumentException("Dữ liệu test không được rỗng");
            }

            if (!classifier.IsTrained)
            {
                throw new InvalidOperationException("Classifier chưa được huấn luyện");
            }

            Console.WriteLine($"Đánh giá model trên {testData.Count} mẫu test...");
            var startTime = DateTime.Now;

            // Thực hiện phân loại
            var predictions = new List<(string actual, string predicted)>();
            
            foreach (var article in testData)
            {
                var result = classifier.Classify(article);
                predictions.Add((article.Category, result.PredictedClass));
            }

            // Tính các metrics
            var evaluationResult = CalculateMetrics(predictions);
            
            var duration = DateTime.Now - startTime;
            Console.WriteLine($"Hoàn thành đánh giá trong {duration.TotalMilliseconds:F0}ms");

            return evaluationResult;
        }

        /// <summary>
        /// Tính các metrics đánh giá
        /// </summary>
        private EvaluationResult CalculateMetrics(List<(string actual, string predicted)> predictions)
        {
            var result = new EvaluationResult();
            result.TotalSamples = predictions.Count;
            
            // Lấy danh sách các lớp
            result.ClassNames = predictions.Select(x => x.actual).Distinct().OrderBy(x => x).ToList();
            var classCount = result.ClassNames.Count;

            // Tạo confusion matrix
            result.ConfusionMatrix = new int[classCount, classCount];

            // Đếm các trường hợp
            var classCounts = new Dictionary<string, (int tp, int fp, int fn, int support)>();
            foreach (var className in result.ClassNames)
            {
                classCounts[className] = (0, 0, 0, 0);
            }

            // Tính confusion matrix và đếm TP, FP, FN
            foreach (var (actual, predicted) in predictions)
            {
                int actualIndex = result.ClassNames.IndexOf(actual);
                int predictedIndex = result.ClassNames.IndexOf(predicted);
                
                result.ConfusionMatrix[actualIndex, predictedIndex]++;

                // Cập nhật counts cho từng lớp
                foreach (var className in result.ClassNames)
                {
                    var counts = classCounts[className];
                    
                    if (actual == className && predicted == className)
                        counts.tp++; // True Positive
                    else if (actual != className && predicted == className)
                        counts.fp++; // False Positive
                    else if (actual == className && predicted != className)
                        counts.fn++; // False Negative
                    
                    if (actual == className)
                        counts.support++; // Support (số mẫu thực tế của lớp này)
                    
                    classCounts[className] = counts;
                }
            }

            // Tính metrics cho từng lớp
            int totalCorrect = 0;
            foreach (var className in result.ClassNames)
            {
                var counts = classCounts[className];
                var metrics = new ClassMetrics();
                
                metrics.TruePositive = counts.tp;
                metrics.FalsePositive = counts.fp;
                metrics.FalseNegative = counts.fn;
                metrics.Support = counts.support;

                totalCorrect += counts.tp;

                // Tính Precision, Recall, F1
                metrics.Precision = counts.tp + counts.fp > 0 ? 
                    (double)counts.tp / (counts.tp + counts.fp) : 0.0;
                
                metrics.Recall = counts.tp + counts.fn > 0 ? 
                    (double)counts.tp / (counts.tp + counts.fn) : 0.0;
                
                metrics.F1Score = metrics.Precision + metrics.Recall > 0 ? 
                    2 * metrics.Precision * metrics.Recall / (metrics.Precision + metrics.Recall) : 0.0;

                result.ClassMetrics[className] = metrics;
            }

            // Tính Accuracy tổng thể
            result.Accuracy = (double)totalCorrect / result.TotalSamples;

            // Tính Kappa coefficient
            result.Kappa = CalculateKappa(result.ConfusionMatrix, result.TotalSamples);

            return result;
        }

        /// <summary>
        /// Tính Cohen's Kappa coefficient
        /// </summary>
        private double CalculateKappa(int[,] confusionMatrix, int totalSamples)
        {
            int classes = confusionMatrix.GetLength(0);
            
            // Tính observed accuracy (Po)
            int correctPredictions = 0;
            for (int i = 0; i < classes; i++)
            {
                correctPredictions += confusionMatrix[i, i];
            }
            double po = (double)correctPredictions / totalSamples;

            // Tính expected accuracy (Pe)
            double pe = 0.0;
            for (int i = 0; i < classes; i++)
            {
                int actualCount = 0;
                int predictedCount = 0;
                
                for (int j = 0; j < classes; j++)
                {
                    actualCount += confusionMatrix[i, j];      // Tổng dòng i
                    predictedCount += confusionMatrix[j, i];   // Tổng cột i
                }
                
                pe += (double)(actualCount * predictedCount) / (totalSamples * totalSamples);
            }

            // Tính Kappa
            return (po - pe) / (1 - pe);
        }

        /// <summary>
        /// In confusion matrix
        /// </summary>
        public void PrintConfusionMatrix(EvaluationResult result)
        {
            Console.WriteLine("\n=== CONFUSION MATRIX ===");
            
            // Header
            Console.Write("Predicted -> ");
            foreach (var className in result.ClassNames)
            {
                Console.Write($"{className.Substring(0, Math.Min(4, className.Length)),-6}");
            }
            Console.WriteLine("| Total");

            Console.WriteLine("Actual");
            
            // Ma trận
            for (int i = 0; i < result.ClassNames.Count; i++)
            {
                Console.Write($"{result.ClassNames[i],-12} ");
                
                int rowTotal = 0;
                for (int j = 0; j < result.ClassNames.Count; j++)
                {
                    int count = result.ConfusionMatrix[i, j];
                    Console.Write($"{count,-6}");
                    rowTotal += count;
                }
                Console.WriteLine($"| {rowTotal}");
            }
        }

        /// <summary>
        /// In báo cáo chi tiết
        /// </summary>
        public void PrintDetailedReport(EvaluationResult result)
        {
            Console.WriteLine("\n=== BÁO CÁO ĐÁNH GIÁ CHI TIẾT ===");
            Console.WriteLine($"Tổng số mẫu: {result.TotalSamples}");
            Console.WriteLine($"Accuracy: {result.Accuracy:P2}");
            Console.WriteLine($"Kappa: {result.Kappa:F4}");

            var (weightedPrecision, weightedRecall, weightedF1) = result.GetWeightedAverages();
            Console.WriteLine($"Weighted Avg - Precision: {weightedPrecision:P2}, Recall: {weightedRecall:P2}, F1: {weightedF1:P2}");

            Console.WriteLine("\n=== METRICS THEO TỪNG LỚP ===");
            Console.WriteLine($"{"Class",-12} {"Precision",-10} {"Recall",-10} {"F1-Score",-10} {"Support",-8}");
            Console.WriteLine(new string('-', 60));

            foreach (var kvp in result.ClassMetrics.OrderBy(x => x.Key))
            {
                var metrics = kvp.Value;
                Console.WriteLine($"{kvp.Key,-12} {metrics.Precision:F3}      {metrics.Recall:F3}      {metrics.F1Score:F3}      {metrics.Support}");
            }
        }
    }
}
