using System.Collections.Generic;

namespace VietnameseNewsWeb.Models
{
    /// <summary>
    /// Metrics đánh giá cho từng lớp
    /// </summary>
    public class ClassMetrics
    {
        public double Precision { get; set; }
        public double Recall { get; set; }
        public double F1Score { get; set; }
        public int TruePositive { get; set; }
        public int FalsePositive { get; set; }
        public int FalseNegative { get; set; }
        public int Support { get; set; } // Tổng số mẫu thực tế của lớp này

        public override string ToString()
        {
            return $"Precision: {Precision:P2}, Recall: {Recall:P2}, F1: {F1Score:P2}";
        }
    }

    /// <summary>
    /// Kết quả đánh giá tổng thể của model
    /// </summary>
    public class EvaluationResult
    {
        public double Accuracy { get; set; }
        public double Kappa { get; set; }
        public Dictionary<string, ClassMetrics> ClassMetrics { get; set; }
        public int[,] ConfusionMatrix { get; set; }
        public List<string> ClassNames { get; set; }
        public int TotalSamples { get; set; }

        public EvaluationResult()
        {
            ClassMetrics = new Dictionary<string, ClassMetrics>();
            ClassNames = new List<string>();
        }

        /// <summary>
        /// Tính weighted average của các metrics
        /// </summary>
        public (double precision, double recall, double f1) GetWeightedAverages()
        {
            double totalSupport = 0;
            double weightedPrecision = 0;
            double weightedRecall = 0;
            double weightedF1 = 0;

            foreach (var metric in ClassMetrics.Values)
            {
                totalSupport += metric.Support;
                weightedPrecision += metric.Precision * metric.Support;
                weightedRecall += metric.Recall * metric.Support;
                weightedF1 += metric.F1Score * metric.Support;
            }

            if (totalSupport > 0)
            {
                weightedPrecision /= totalSupport;
                weightedRecall /= totalSupport;
                weightedF1 /= totalSupport;
            }

            return (weightedPrecision, weightedRecall, weightedF1);
        }

        public override string ToString()
        {
            var (precision, recall, f1) = GetWeightedAverages();
            return $"Accuracy: {Accuracy:P2}, Precision: {precision:P2}, Recall: {recall:P2}, F1: {f1:P2}";
        }
    }
}
