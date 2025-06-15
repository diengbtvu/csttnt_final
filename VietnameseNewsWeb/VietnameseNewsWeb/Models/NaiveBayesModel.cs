using System.Collections.Generic;

namespace VietnameseNewsWeb.Models
{
    /// <summary>
    /// Thống kê đặc trưng cho mỗi lớp (mean và variance)
    /// </summary>
    public class FeatureStatistics
    {
        public double Mean { get; set; }
        public double Variance { get; set; }
        public int Count { get; set; }

        public FeatureStatistics(double mean, double variance, int count)
        {
            Mean = mean;
            Variance = variance;
            Count = count;
        }

        public override string ToString()
        {
            return $"Mean: {Mean:F4}, Variance: {Variance:F4}, Count: {Count}";
        }
    }

    /// <summary>
    /// Model của thuật toán Naïve Bayes
    /// </summary>
    public class NaiveBayesModel
    {
        public Dictionary<string, double> ClassProbabilities { get; set; }
        public Dictionary<string, Dictionary<string, FeatureStatistics>> FeatureStatistics { get; set; }
        public List<string> Classes { get; set; }
        public List<string> Features { get; set; }
        public int TotalSamples { get; set; }

        public NaiveBayesModel()
        {
            ClassProbabilities = new Dictionary<string, double>();
            FeatureStatistics = new Dictionary<string, Dictionary<string, FeatureStatistics>>();
            Classes = new List<string>();
            Features = new List<string>();
        }

        /// <summary>
        /// Kiểm tra xem model đã được train chưa
        /// </summary>
        public bool IsTrained => Classes.Count > 0 && Features.Count > 0;

        public override string ToString()
        {
            return $"NaiveBayesModel: {Classes.Count} classes, {Features.Count} features, {TotalSamples} samples";
        }
    }
}
