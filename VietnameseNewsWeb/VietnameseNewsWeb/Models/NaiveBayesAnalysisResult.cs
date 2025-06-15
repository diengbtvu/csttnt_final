using System;
using System.Collections.Generic;

namespace VietnameseNewsWeb.Models
{
    /// <summary>
    /// Thông tin chi tiết về một đặc trưng trong quá trình phân loại
    /// </summary>
    public class FeatureAnalysis
    {
        public string FeatureName { get; set; } = string.Empty;
        public double FeatureValue { get; set; }
        public double Mean { get; set; }
        public double Variance { get; set; }
        public double StandardDeviation { get; set; }
        public double GaussianProbability { get; set; }
        public double LogLikelihood { get; set; }
        public string GaussianFormula { get; set; } = string.Empty;
        public string GaussianCalculation { get; set; } = string.Empty;
    }

    /// <summary>
    /// Kết quả phân tích chi tiết cho một lớp
    /// </summary>
    public class ClassAnalysis
    {
        public string ClassName { get; set; } = string.Empty;
        public double PriorProbability { get; set; }
        public double LogPriorProbability { get; set; }
        public List<FeatureAnalysis> Features { get; set; } = new List<FeatureAnalysis>();
        public double TotalLogLikelihood { get; set; }
        public double FinalLogProbability { get; set; }
        public int Rank { get; set; }
        public bool IsPredicted { get; set; }
    }

    /// <summary>
    /// Thông tin tổng quan về model Naive Bayes
    /// </summary>
    public class ModelInfo
    {
        public int TotalSamples { get; set; }
        public int TotalClasses { get; set; }
        public int TotalFeatures { get; set; }
        public List<string> ClassNames { get; set; } = new List<string>();
        public Dictionary<string, double> ClassProbabilities { get; set; } = new Dictionary<string, double>();
        public Dictionary<string, Dictionary<string, (double Mean, double Variance, int Count)>> FeatureStatistics { get; set; } = new Dictionary<string, Dictionary<string, (double, double, int)>>();
    }

    /// <summary>
    /// Simple feature information for display
    /// </summary>
    public class SimpleFeature
    {
        public string Name { get; set; } = string.Empty;
        public double Value { get; set; }
    }

    /// <summary>
    /// Kết quả phân tích toàn diện Naive Bayes cho giao diện web
    /// </summary>
    public class NaiveBayesAnalysisResult
    {
        public string ArticleId { get; set; } = string.Empty;
        public string OriginalText { get; set; } = string.Empty;
        public string ActualCategory { get; set; } = string.Empty;
        public int TotalFeatures { get; set; }
        public int SignificantFeatures { get; set; }
        public List<Dictionary<string, object>> NonZeroFeatures { get; set; } = new List<Dictionary<string, object>>();
        
        public ModelInfo Model { get; set; } = new ModelInfo();
        public List<ClassAnalysis> ClassAnalyses { get; set; } = new List<ClassAnalysis>();
        
        public string PredictedClass { get; set; } = string.Empty;
        public bool IsCorrect { get; set; }
        public double PredictedLogProbability { get; set; }
        public double ActualLogProbability { get; set; }
        public double LogProbabilityDifference { get; set; }
        public int ActualClassRank { get; set; }
        
        public DateTime AnalysisTime { get; set; } = DateTime.Now;
        public string NaiveBayesFormula { get; set; } = "P(C|X) = P(C) × ∏P(Xi|C)";
        public string LogFormula { get; set; } = "Log P(C|X) = Log P(C) + Σ Log P(Xi|C)";
    }

    /// <summary>
    /// Request model cho phân tích chi tiết Naive Bayes
    /// </summary>
    public class NaiveBayesAnalysisRequest
    {
        public string Text { get; set; } = string.Empty;
        public bool ShowAllFeatures { get; set; } = false;
        public int MaxFeaturesToShow { get; set; } = 10;
    }
}
