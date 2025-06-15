using System.Collections.Generic;

namespace VietnameseNewsWeb.Models
{
    /// <summary>
    /// Kết quả phân loại của thuật toán Naïve Bayes
    /// </summary>
    public class ClassificationResult
    {
        public string PredictedClass { get; set; }
        public Dictionary<string, double> Probabilities { get; set; }
        public double Confidence { get; set; }

        public ClassificationResult()
        {
            Probabilities = new Dictionary<string, double>();
        }

        public ClassificationResult(string predictedClass, Dictionary<string, double> probabilities)
        {
            PredictedClass = predictedClass;
            Probabilities = probabilities;
            
            // Tính confidence là xác suất cao nhất
            double maxProb = double.MinValue;
            foreach (var prob in probabilities.Values)
            {
                if (prob > maxProb)
                    maxProb = prob;
            }
            Confidence = maxProb;
        }

        public override string ToString()
        {
            return $"Predicted: {PredictedClass} (Confidence: {Confidence:F4})";
        }
    }
}
