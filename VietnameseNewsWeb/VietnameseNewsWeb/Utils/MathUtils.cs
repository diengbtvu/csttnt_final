using System;

namespace VietnameseNewsWeb.Utils
{
    /// <summary>
    /// Các hàm toán học hỗ trợ
    /// </summary>
    public static class MathUtils
    {
        /// <summary>
        /// Tính xác suất theo phân bố Gaussian (Normal Distribution)
        /// </summary>
        /// <param name="x">Giá trị cần tính xác suất</param>
        /// <param name="mean">Giá trị trung bình</param>
        /// <param name="variance">Phương sai</param>
        /// <returns>Xác suất theo phân bố Gaussian</returns>
        public static double GaussianProbability(double x, double mean, double variance)
        {
            // Đảm bảo variance không bằng 0
            const double epsilon = 1e-9;
            variance = Math.Max(variance, epsilon);

            // Công thức: (1/sqrt(2π*σ²)) * e^(-((x-μ)²)/(2σ²))
            double coefficient = 1.0 / Math.Sqrt(2 * Math.PI * variance);
            double exponent = -Math.Pow(x - mean, 2) / (2 * variance);
            
            return coefficient * Math.Exp(exponent);
        }

        /// <summary>
        /// Tính log của xác suất Gaussian (để tránh underflow)
        /// </summary>
        public static double LogGaussianProbability(double x, double mean, double variance)
        {
            const double epsilon = 1e-9;
            variance = Math.Max(variance, epsilon);

            double logCoefficient = -0.5 * Math.Log(2 * Math.PI * variance);
            double logExponent = -Math.Pow(x - mean, 2) / (2 * variance);
            
            return logCoefficient + logExponent;
        }

        /// <summary>
        /// Tính entropy của một tập dữ liệu
        /// </summary>
        public static double CalculateEntropy(double[] probabilities)
        {
            double entropy = 0.0;
            foreach (var p in probabilities)
            {
                if (p > 0)
                {
                    entropy -= p * Math.Log2(p);
                }
            }
            return entropy;
        }

        /// <summary>
        /// Tính khoảng cách Euclidean giữa hai vector
        /// </summary>
        public static double EuclideanDistance(double[] vector1, double[] vector2)
        {
            if (vector1.Length != vector2.Length)
            {
                throw new ArgumentException("Hai vector phải có cùng độ dài");
            }

            double sum = 0.0;
            for (int i = 0; i < vector1.Length; i++)
            {
                sum += Math.Pow(vector1[i] - vector2[i], 2);
            }
            
            return Math.Sqrt(sum);
        }

        /// <summary>
        /// Normalize vector về [0, 1]
        /// </summary>
        public static double[] NormalizeVector(double[] vector)
        {
            double min = double.MaxValue;
            double max = double.MinValue;
            
            foreach (var value in vector)
            {
                if (value < min) min = value;
                if (value > max) max = value;
            }

            if (Math.Abs(max - min) < 1e-10) // Tất cả giá trị giống nhau
            {
                return new double[vector.Length]; // Trả về vector 0
            }

            var normalized = new double[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                normalized[i] = (vector[i] - min) / (max - min);
            }
            
            return normalized;
        }

        /// <summary>
        /// Tính mean của một mảng
        /// </summary>
        public static double Mean(double[] values)
        {
            if (values.Length == 0) return 0.0;
            
            double sum = 0.0;
            foreach (var value in values)
            {
                sum += value;
            }
            return sum / values.Length;
        }

        /// <summary>
        /// Tính variance của một mảng
        /// </summary>
        public static double Variance(double[] values)
        {
            if (values.Length <= 1) return 0.0;
            
            double mean = Mean(values);
            double sumSquaredDiff = 0.0;
            
            foreach (var value in values)
            {
                sumSquaredDiff += Math.Pow(value - mean, 2);
            }
            
            return sumSquaredDiff / (values.Length - 1); // Sample variance
        }

        /// <summary>
        /// Tính standard deviation
        /// </summary>
        public static double StandardDeviation(double[] values)
        {
            return Math.Sqrt(Variance(values));
        }
    }
}
