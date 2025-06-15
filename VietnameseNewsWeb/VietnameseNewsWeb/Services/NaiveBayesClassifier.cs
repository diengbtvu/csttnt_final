using System;
using System.Collections.Generic;
using System.Linq;
using VietnameseNewsWeb.Models;
using VietnameseNewsWeb.Utils;

namespace VietnameseNewsWeb.Services
{
    /// <summary>
    /// Cài đặt thuật toán Naïve Bayes cho phân loại tin tức với log chi tiết
    /// </summary>
    public class NaiveBayesClassifier
    {
        private NaiveBayesModel _model;
        private readonly double _smoothingFactor = 1e-9; // Để tránh chia cho 0

        public NaiveBayesClassifier()
        {
            _model = new NaiveBayesModel();
        }

        /// <summary>
        /// Huấn luyện model với dữ liệu training
        /// </summary>
        public void Train(List<NewsArticle> trainingData)
        {
            if (trainingData == null || trainingData.Count == 0)
            {
                throw new ArgumentException("Dữ liệu training không được rỗng");
            }

            Console.WriteLine("Bắt đầu huấn luyện model...");
            var startTime = DateTime.Now;

            // Khởi tạo model
            _model = new NaiveBayesModel();
            _model.Classes = trainingData.Select(x => x.Category).Distinct().ToList();
            _model.Features = trainingData.First().Features.Keys.ToList();
            _model.TotalSamples = trainingData.Count;

            Console.WriteLine($"Số lớp: {_model.Classes.Count}");
            Console.WriteLine($"Số đặc trưng: {_model.Features.Count}");

            // Tính xác suất tiên nghiệm P(C)
            CalculateClassProbabilities(trainingData);

            // Tính thống kê đặc trưng P(X|C)
            CalculateFeatureStatistics(trainingData);

            var duration = DateTime.Now - startTime;
            Console.WriteLine($"Hoàn thành huấn luyện trong {duration.TotalMilliseconds:F0}ms");
        }

        /// <summary>
        /// Tính xác suất tiên nghiệm cho từng lớp
        /// </summary>
        private void CalculateClassProbabilities(List<NewsArticle> trainingData)
        {
            var totalCount = trainingData.Count;
            
            foreach (var className in _model.Classes)
            {
                var classCount = trainingData.Count(x => x.Category == className);
                _model.ClassProbabilities[className] = (double)classCount / totalCount;
            }

            Console.WriteLine("Xác suất tiên nghiệm:");
            foreach (var kvp in _model.ClassProbabilities)
            {
                Console.WriteLine($"  P({kvp.Key}) = {kvp.Value:F4}");
            }
        }

        /// <summary>
        /// Tính thống kê đặc trưng cho từng lớp
        /// </summary>
        private void CalculateFeatureStatistics(List<NewsArticle> trainingData)
        {
            foreach (var className in _model.Classes)
            {
                _model.FeatureStatistics[className] = new Dictionary<string, FeatureStatistics>();
                var classData = trainingData.Where(x => x.Category == className).ToList();

                foreach (var featureName in _model.Features)
                {
                    var values = classData.Select(x => x.GetFeature(featureName)).ToList();
                    var mean = values.Average();
                    var variance = values.Count > 1 ? 
                        values.Sum(x => Math.Pow(x - mean, 2)) / (values.Count - 1) : 
                        _smoothingFactor;

                    // Đảm bảo variance không bằng 0
                    variance = Math.Max(variance, _smoothingFactor);

                    _model.FeatureStatistics[className][featureName] = 
                        new FeatureStatistics(mean, variance, values.Count);
                }
            }

            Console.WriteLine($"Đã tính thống kê cho {_model.Classes.Count} lớp và {_model.Features.Count} đặc trưng");
        }

        /// <summary>
        /// Phân loại một bài báo
        /// </summary>
        public ClassificationResult Classify(NewsArticle article)
        {
            if (!_model.IsTrained)
            {
                throw new InvalidOperationException("Model chưa được huấn luyện");
            }

            var logProbabilities = new Dictionary<string, double>();

            foreach (var className in _model.Classes)
            {
                // Bắt đầu với log của xác suất tiên nghiệm
                var logProb = Math.Log(_model.ClassProbabilities[className]);

                // Cộng log của likelihood cho từng đặc trưng
                foreach (var featureName in _model.Features)
                {
                    var featureValue = article.GetFeature(featureName);
                    var stats = _model.FeatureStatistics[className][featureName];
                    
                    var likelihood = MathUtils.GaussianProbability(featureValue, stats.Mean, stats.Variance);
                    logProb += Math.Log(likelihood);
                }

                logProbabilities[className] = logProb;
            }

            // Tìm lớp có xác suất cao nhất
            var predictedClass = logProbabilities.OrderByDescending(x => x.Value).First().Key;

            // Chuyển về xác suất thực (không cần thiết nhưng để debug)
            var probabilities = new Dictionary<string, double>();
            var maxLogProb = logProbabilities.Values.Max();
            
            foreach (var kvp in logProbabilities)
            {
                probabilities[kvp.Key] = Math.Exp(kvp.Value - maxLogProb);
            }

            return new ClassificationResult(predictedClass, probabilities);
        }

        /// <summary>
        /// Phân loại nhiều bài báo cùng lúc
        /// </summary>
        public List<ClassificationResult> ClassifyBatch(List<NewsArticle> articles)
        {
            return articles.Select(Classify).ToList();
        }

        /// <summary>
        /// Lấy thông tin model
        /// </summary>
        public NaiveBayesModel GetModel()
        {
            return _model;
        }

        /// <summary>
        /// Kiểm tra xem model đã được train chưa
        /// </summary>
        public bool IsTrained => _model.IsTrained;

        /// <summary>
        /// In thông tin chi tiết về model Naive Bayes
        /// </summary>
        public void PrintModelInfo()
        {
            if (!_model.IsTrained)
            {
                Console.WriteLine("Model chưa được huấn luyện!");
                return;
            }

            Console.WriteLine("=== THÔNG TIN MODEL NAIVE BAYES ===");
            Console.WriteLine($"Tổng số mẫu training: {_model.TotalSamples}");
            Console.WriteLine($"Số lớp (classes): {_model.Classes.Count}");
            Console.WriteLine($"Số đặc trưng (features): {_model.Features.Count}");
            Console.WriteLine();

            // In danh sách các lớp
            Console.WriteLine("DANH SÁCH CÁC LỚP:");
            foreach (var className in _model.Classes.OrderBy(x => x))
            {
                Console.WriteLine($"  - {className}");
            }
            Console.WriteLine();

            // In xác suất tiên nghiệm P(C)
            Console.WriteLine("XÁC SUẤT TIÊN NGHIỆM P(C):");
            Console.WriteLine($"{"Lớp",-15} {"Xác suất",-12} {"Tỷ lệ %",-8}");
            Console.WriteLine(new string('-', 40));
            
            foreach (var kvp in _model.ClassProbabilities.OrderByDescending(x => x.Value))
            {
                Console.WriteLine($"{kvp.Key,-15} {kvp.Value:F6}     {kvp.Value * 100:F2}%");
            }
            Console.WriteLine();

            // In một số thống kê đặc trưng tiêu biểu
            Console.WriteLine("THỐNG KÊ ĐẶC TRƯNG (5 đặc trưng đầu tiên):");
            var sampleFeatures = _model.Features.Take(5).ToList();
            
            foreach (var feature in sampleFeatures)
            {
                Console.WriteLine($"\nĐặc trưng: '{feature}'");
                Console.WriteLine($"{"Lớp",-15} {"Mean",-8} {"Variance",-10} {"Count",-6}");
                Console.WriteLine(new string('-', 42));
                
                foreach (var className in _model.Classes)
                {
                    var stats = _model.FeatureStatistics[className][feature];
                    Console.WriteLine($"{className,-15} {stats.Mean:F3}    {stats.Variance:F6}   {stats.Count}");
                }
            }
        }

        /// <summary>
        /// In quá trình phân loại chi tiết cho một bài báo với log từng bước
        /// </summary>
        public void PrintClassificationProcess(NewsArticle article)
        {
            if (!_model.IsTrained)
            {
                Console.WriteLine("Model chưa được huấn luyện!");
                return;
            }

            Console.WriteLine("=== QUÁ TRÌNH PHÂN LOẠI CHI TIẾT ===");
            Console.WriteLine($"Bài báo ID: {article.Id}");
            Console.WriteLine($"Danh mục thực tế: {article.Category}");
            Console.WriteLine($"Số đặc trưng: {article.Features.Count}");
            Console.WriteLine();

            // Hiển thị một số đặc trưng có giá trị > 0
            var nonZeroFeatures = article.Features.Where(f => f.Value > 0).Take(10).ToList();
            if (nonZeroFeatures.Any())
            {
                Console.WriteLine("CÁC ĐẶC TRƯNG CÓ GIÁ TRỊ > 0 (10 đầu tiên):");
                foreach (var feature in nonZeroFeatures)
                {
                    Console.WriteLine($"  - {feature.Key}: {feature.Value}");
                }
                Console.WriteLine();
            }

            // Tính log probabilities cho từng lớp với log chi tiết từng bước
            var logProbabilities = new Dictionary<string, double>();

            Console.WriteLine("TÍNH TOÁN XÁC SUẤT CHO TỪNG LỚP (CHI TIẾT TỪNG BƯỚC):");
            Console.WriteLine();

            foreach (var className in _model.Classes)
            {
                Console.WriteLine($"LỚP: {className.ToUpper()}");
                Console.WriteLine(new string('=', 60));
                
                // BƯỚC 1: Xác suất tiên nghiệm
                var logPrior = Math.Log(_model.ClassProbabilities[className]);
                Console.WriteLine($"BƯỚC 1 - Xác suất tiên nghiệm:");
                Console.WriteLine($"  P({className}) = {_model.ClassProbabilities[className]:F6}");
                Console.WriteLine($"  Log P({className}) = ln({_model.ClassProbabilities[className]:F6}) = {logPrior:F6}");
                Console.WriteLine();

                // BƯỚC 2: Tính likelihood cho từng feature có giá trị > 0
                var significantFeatures = article.Features.Where(f => f.Value > 0).ToList();
                double totalLogLikelihood = 0;
                int featureCount = 0;

                Console.WriteLine($"BƯỚC 2 - Tính Likelihood cho {significantFeatures.Count} đặc trưng có giá trị > 0:");
                Console.WriteLine();

                foreach (var featureKvp in significantFeatures)
                {
                    var featureName = featureKvp.Key;
                    var featureValue = featureKvp.Value;
                    
                    if (_model.FeatureStatistics[className].ContainsKey(featureName))
                    {
                        var stats = _model.FeatureStatistics[className][featureName];
                        var likelihood = MathUtils.GaussianProbability(featureValue, stats.Mean, stats.Variance);
                        var logLikelihood = Math.Log(likelihood);
                        
                        totalLogLikelihood += logLikelihood;
                        featureCount++;
                        
                        Console.WriteLine($"  Feature [{featureCount}]: {featureName} = {featureValue}");
                        Console.WriteLine($"    Thống kê từ training data:");
                        Console.WriteLine($"      Mean (μ) = {stats.Mean:F3}");
                        Console.WriteLine($"      Variance (σ²) = {stats.Variance:F3}");
                        Console.WriteLine($"      Standard Deviation (σ) = {Math.Sqrt(stats.Variance):F3}");
                        Console.WriteLine();
                        
                        // Hiển thị công thức Gaussian chi tiết
                        var exponent = -0.5 * Math.Pow(featureValue - stats.Mean, 2) / stats.Variance;
                        var coefficient = 1.0 / Math.Sqrt(2 * Math.PI * stats.Variance);
                        
                        Console.WriteLine($"    Áp dụng công thức phân phối Gaussian:");
                        Console.WriteLine($"      P(X={featureValue}|{className}) = (1/√(2π×σ²)) × e^(-(x-μ)²/(2×σ²))");
                        Console.WriteLine($"      = (1/√(2π×{stats.Variance:F3})) × e^(-({featureValue}-{stats.Mean:F3})²/(2×{stats.Variance:F3}))");
                        Console.WriteLine($"      = {coefficient:E3} × e^({exponent:F3})");
                        Console.WriteLine($"      = {likelihood:E6}");
                        Console.WriteLine();
                        Console.WriteLine($"    Log P({featureName}={featureValue}|{className}) = ln({likelihood:E6}) = {logLikelihood:F6}");
                        Console.WriteLine($"    {new string('-', 50)}");
                        Console.WriteLine();
                    }
                }

                // BƯỚC 3: Kết hợp xác suất tiên nghiệm và likelihood
                var finalLogProb = logPrior + totalLogLikelihood;
                logProbabilities[className] = finalLogProb;

                Console.WriteLine($"BƯỚC 3 - Kết hợp theo công thức Naive Bayes:");
                Console.WriteLine($"  Log P({className}|X) = Log P({className}) + Σ Log P(Xi|{className})");
                Console.WriteLine($"  = {logPrior:F6} + ({totalLogLikelihood:F6})");
                Console.WriteLine($"  = {finalLogProb:F6}");
                Console.WriteLine();
                Console.WriteLine(new string('=', 60));
                Console.WriteLine();
            }

            // BƯỚC 4: So sánh và chọn lớp tốt nhất
            Console.WriteLine("BƯỚC 4 - SO SÁNH VÀ CHỌN LỚP TỐT NHẤT:");
            Console.WriteLine();
            
            var rankedResults = logProbabilities.OrderByDescending(x => x.Value).ToList();
            var predictedClass = rankedResults[0].Key;
            
            Console.WriteLine("KẾT QUẢ PHÂN LOẠI:");
            Console.WriteLine($"{"Lớp",-15} {"Log Probability",-15} {"Rank",-6}");
            Console.WriteLine(new string('-', 40));
            
            for (int i = 0; i < rankedResults.Count; i++)
            {
                var kvp = rankedResults[i];
                var marker = kvp.Key == predictedClass ? ">>>" : "   ";
                Console.WriteLine($"{marker} {kvp.Key,-15} {kvp.Value,-15:F6} #{i + 1}");
            }
            
            Console.WriteLine();
            Console.WriteLine($"Dự đoán: {predictedClass}");
            Console.WriteLine($"Chính xác: {(predictedClass == article.Category ? "Đúng" : "Sai")}");
            
            // Thêm thông tin về chênh lệch nếu dự đoán sai
            if (predictedClass != article.Category && article.Category != "Unknown")
            {
                if (logProbabilities.ContainsKey(article.Category))
                {
                    var actualLogProb = logProbabilities[article.Category];
                    var predictedLogProb = logProbabilities[predictedClass];
                    var difference = predictedLogProb - actualLogProb;
                    Console.WriteLine($"Chênh lệch log probability: {difference:F6}");
                    
                    var actualRank = rankedResults.FindIndex(x => x.Key == article.Category) + 1;
                    Console.WriteLine($"Thứ hạng lớp thực tế '{article.Category}': #{actualRank}");
                }
            }
        }

        /// <summary>
        /// In ma trận nhầm lẫn cho tập test
        /// </summary>
        public void PrintConfusionMatrixForTestSet(List<NewsArticle> testData)
        {
            if (!_model.IsTrained)
            {
                Console.WriteLine("Model chưa được huấn luyện!");
                return;
            }

            Console.WriteLine("=== ĐÁNH GIÁ TRÊN TẬP TEST ===");
            Console.WriteLine($"Số mẫu test: {testData.Count}");
            Console.WriteLine();

            // Thực hiện phân loại
            var predictions = new Dictionary<string, Dictionary<string, int>>();
            var classNames = _model.Classes.OrderBy(x => x).ToList();
            
            // Khởi tạo ma trận nhầm lẫn
            foreach (var actual in classNames)
            {
                predictions[actual] = new Dictionary<string, int>();
                foreach (var predicted in classNames)
                {
                    predictions[actual][predicted] = 0;
                }
            }

            int correctPredictions = 0;

            foreach (var article in testData)
            {
                var result = Classify(article);
                predictions[article.Category][result.PredictedClass]++;
                
                if (article.Category == result.PredictedClass)
                {
                    correctPredictions++;
                }
            }

            // In ma trận nhầm lẫn
            Console.WriteLine("MA TRẬN NHẦM LẪN:");
            Console.Write($"{"Actual\\Predicted",-15}");
            foreach (var predicted in classNames)
            {
                Console.Write($"{predicted,-12}");
            }
            Console.WriteLine();
            Console.WriteLine(new string('-', 15 + classNames.Count * 12));

            foreach (var actual in classNames)
            {
                Console.Write($"{actual,-15}");
                foreach (var predicted in classNames)
                {
                    Console.Write($"{predictions[actual][predicted],-12}");
                }
                Console.WriteLine();
            }

            Console.WriteLine();
            Console.WriteLine($"Độ chính xác tổng thể: {(double)correctPredictions / testData.Count:P2}");

            // Tính precision, recall, F1 cho từng lớp
            Console.WriteLine();
            Console.WriteLine("METRICS CHI TIẾT:");
            Console.WriteLine($"{"Lớp",-15} {"Precision",-10} {"Recall",-10} {"F1-Score",-10}");
            Console.WriteLine(new string('-', 50));

            foreach (var className in classNames)
            {
                int tp = predictions[className][className];
                int fp = classNames.Where(c => c != className).Sum(c => predictions[c][className]);
                int fn = classNames.Where(c => c != className).Sum(c => predictions[className][c]);

                double precision = tp + fp > 0 ? (double)tp / (tp + fp) : 0;
                double recall = tp + fn > 0 ? (double)tp / (tp + fn) : 0;
                double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;

                Console.WriteLine($"{className,-15} {precision:F3}      {recall:F3}    {f1:F3}");
            }
        }

        /// <summary>
        /// Lấy thông tin model chi tiết
        /// </summary>
        public ModelInfo GetDetailedModelInfo()
        {
            if (!_model.IsTrained)
            {
                return null;
            }

            var modelInfo = new ModelInfo
            {
                TotalSamples = _model.TotalSamples,
                TotalClasses = _model.Classes.Count,
                TotalFeatures = _model.Features.Count,
                ClassNames = _model.Classes.OrderBy(x => x).ToList(),
                ClassProbabilities = new Dictionary<string, double>(_model.ClassProbabilities)
            };

            // Thu thập thống kê đặc trưng
            foreach (var className in _model.Classes)
            {
                modelInfo.FeatureStatistics[className] = new Dictionary<string, (double, double, int)>();
                foreach (var featureName in _model.Features.Take(20)) // Lấy 20 đặc trưng đầu tiên để tránh quá tải
                {
                    if (_model.FeatureStatistics[className].ContainsKey(featureName))
                    {
                        var stats = _model.FeatureStatistics[className][featureName];
                        modelInfo.FeatureStatistics[className][featureName] = (stats.Mean, stats.Variance, stats.Count);
                    }
                }
            }

            return modelInfo;
        }

        /// <summary>
        /// Lấy phân tích chi tiết quá trình phân loại cho giao diện web
        /// </summary>
        public NaiveBayesAnalysisResult GetDetailedClassificationAnalysis(NewsArticle article, int maxFeaturesToShow = 10)
        {
            if (!_model.IsTrained)
            {
                return null;
            }

            var result = new NaiveBayesAnalysisResult
            {
                ArticleId = article.Id.ToString(),
                OriginalText = "", // Sẽ được set từ ngoài
                ActualCategory = article.Category,
                TotalFeatures = article.Features.Count,
                Model = GetDetailedModelInfo()
            };

            // Lấy các đặc trưng có giá trị > 0
            var nonZeroFeatures = article.Features.Where(f => f.Value > 0).ToList();
            result.SignificantFeatures = nonZeroFeatures.Count;
            result.NonZeroFeatures = nonZeroFeatures.Take(maxFeaturesToShow)
                .Select(f => new Dictionary<string, object> { { "name", f.Key }, { "value", f.Value } })
                .ToList();

            // Tính log probabilities cho từng lớp với thông tin chi tiết
            var logProbabilities = new Dictionary<string, double>();

            foreach (var className in _model.Classes)
            {
                var classAnalysis = new ClassAnalysis
                {
                    ClassName = className,
                    PriorProbability = _model.ClassProbabilities[className],
                    LogPriorProbability = Math.Log(_model.ClassProbabilities[className])
                };

                // Phân tích từng đặc trưng có giá trị > 0
                double totalLogLikelihood = 0;
                var significantFeatures = nonZeroFeatures.Take(maxFeaturesToShow);

                foreach (var featureKvp in significantFeatures)
                {
                    var featureName = featureKvp.Key;
                    var featureValue = featureKvp.Value;

                    if (_model.FeatureStatistics[className].ContainsKey(featureName))
                    {
                        var stats = _model.FeatureStatistics[className][featureName];
                        var likelihood = MathUtils.GaussianProbability(featureValue, stats.Mean, stats.Variance);
                        var logLikelihood = Math.Log(likelihood);

                        totalLogLikelihood += logLikelihood;

                        var featureAnalysis = new FeatureAnalysis
                        {
                            FeatureName = featureName,
                            FeatureValue = featureValue,
                            Mean = stats.Mean,
                            Variance = stats.Variance,
                            StandardDeviation = Math.Sqrt(stats.Variance),
                            GaussianProbability = likelihood,
                            LogLikelihood = logLikelihood,
                            GaussianFormula = $"P(X={featureValue:F1}|{className}) = (1/√(2π×σ²)) × e^(-(x-μ)²/(2×σ²))",
                            GaussianCalculation = $"= (1/√(2π×{stats.Variance:F3})) × e^(-({featureValue:F1}-{stats.Mean:F3})²/(2×{stats.Variance:F3})) = {likelihood:E6}"
                        };

                        classAnalysis.Features.Add(featureAnalysis);
                    }
                }

                classAnalysis.TotalLogLikelihood = totalLogLikelihood;
                classAnalysis.FinalLogProbability = classAnalysis.LogPriorProbability + totalLogLikelihood;
                logProbabilities[className] = classAnalysis.FinalLogProbability;

                result.ClassAnalyses.Add(classAnalysis);
            }

            // Xếp hạng và xác định lớp dự đoán
            var rankedResults = logProbabilities.OrderByDescending(x => x.Value).ToList();
            result.PredictedClass = rankedResults[0].Key;
            result.PredictedLogProbability = rankedResults[0].Value;

            for (int i = 0; i < result.ClassAnalyses.Count; i++)
            {
                var analysis = result.ClassAnalyses.First(a => a.ClassName == rankedResults[i].Key);
                analysis.Rank = i + 1;
                analysis.IsPredicted = i == 0;
            }

            // Thông tin về độ chính xác
            result.IsCorrect = result.PredictedClass == article.Category;
            if (logProbabilities.ContainsKey(article.Category))
            {
                result.ActualLogProbability = logProbabilities[article.Category];
                result.LogProbabilityDifference = result.PredictedLogProbability - result.ActualLogProbability;
                result.ActualClassRank = rankedResults.FindIndex(x => x.Key == article.Category) + 1;
            }

            return result;
        }
    }
}
