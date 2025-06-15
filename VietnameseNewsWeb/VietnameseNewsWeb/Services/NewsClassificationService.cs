using VietnameseNewsWeb.Models;
using System.Text.RegularExpressions;
using System.Text;

namespace VietnameseNewsWeb.Services
{
    public class NewsClassificationService
    {
        private readonly NaiveBayesClassifier _classifier;
        private readonly ILogger<NewsClassificationService> _logger;
        private readonly Dictionary<string, List<string>> _categoryKeywords;

        public NewsClassificationService(ILogger<NewsClassificationService> logger)
        {
            _logger = logger;
            _classifier = new NaiveBayesClassifier();
            _categoryKeywords = InitializeKeywords();
            InitializeClassifier();
        }

        private Dictionary<string, List<string>> InitializeKeywords()
        {
            return new Dictionary<string, List<string>>
            {
                ["Technology"] = new List<string>
                {
                    "ai", "blockchain", "app", "điện thoại", "internet", "mạng 5g", "máy tính", 
                    "phần mềm", "robot", "startup", "thiết bị", "thông minh", "công nghệ", 
                    "số hóa", "ứng dụng", "website", "facebook", "google", "apple"
                },
                ["Sports"] = new List<string>
                {
                    "bóng đá", "bóng rổ", "cầu thủ", "đội tuyển", "giải đấu", "hlv", "huy chương", 
                    "olympic", "tennis", "thể thao", "trận đấu", "v-league", "vô địch", "world cup",
                    "sea games", "asian games", "premier league", "la liga", "bundesliga"
                },
                ["Entertainment"] = new List<string>
                {
                    "âm nhạc", "bài hát", "ca sĩ", "concert", "đạo diễn", "diễn viên", "gameshow", 
                    "idol", "liveshow", "mv", "nghệ sĩ", "phim", "sân khấu", "truyền hình",
                    "hollywood", "vpop", "kpop", "netflix", "youtube", "chương trình", 
                    "cuộc thi", "chung kết", "tài năng", "biểu diễn", "khán giả"
                },
                ["Business"] = new List<string>
                {
                    "cổ phiếu", "đầu tư", "doanh nghiệp", "gdp", "giá vàng", "kinh doanh", "lãi suất", 
                    "lạm phát", "ngân hàng", "thị trường", "thương mại", "xuất khẩu", "nhập khẩu",
                    "chứng khoán", "forex", "bitcoin", "bất động sản", "startup", "thương hiệu"
                },
                ["Politics"] = new List<string>
                {
                    "bộ trưởng", "chính phủ", "chính sách", "chủ tịch", "đại biểu", "đảng", "hiệp định", 
                    "hội nghị", "lãnh đạo", "luật", "ngoại giao", "quốc hội", "thủ tướng",
                    "bầu cử", "nghị quyết", "ủy ban", "trung ương", "quân đội", "quốc phòng", 
                    "vũ khí", "tên lửa", "quân sự", "chiến tranh", "binh sĩ", "lính", "phòng thủ",
                    "an ninh", "tướng", "đại tá", "thiếu tá", "trung úy", "radar", "máy bay chiến đấu",
                    "tàu chiến", "súng", "đạn", "bom", "lựu đạn", "xe tăng", "pháo", "căn cứ quân sự"
                },
                ["Health"] = new List<string>
                {
                    "bác sĩ", "bệnh viện", "biến chứng", "dịch bệnh", "phẫu thuật", "sức khỏe", 
                    "thuốc", "tiêm chủng", "ung thư", "vaccine", "y tế", "covid", "virus",
                    "bệnh nhân", "điều trị", "khám bệnh", "sars", "h5n1"
                },
                ["Education"] = new List<string>
                {
                    "đại học", "đào tạo", "điểm chuẩn", "giáo dục", "giáo viên", "học bổng", 
                    "học phí", "học sinh", "học tập", "môn học", "năm học", "tốt nghiệp", "trường học",
                    "thạc sĩ", "tiến sĩ", "nghiên cứu", "khoa học", "thi cử"
                }
            };
        }

        private void InitializeClassifier()
        {
            try
            {
                // Load dữ liệu từ file CSV
                var dataLoader = new DataLoader();
                string csvPath = Path.Combine(Directory.GetCurrentDirectory(), "vietnamese_news_dataset_cleaned.csv");
                
                if (!File.Exists(csvPath))
                {
                    _logger.LogWarning($"CSV file not found at {csvPath}. Using mock classifier.");
                    return;
                }

                var allData = dataLoader.LoadFromCsv(csvPath);
                if (allData.Count > 0)
                {
                    _classifier.Train(allData);
                    _logger.LogInformation($"Classifier trained with {allData.Count} samples");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error initializing classifier");
            }
        }

        public async Task<ClassificationResult> ClassifyTextAsync(string text)
        {
            return await Task.Run(() => ClassifyText(text));
        }

        private ClassificationResult ClassifyText(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                throw new ArgumentException("Text cannot be null or empty");
            }

            // Chỉ sử dụng Naive Bayes thuần túy
            if (!_classifier.IsTrained)
            {
                throw new InvalidOperationException("Naive Bayes model chưa được huấn luyện");
            }

            _logger.LogInformation("Using pure Naive Bayes classification");
            
            // Chuyển text thành NewsArticle với features
            var article = ExtractFeaturesFromText(text);
            
            // Phân loại bằng Naive Bayes
            return _classifier.Classify(article);
        }

        private ClassificationResult ClassifyByKeywords(string text)
        {
            var textLower = text.ToLower();
            var scores = new Dictionary<string, double>();

            foreach (var category in _categoryKeywords.Keys)
            {
                scores[category] = 0.0;
            }

            // Đếm số từ khóa xuất hiện trong text
            _logger.LogDebug("=== PHÂN TÍCH KEYWORD MATCHING ===");
            _logger.LogDebug("Text: {TextPreview}...", text.Substring(0, Math.Min(100, text.Length)));
            
            foreach (var kvp in _categoryKeywords)
            {
                string category = kvp.Key;
                var keywords = kvp.Value;
                double categoryScore = 0.0;
                var matchedKeywords = new List<string>();

                foreach (var keyword in keywords)
                {
                    int count = Regex.Matches(textLower, Regex.Escape(keyword.ToLower())).Count;
                    if (count > 0)
                    {
                        matchedKeywords.Add($"'{keyword}': {count}");
                    }
                    categoryScore += count;
                }

                // Normalize bằng số lượng keywords của category
                scores[category] = categoryScore / keywords.Count;
                
                if (matchedKeywords.Count > 0)
                {
                    _logger.LogDebug("Category {Category}: {MatchedKeywords}, Score: {Score:F4}", 
                        category, string.Join(", ", matchedKeywords), scores[category]);
                }
            }

            // Tìm category có điểm cao nhất
            var bestCategory = scores.OrderByDescending(x => x.Value).First();
            var totalScore = scores.Values.Sum();
            var confidence = totalScore > 0 ? bestCategory.Value / totalScore : 0.5;

            _logger.LogInformation("Classification result: {Category} (confidence: {Confidence:P2})", 
                bestCategory.Key, confidence);

            // Chuyển scores thành probabilities
            var probabilities = new Dictionary<string, double>();
            foreach (var kvp in scores)
            {
                probabilities[kvp.Key] = totalScore > 0 ? kvp.Value / totalScore : 1.0 / scores.Count;
            }

            return new ClassificationResult(bestCategory.Key, probabilities)
            {
                Confidence = confidence
            };
        }

        private NewsArticle ExtractFeaturesFromText(string text)
        {
            var article = new NewsArticle();
            var textLower = text.ToLower();

            _logger.LogDebug("=== FEATURE EXTRACTION DEBUG ===");
            _logger.LogDebug("Text: {Text}", text);

            // Extract features based on keyword counts
            foreach (var kvp in _categoryKeywords)
            {
                foreach (var keyword in kvp.Value)
                {
                    int count = Regex.Matches(textLower, Regex.Escape(keyword.ToLower())).Count;
                    // Normalize feature name to match CSV format (no accents, replace spaces with underscores)
                    string featureName = NormalizeFeatureName(keyword);
                    
                    if (count > 0)
                    {
                        _logger.LogDebug("Found keyword '{Keyword}' → '{FeatureName}': {Count} times", 
                            keyword, featureName, count);
                    }
                    
                    // Check if feature already exists (potential duplicate)
                    var existingValue = article.GetFeature(featureName);
                    if (existingValue > 0 && count > 0)
                    {
                        _logger.LogWarning("DUPLICATE FEATURE: '{FeatureName}' already has value {ExistingValue}, setting to {NewValue}", 
                            featureName, existingValue, count);
                    }
                    
                    // Apply keyword boosting to compensate for training data bias
                    double finalValue = count;
                    if (count > 0 && IsEntertainmentKeyword(keyword))
                    {
                        // Boost entertainment keywords to match training data expectations
                        // Training data has Entertainment mean ~4.0, so boost our values significantly
                        finalValue = count * 4.5; // This brings 1 → 4.5, closer to training mean
                        _logger.LogInformation("ENTERTAINMENT BOOST: '{Keyword}' boosted from {OriginalCount} to {BoostedValue}", 
                            keyword, count, finalValue);
                    }
                    else if (count > 0 && IsSportsKeyword(keyword))
                    {
                        // Boost sports keywords to compensate for training data bias
                        // Similar to entertainment, sports articles may be under-represented in training data
                        finalValue = count * 3.5; // This brings 1 → 3.5, adjusted for sports
                        _logger.LogInformation("SPORTS BOOST: '{Keyword}' boosted from {OriginalCount} to {BoostedValue}", 
                            keyword, count, finalValue);
                    }
                    
                    article.SetFeature(featureName, finalValue);
                }
            }

            // Log all non-zero features
            var nonZeroFeatures = article.Features.Where(f => f.Value > 0).ToList();
            _logger.LogDebug("Total non-zero features: {Count}", nonZeroFeatures.Count);
            foreach (var feature in nonZeroFeatures)
            {
                _logger.LogDebug("Final feature: {Name} = {Value}", feature.Key, feature.Value);
            }

            return article;
        }

        private bool IsEntertainmentKeyword(string keyword)
        {
            // Keywords that need boosting due to training data bias
            var entertainmentKeywords = new HashSet<string> 
            { 
                "âm nhạc", "ca sĩ", "chương trình", "nghệ sĩ", "concert", 
                "liveshow", "gameshow", "cuộc thi", "chung kết", "tài năng",
                "biểu diễn", "khán giả", "bài hát", "mv"
            };
            return entertainmentKeywords.Contains(keyword.ToLower());
        }

        private bool IsSportsKeyword(string keyword)
        {
            // Keywords that need boosting due to training data bias
            var sportsKeywords = new HashSet<string> 
            { 
                "bóng đá", "cầu thủ", "đội tuyển", "hlv", "giải đấu", "v-league", 
                "vô địch", "trận đấu", "thể thao", "olympic", "world cup", "huy chương",
                "bàn thắng", "tập luyện", "sân vận động", "khán giả"
            };
            return sportsKeywords.Contains(keyword.ToLower());
        }

        private string NormalizeFeatureName(string keyword)
        {
            // Remove Vietnamese accents and normalize to match CSV header format
            string normalized = keyword.ToLower()
                .Replace("ã", "a").Replace("á", "a").Replace("à", "a").Replace("ả", "a").Replace("ạ", "a")
                .Replace("ă", "a").Replace("ắ", "a").Replace("ằ", "a").Replace("ẳ", "a").Replace("ẵ", "a").Replace("ặ", "a")
                .Replace("â", "a").Replace("ấ", "a").Replace("ầ", "a").Replace("ẩ", "a").Replace("ẫ", "a").Replace("ậ", "a")
                .Replace("é", "e").Replace("è", "e").Replace("ẻ", "e").Replace("ẽ", "e").Replace("ẹ", "e")
                .Replace("ê", "e").Replace("ế", "e").Replace("ề", "e").Replace("ể", "e").Replace("ễ", "e").Replace("ệ", "e")
                .Replace("í", "i").Replace("ì", "i").Replace("ỉ", "i").Replace("ĩ", "i").Replace("ị", "i")
                .Replace("ó", "o").Replace("ò", "o").Replace("ỏ", "o").Replace("õ", "o").Replace("ọ", "o")
                .Replace("ô", "o").Replace("ố", "o").Replace("ồ", "o").Replace("ổ", "o").Replace("ỗ", "o").Replace("ộ", "o")
                .Replace("ơ", "o").Replace("ớ", "o").Replace("ờ", "o").Replace("ở", "o").Replace("ỡ", "o").Replace("ợ", "o")
                .Replace("ú", "u").Replace("ù", "u").Replace("ủ", "u").Replace("ũ", "u").Replace("ụ", "u")
                .Replace("ư", "u").Replace("ứ", "u").Replace("ừ", "u").Replace("ử", "u").Replace("ữ", "u").Replace("ự", "u")
                .Replace("ý", "y").Replace("ỳ", "y").Replace("ỷ", "y").Replace("ỹ", "y").Replace("ỵ", "y")
                .Replace("đ", "d")
                .Replace(" ", "_");
            
            return normalized;
        }

        public Dictionary<string, List<string>> GetCategoryKeywords()
        {
            return _categoryKeywords;
        }

        /// <summary>
        /// Phân tích tần xuất xuất hiện của các từ khóa trong văn bản
        /// </summary>
        public Dictionary<string, Dictionary<string, int>> AnalyzeKeywordFrequency(string text)
        {
            var textLower = text.ToLower();
            var result = new Dictionary<string, Dictionary<string, int>>();

            Console.WriteLine("=== PHÂN TÍCH TẦN XUẤT TỪ KHÓA ===");
            Console.WriteLine($"Văn bản phân tích: {text.Substring(0, Math.Min(100, text.Length))}...");
            Console.WriteLine();

            foreach (var categoryKvp in _categoryKeywords)
            {
                string category = categoryKvp.Key;
                var keywords = categoryKvp.Value;
                var categoryFrequency = new Dictionary<string, int>();

                Console.WriteLine($"DANH MỤC: {category.ToUpper()}");
                Console.WriteLine(new string('-', 50));

                int totalCategoryCount = 0;
                var foundKeywords = new List<(string keyword, int count)>();

                foreach (var keyword in keywords)
                {
                    int count = Regex.Matches(textLower, Regex.Escape(keyword.ToLower())).Count;
                    categoryFrequency[keyword] = count;
                    totalCategoryCount += count;

                    if (count > 0)
                    {
                        foundKeywords.Add((keyword, count));
                    }
                }

                // Sắp xếp các từ khóa tìm thấy theo tần xuất giảm dần
                foundKeywords = foundKeywords.OrderByDescending(x => x.count).ToList();

                if (foundKeywords.Any())
                {
                    Console.WriteLine("Từ khóa tìm thấy:");
                    foreach (var (keyword, count) in foundKeywords)
                    {
                        Console.WriteLine($"  - '{keyword}': {count} lần");
                    }
                }
                else
                {
                    Console.WriteLine("  Không tìm thấy từ khóa nào");
                }

                Console.WriteLine($"Tổng số từ khóa tìm thấy: {totalCategoryCount}");
                Console.WriteLine($"Tỷ lệ: {(totalCategoryCount > 0 ? (double)totalCategoryCount / keywords.Count * 100 : 0):F1}%");
                Console.WriteLine();

                result[category] = categoryFrequency;
            }

            return result;
        }

        /// <summary>
        /// In báo cáo tóm tắt tần xuất từ khóa
        /// </summary>
        public void PrintKeywordFrequencySummary(string text)
        {
            var analysis = AnalyzeKeywordFrequency(text);
            
            Console.WriteLine("=== TỔNG KẾT TẦN XUẤT TỪ KHÓA ===");
            
            var categoryTotals = new List<(string category, int totalCount, double percentage)>();
            
            foreach (var categoryKvp in analysis)
            {
                string category = categoryKvp.Key;
                var frequencies = categoryKvp.Value;
                
                int totalCount = frequencies.Values.Sum();
                int totalKeywords = frequencies.Count;
                double percentage = totalKeywords > 0 ? (double)totalCount / totalKeywords * 100 : 0;
                
                categoryTotals.Add((category, totalCount, percentage));
            }
            
            // Sắp xếp theo tổng số từ khóa tìm thấy
            categoryTotals = categoryTotals.OrderByDescending(x => x.totalCount).ToList();
            
            Console.WriteLine($"{"Danh mục",-15} {"Tổng số",-8} {"Tỷ lệ",-8} {"Độ phù hợp",-12}");
            Console.WriteLine(new string('=', 50));
            
            foreach (var (category, totalCount, percentage) in categoryTotals)
            {
                string suitability = GetSuitabilityLevel(percentage);
                Console.WriteLine($"{category,-15} {totalCount,-8} {percentage:F1}%{"",-3} {suitability,-12}");
            }
            
            if (categoryTotals.Any())
            {
                var bestMatch = categoryTotals.First();
                Console.WriteLine();
                Console.WriteLine($"Danh mục phù hợp nhất: {bestMatch.category} ({bestMatch.totalCount} từ khóa, {bestMatch.percentage:F1}%)");
            }
        }

        private string GetSuitabilityLevel(double percentage)
        {
            return percentage switch
            {
                >= 20 => "Rất cao",
                >= 10 => "Cao", 
                >= 5 => "Trung bình",
                >= 1 => "Thấp",
                _ => "Rất thấp"
            };
        }

        public bool IsClassifierTrained => _classifier.IsTrained;

        /// <summary>
        /// In thông tin chi tiết về Naive Bayes model
        /// </summary>
        public void PrintNaiveBayesModelInfo()
        {
            Console.WriteLine();
            Console.WriteLine("=== THÔNG TIN CHI TIẾT VỀ NAIVE BAYES MODEL ===");
            
            if (!_classifier.IsTrained)
            {
                Console.WriteLine("Model chưa được huấn luyện với dữ liệu thực!");
                Console.WriteLine("Đang sử dụng keyword-based classification");
                Console.WriteLine();
                PrintKeywordBasedInfo();
                return;
            }

            _classifier.PrintModelInfo();
        }

        /// <summary>
        /// Phân tích chi tiết quá trình phân loại một văn bản
        /// </summary>
        public void AnalyzeClassificationProcess(string text)
        {
            Console.WriteLine();
            Console.WriteLine("=== PHÂN TÍCH QUÁ TRÌNH PHÂN LOẠI NAIVE BAYES ===");
            Console.WriteLine($"Văn bản: {text.Substring(0, Math.Min(150, text.Length))}...");
            Console.WriteLine();

            if (!_classifier.IsTrained)
            {
                throw new InvalidOperationException("Model Naive Bayes chưa được huấn luyện");
            }

            // Chuyển text thành NewsArticle với features
            var article = ExtractFeaturesFromText(text);
            article.Category = "Unknown"; // Đặt category tạm thời

            // In quá trình phân loại chi tiết bằng Naive Bayes
            _classifier.PrintClassificationProcess(article);
        }

        /// <summary>
        /// Phân tích keyword-based classification
        /// </summary>
        private void AnalyzeKeywordClassification(string text)
        {
            var textLower = text.ToLower();
            var scores = new Dictionary<string, double>();
            var detailedMatches = new Dictionary<string, List<(string keyword, int count)>>();

            Console.WriteLine("PHÂN TÍCH KEYWORD MATCHING:");
            Console.WriteLine();

            foreach (var categoryKvp in _categoryKeywords)
            {
                string category = categoryKvp.Key;
                var keywords = categoryKvp.Value;
                double categoryScore = 0.0;
                var matches = new List<(string keyword, int count)>();

                Console.WriteLine($"Danh mục: {category}");
                Console.WriteLine($"Tổng số từ khóa: {keywords.Count}");

                foreach (var keyword in keywords)
                {
                    int count = Regex.Matches(textLower, Regex.Escape(keyword.ToLower())).Count;
                    if (count > 0)
                    {
                        matches.Add((keyword, count));
                        categoryScore += count;
                    }
                }

                scores[category] = categoryScore / keywords.Count; // Normalize
                detailedMatches[category] = matches;

                Console.WriteLine($"Từ khóa tìm thấy: {matches.Count}");
                Console.WriteLine($"Điểm số: {categoryScore} (normalized: {scores[category]:F4})");

                if (matches.Any())
                {
                    var topMatches = matches.OrderByDescending(x => x.count).Take(5);
                    Console.WriteLine("Top từ khóa:");
                    foreach (var (keyword, count) in topMatches)
                    {
                        Console.WriteLine($"    - '{keyword}': {count} lần");
                    }
                }
                Console.WriteLine();
            }

            // Kết quả cuối cùng
            var rankedResults = scores.OrderByDescending(x => x.Value).ToList();
            Console.WriteLine("KẾT QUẢ CUỐI CÙNG:");
            Console.WriteLine($"{"Danh mục",-15} {"Điểm số",-10} {"Từ khóa",-8} {"Rank",-6}");
            Console.WriteLine(new string('-', 45));

            for (int i = 0; i < rankedResults.Count; i++)
            {
                var kvp = rankedResults[i];
                var matchCount = detailedMatches[kvp.Key].Count;
                var marker = i == 0 ? ">>>" : "   ";
                Console.WriteLine($"{marker} {kvp.Key,-15} {kvp.Value,-10:F4} {matchCount,-8} #{i + 1}");
            }

            if (rankedResults.Any() && rankedResults[0].Value > 0)
            {
                var winner = rankedResults[0];
                Console.WriteLine();
                Console.WriteLine($"Dự đoán: {winner.Key} (confidence: {winner.Value:F4})");
            }
        }

        /// <summary>
        /// In thông tin về keyword-based classification
        /// </summary>
        private void PrintKeywordBasedInfo()
        {
            Console.WriteLine("THÔNG TIN KEYWORD-BASED CLASSIFICATION:");
            Console.WriteLine();

            int totalKeywords = 0;
            Console.WriteLine($"{"Danh mục",-15} {"Số từ khóa",-10} {"Ví dụ từ khóa",-30}");
            Console.WriteLine(new string('-', 60));

            var sortedCategories = _categoryKeywords.OrderBy(x => x.Key).ToList();
            foreach (var kvp in sortedCategories)
            {
                var category = kvp.Key;
                var keywords = kvp.Value;
                var examples = string.Join(", ", keywords.Take(3));
                
                Console.WriteLine($"{category,-15} {keywords.Count,-10} {examples,-30}");
                totalKeywords += keywords.Count;
            }

            Console.WriteLine(new string('-', 60));
            Console.WriteLine($"{"TỔNG CỘNG",-15} {totalKeywords,-10}");
            Console.WriteLine();

            Console.WriteLine("THUẬT TOÁN:");
            Console.WriteLine("  1. Đếm số lần xuất hiện của từng từ khóa trong văn bản");
            Console.WriteLine("  2. Tính điểm cho mỗi danh mục = tổng từ khóa tìm thấy / tổng từ khóa");
            Console.WriteLine("  3. Chọn danh mục có điểm cao nhất");
            Console.WriteLine("  4. Tính confidence = điểm danh mục tốt nhất / tổng điểm");
        }        /// <summary>
        /// Chạy phân tích toàn diện chỉ bằng Naive Bayes
        /// </summary>
        public void RunComprehensiveAnalysis(string text)
        {
            Console.WriteLine();
            Console.WriteLine("=== PHÂN TÍCH TOÀN DIỆN NAIVE BAYES ===");
            Console.WriteLine($"Thời gian: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            Console.WriteLine();

            // 1. In thông tin model
            PrintNaiveBayesModelInfo();

            // 2. Phân tích quá trình phân loại bằng Naive Bayes
            Console.WriteLine();
            Console.WriteLine("BƯỚC 1: PHÂN TÍCH QUÁ TRÌNH PHÂN LOẠI");
            AnalyzeClassificationProcess(text);

            // 3. Thực hiện phân loại và hiển thị kết quả
            Console.WriteLine();
            Console.WriteLine("BƯỚC 2: KẾT QUẢ PHÂN LOẠI");
            var result = ClassifyText(text);
            
            Console.WriteLine($"Danh mục dự đoán: {result.PredictedClass}");
            Console.WriteLine($"Độ tin cậy: {result.Confidence:P2}");
            Console.WriteLine();

            Console.WriteLine("PHÂN BỐ XÁC SUẤT:");
            var sortedProbs = result.Probabilities.OrderByDescending(x => x.Value).ToList();
            foreach (var kvp in sortedProbs)
            {
                var bar = new string('=', (int)(kvp.Value * 20));
                Console.WriteLine($"  {kvp.Key,-15} {kvp.Value:P2} {bar}");
            }

            Console.WriteLine();
            Console.WriteLine("Phân tích hoàn tất!");
        }

        /// <summary>
        /// Lấy phân tích chi tiết Naive Bayes cho giao diện web
        /// </summary>
        public NaiveBayesAnalysisResult GetDetailedNaiveBayesAnalysis(string text, int maxFeaturesToShow = 10)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return null;
            }

            try
            {
                // Trích xuất đặc trưng từ văn bản
                var article = ExtractFeaturesFromText(text);
                article.Category = "Unknown"; // Chưa biết danh mục thực tế

                if (!_classifier.IsTrained)
                {
                    _logger.LogWarning("Classifier chưa được huấn luyện với dữ liệu thực");
                    return null;
                }

                // Lấy phân tích chi tiết
                var analysis = _classifier.GetDetailedClassificationAnalysis(article, maxFeaturesToShow);
                if (analysis != null)
                {
                    analysis.OriginalText = text;
                }

                return analysis;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Lỗi khi phân tích chi tiết Naive Bayes");
                return null;
            }
        }
    }
}