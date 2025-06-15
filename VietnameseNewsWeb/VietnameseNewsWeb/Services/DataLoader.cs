using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using VietnameseNewsWeb.Models;

namespace VietnameseNewsWeb.Services
{
    /// <summary>
    /// Service để load dữ liệu từ file CSV
    /// </summary>
    public class DataLoader
    {
        /// <summary>
        /// Load dữ liệu từ file CSV
        /// </summary>
        public List<NewsArticle> LoadFromCsv(string filePath)
        {
            var articles = new List<NewsArticle>();
            
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File không tồn tại: {filePath}");
            }

            var lines = File.ReadAllLines(filePath);
            if (lines.Length < 2)
            {
                throw new InvalidDataException("File CSV phải có ít nhất 2 dòng (header và data)");
            }

            // Đọc header
            var headers = lines[0].Split(',');
            var featureHeaders = headers.Skip(1).Take(headers.Length - 2).ToList(); // Bỏ id và category
            
            Console.WriteLine($"Loaded {featureHeaders.Count} features from CSV");

            // Đọc dữ liệu
            for (int i = 1; i < lines.Length; i++)
            {
                try
                {
                    var values = lines[i].Split(',');
                    if (values.Length != headers.Length)
                    {
                        Console.WriteLine($"Bỏ qua dòng {i + 1}: Số cột không khớp");
                        continue;
                    }

                    var article = new NewsArticle();
                    
                    // Parse ID
                    if (int.TryParse(values[0], out int id))
                    {
                        article.Id = id;
                    }
                    else
                    {
                        article.Id = i; // Use line number as fallback
                    }

                    // Parse features
                    for (int j = 1; j < values.Length - 1; j++)
                    {
                        string featureName = featureHeaders[j - 1];
                        if (double.TryParse(values[j], NumberStyles.Float, CultureInfo.InvariantCulture, out double featureValue))
                        {
                            article.SetFeature(featureName, featureValue);
                        }
                        else
                        {
                            article.SetFeature(featureName, 0.0); // Default value for invalid data
                        }
                    }

                    // Parse category
                    article.Category = values[values.Length - 1].Trim();

                    articles.Add(article);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Lỗi khi parse dòng {i + 1}: {ex.Message}");
                }
            }

            Console.WriteLine($"Đã load {articles.Count} bài báo từ {filePath}");
            return articles;
        }

        /// <summary>
        /// Chia dữ liệu thành tập train và test
        /// </summary>
        public (List<NewsArticle> trainData, List<NewsArticle> testData) SplitData(
            List<NewsArticle> data, double trainRatio = 0.8, int? randomSeed = null)
        {
            if (trainRatio <= 0 || trainRatio >= 1)
            {
                throw new ArgumentException("Train ratio phải trong khoảng (0, 1)");
            }

            var random = randomSeed.HasValue ? new Random(randomSeed.Value) : new Random();
            var shuffledData = data.OrderBy(x => random.Next()).ToList();
            
            int trainSize = (int)(shuffledData.Count * trainRatio);
            
            var trainData = shuffledData.Take(trainSize).ToList();
            var testData = shuffledData.Skip(trainSize).ToList();

            Console.WriteLine($"Chia dữ liệu: {trainData.Count} train, {testData.Count} test");
            return (trainData, testData);
        }

        /// <summary>
        /// Phân tích phân bố dữ liệu theo các lớp
        /// </summary>
        public void AnalyzeDataDistribution(List<NewsArticle> data)
        {
            var distribution = data.GroupBy(x => x.Category)
                                  .ToDictionary(g => g.Key, g => g.Count());

            Console.WriteLine("\n=== PHÂN BỐ DỮ LIỆU ===");
            Console.WriteLine($"{"Danh mục",-15} {"Số lượng",-10} {"Tỷ lệ",-10}");
            Console.WriteLine(new string('-', 35));

            foreach (var kvp in distribution.OrderByDescending(x => x.Value))
            {
                double percentage = (double)kvp.Value / data.Count * 100;
                Console.WriteLine($"{kvp.Key,-15} {kvp.Value,-10} {percentage:F1}%");
            }

            Console.WriteLine($"\nTổng cộng: {data.Count} bài báo, {distribution.Count} danh mục");
        }
    }
}
