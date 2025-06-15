using System.Collections.Generic;

namespace VietnameseNewsWeb.Models
{
    /// <summary>
    /// Đại diện cho một bài báo tin tức với các đặc trưng từ khóa
    /// </summary>
    public class NewsArticle
    {
        public int Id { get; set; }
        public Dictionary<string, double> Features { get; set; }
        public string Category { get; set; }
        
        public NewsArticle()
        {
            Features = new Dictionary<string, double>();
        }

        public NewsArticle(int id, string category)
        {
            Id = id;
            Category = category;
            Features = new Dictionary<string, double>();
        }

        /// <summary>
        /// Thêm hoặc cập nhật giá trị đặc trưng
        /// </summary>
        public void SetFeature(string featureName, double value)
        {
            Features[featureName] = value;
        }

        /// <summary>
        /// Lấy giá trị đặc trưng, trả về 0 nếu không tồn tại
        /// </summary>
        public double GetFeature(string featureName)
        {
            return Features.ContainsKey(featureName) ? Features[featureName] : 0.0;
        }

        public override string ToString()
        {
            return $"Article {Id}: {Category} ({Features.Count} features)";
        }
    }
}
