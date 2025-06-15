using System.ComponentModel.DataAnnotations;

namespace VietnameseNewsWeb.Models
{
    /// <summary>
    /// Model cho request phân loại tin tức
    /// </summary>
    public class ClassificationRequest
    {
        [Required(ErrorMessage = "Vui lòng nhập nội dung bài báo")]
        [MinLength(10, ErrorMessage = "Nội dung bài báo phải có ít nhất 10 ký tự")]
        [Display(Name = "Nội dung bài báo")]
        public string NewsText { get; set; } = string.Empty;
    }

    /// <summary>
    /// Model cho response phân loại tin tức
    /// </summary>
    public class ClassificationResponse
    {
        public string OriginalText { get; set; } = string.Empty;
        public string PredictedCategory { get; set; } = string.Empty;
        public double Confidence { get; set; }
        public Dictionary<string, double> AllProbabilities { get; set; } = new();
        public DateTime ProcessingTime { get; set; }
        
        public string GetCategoryDisplayName()
        {
            return PredictedCategory switch
            {
                "Business" => "Kinh doanh",
                "Sports" => "Thể thao", 
                "Entertainment" => "Giải trí",
                "Technology" => "Công nghệ",
                "Health" => "Sức khỏe",
                "Education" => "Giáo dục",
                "Politics" => "Chính trị",
                _ => PredictedCategory
            };
        }

        public string GetConfidenceLevel()
        {
            return Confidence switch
            {
                >= 0.8 => "Rất cao",
                >= 0.6 => "Cao", 
                >= 0.4 => "Trung bình",
                >= 0.2 => "Thấp",
                _ => "Rất thấp"
            };
        }

        public string GetConfidenceClass()
        {
            return Confidence switch
            {
                >= 0.8 => "success",
                >= 0.6 => "info",
                >= 0.4 => "warning", 
                _ => "danger"
            };
        }
    }

    /// <summary>
    /// Model cho trang lỗi
    /// </summary>
    public class ErrorViewModel
    {
        public string? RequestId { get; set; }
        public bool ShowRequestId => !string.IsNullOrEmpty(RequestId);
    }

    /// <summary>
    /// Model cho request phân tích từ khóa
    /// </summary>
    public class KeywordAnalysisRequest
    {
        public string Text { get; set; } = string.Empty;
    }
}