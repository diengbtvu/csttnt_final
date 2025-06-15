using Microsoft.AspNetCore.Mvc;
using VietnameseNewsWeb.Models;
using VietnameseNewsWeb.Services;
using System.Diagnostics;

namespace VietnameseNewsWeb.Controllers
{
    public class HomeController : Controller
    {
        private readonly NewsClassificationService _classificationService;
        private readonly ILogger<HomeController> _logger;

        public HomeController(NewsClassificationService classificationService, ILogger<HomeController> logger)
        {
            _classificationService = classificationService;
            _logger = logger;
        }

        public IActionResult Index()
        {
            var model = new ClassificationRequest();
            return View(model);
        }

        [HttpPost]
        public async Task<IActionResult> Classify(ClassificationRequest request)
        {
            if (!ModelState.IsValid)
            {
                return View("Index", request);
            }

            try
            {
                // Phân loại bài báo
                var result = await _classificationService.ClassifyTextAsync(request.NewsText);
                
                var viewModel = new ClassificationResponse
                {
                    OriginalText = request.NewsText,
                    PredictedCategory = result.PredictedClass,
                    Confidence = result.Confidence,
                    AllProbabilities = result.Probabilities,
                    ProcessingTime = DateTime.Now
                };

                return View("Result", viewModel);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error occurred during classification");
                ModelState.AddModelError("", "Có lỗi xảy ra khi phân loại bài báo: " + ex.Message);
                return View("Index", request);
            }
        }

        public IActionResult About()
        {
            return View();
        }

        [HttpPost]
        public IActionResult AnalyzeKeywords([FromBody] KeywordAnalysisRequest request)
        {
            if (string.IsNullOrWhiteSpace(request?.Text))
            {
                return BadRequest("Văn bản không được để trống");
            }

            try
            {
                // Phân tích tần suất từ khóa  
                var frequencyAnalysis = _classificationService.AnalyzeKeywordFrequency(request.Text);
                
                // In kết quả ra console
                _classificationService.PrintKeywordFrequencySummary(request.Text);
                
                return Json(new { 
                    success = true, 
                    message = "Phân tích tần suất từ khóa đã hoàn thành. Xem kết quả chi tiết trong console.",
                    data = frequencyAnalysis 
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error occurred during keyword analysis");
                return Json(new { 
                    success = false, 
                    message = "Có lỗi xảy ra khi phân tích: " + ex.Message 
                });
            }
        }

        [HttpPost]
        public IActionResult AnalyzeNaiveBayes([FromBody] KeywordAnalysisRequest request)
        {
            if (string.IsNullOrWhiteSpace(request?.Text))
            {
                return BadRequest("Văn bản không được để trống");
            }

            try
            {
                // Chạy phân tích toàn diện Naive Bayes
                _classificationService.RunComprehensiveAnalysis(request.Text);
                
                return Json(new { 
                    success = true, 
                    message = "Phân tích Naive Bayes đã hoàn thành. Xem kết quả chi tiết trong console server." 
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error occurred during Naive Bayes analysis");
                return Json(new { 
                    success = false, 
                    message = "Có lỗi xảy ra khi phân tích Naive Bayes: " + ex.Message 
                });
            }
        }

        [HttpGet]
        public IActionResult ShowModelInfo()
        {
            try
            {
                // In thông tin model Naive Bayes
                _classificationService.PrintNaiveBayesModelInfo();
                
                return Json(new { 
                    success = true, 
                    message = "Thông tin Naive Bayes model đã được in ra console server." 
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error occurred while showing model info");
                return Json(new { 
                    success = false, 
                    message = "Có lỗi xảy ra khi hiển thị thông tin model: " + ex.Message 
                });
            }
        }

        [HttpPost]
        public IActionResult GetDetailedNaiveBayesAnalysis([FromBody] NaiveBayesAnalysisRequest request)
        {
            if (string.IsNullOrWhiteSpace(request?.Text))
            {
                return BadRequest("Văn bản không được để trống");
            }

            try
            {
                var analysis = _classificationService.GetDetailedNaiveBayesAnalysis(
                    request.Text, 
                    request.MaxFeaturesToShow > 0 ? request.MaxFeaturesToShow : 10
                );

                if (analysis == null)
                {
                    return Json(new { 
                        success = false, 
                        message = "Không thể thực hiện phân tích. Model chưa được huấn luyện hoặc có lỗi xảy ra." 
                    });
                }

                return Json(new { 
                    success = true, 
                    data = analysis,
                    message = "Phân tích Naive Bayes chi tiết đã hoàn thành"
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error occurred during detailed Naive Bayes analysis");
                return Json(new { 
                    success = false, 
                    message = "Có lỗi xảy ra khi phân tích chi tiết: " + ex.Message 
                });
            }
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
