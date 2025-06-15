# 📊 GIẢI THÍCH: XÁC SUẤT TIÊN NGHIỆM P(C) TRONG NAIVE BAYES

## 🎯 **XÁC SUẤT TIÊN NGHIỆM P(C) LẤY TỪ ĐÂU?**

### 1. **Định nghĩa**
**Xác suất tiên nghiệm P(C)** là xác suất xuất hiện của mỗi lớp (category) trong **tập dữ liệu training**, được tính **TRƯỚC KHI** xem nội dung bài báo.

### 2. **Công thức tính**
```
P(C) = Số lượng bài báo thuộc lớp C / Tổng số bài báo trong tập training
```

### 3. **Ví dụ cụ thể**

Giả sử trong tập dữ liệu training có **1000 bài báo**:

| Lớp (Category) | Số lượng bài | Tính toán | Xác suất tiên nghiệm P(C) |  % |
|----------------|-------------|-----------|---------------------------|-----|
| **Thể thao**   | 200 bài     | 200/1000  | **0.200000**             | 20% |
| **Kinh doanh** | 180 bài     | 180/1000  | **0.180000**             | 18% |
| **Giải trí**   | 150 bài     | 150/1000  | **0.150000**             | 15% |
| **Công nghệ**  | 140 bài     | 140/1000  | **0.140000**             | 14% |
| **Chính trị**  | 130 bài     | 130/1000  | **0.130000**             | 13% |
| **Sức khỏe**   | 120 bài     | 120/1000  | **0.120000**             | 12% |
| **Giáo dục**   | 80 bài      | 80/1000   | **0.080000**             | 8%  |
| **TỔNG**       | **1000**    |           | **1.000000**             | 100% |

### 4. **Code implementation**

```csharp
/// <summary>
/// Tính xác suất tiên nghiệm cho từng lớp
/// </summary>
private void CalculateClassProbabilities(List<NewsArticle> trainingData)
{
    var totalCount = trainingData.Count;  // Tổng số bài báo = 1000
    
    foreach (var className in _model.Classes)
    {
        // Đếm số bài báo của từng lớp
        var classCount = trainingData.Count(x => x.Category == className);
        
        // Tính P(C) = số bài của lớp / tổng số bài
        _model.ClassProbabilities[className] = (double)classCount / totalCount;
    }

    Console.WriteLine("Xác suất tiên nghiệm:");
    foreach (var kvp in _model.ClassProbabilities)
    {
        Console.WriteLine($"  P({kvp.Key}) = {kvp.Value:F6}");
    }
}
```

### 5. **Tại sao gọi là "tiên nghiệm"?**

- 🔍 **"Tiên nghiệm" (A priori)** có nghĩa là "trước khi biết", tức là **trước khi đọc nội dung bài báo**
- 📊 Chỉ dựa vào **thống kê tần suất** của từng lớp trong tập training
- 🎲 Nếu bạn **chọn ngẫu nhiên** một bài báo từ tập training, P(C) cho biết **khả năng** bài đó thuộc lớp C

### 6. **Vai trò trong công thức Naive Bayes**

```
P(C|X) = P(C) × P(X|C)
         ↑      ↑
   Tiên nghiệm  Likelihood
```

- **P(C)**: "Trước khi đọc bài báo, khả năng nó thuộc lớp C là bao nhiêu?"
- **P(X|C)**: "Nếu bài báo thuộc lớp C, khả năng nó có nội dung X là bao nhiêu?"

### 7. **Ví dụ thực tế**

Khi phân loại bài: *"Cầu thủ Messi ghi bàn thắng quyết định"*

1. **Bước 1 - Xác suất tiên nghiệm**:
   - P(Thể thao) = 0.200000 (20%)
   - P(Giải trí) = 0.150000 (15%)
   - P(Kinh doanh) = 0.180000 (18%)
   - ...

2. **Bước 2 - Likelihood**: Tính P("Messi", "cầu thủ", "bàn thắng" | mỗi lớp)

3. **Bước 3 - Kết hợp**: 
   - Log P(Thể thao|X) = Log(0.2) + Log P(từ khóa|Thể thao)
   - Log P(Giải trí|X) = Log(0.15) + Log P(từ khóa|Giải trí)
   - ...

### 8. **Lưu ý quan trọng**

✅ **Xác suất tiên nghiệm được tính từ tập training data**
✅ **Phản ánh tỷ lệ phân bố của các lớp trong dữ liệu**
✅ **Không thay đổi khi phân loại bài báo mới**
✅ **Ảnh hưởng đến kết quả cuối cùng - lớp có P(C) cao có lợi thế ban đầu**

❗ **Chú ý**: Nếu tập training không cân bằng (ví dụ: 70% Thể thao, 5% Giáo dục), model sẽ có xu hướng dự đoán về lớp đông hơn!

### 9. **Xem trong giao diện web**

Khi bạn click **"Xem chi tiết quá trình Naive Bayes"**, bạn sẽ thấy:

```
BƯỚC 1 - Xác suất tiên nghiệm:
P(Thể thao) = 0.200000
Log P(Thể thao) = ln(0.200000) = -1.609438
```

Đây chính là xác suất được tính từ tập training data!

---

## 🔗 **Tóm lại**

**Xác suất tiên nghiệm P(C) = Tần suất xuất hiện của lớp C trong tập dữ liệu training**

Đây là thông tin thống kê **cơ bản** và **không đổi**, được sử dụng làm "điểm khởi đầu" trong quá trình phân loại Naive Bayes!
