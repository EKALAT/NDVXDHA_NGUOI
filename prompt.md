🧠 Prompt: Hệ thống đếm số lượng người vào – ra theo thời gian thực

Prompt tiếng Việt:

Hãy xây dựng một hệ thống nhận diện và đếm số lượng người theo thời gian thực bằng mô hình YOLOv8.

Hệ thống sử dụng camera để nhận diện các đối tượng là “person” và hiển thị khung bao quanh từng người.

Trên giao diện video, hiển thị hai đường đếm:

Một đường IN (vào) – thể hiện khi người di chuyển qua từ dưới lên (ví dụ đi vào khu vực).

Một đường OUT (ra) – thể hiện khi người di chuyển qua từ trên xuống (ra khỏi khu vực).

Mỗi khi một người vượt qua vạch IN hoặc OUT, hệ thống sẽ tăng bộ đếm tương ứng (số người vào / ra).

Các tính năng chính cần có:

Phát hiện người trong thời gian thực bằng mô hình YOLOv8 (class = “person”).

Xác định vị trí của từng người trong khung hình bằng bounding box và tâm đối tượng (centroid).

Sử dụng kỹ thuật tracking ID để theo dõi người, tránh đếm trùng (ví dụ dùng DeepSORT hoặc ByteTrack).

Thiết lập hai đường kẻ (ROI lines): IN và OUT, có thể đặt ở giữa khung hình hoặc cửa ra vào.

Khi centroid của người di chuyển cắt qua vạch IN hoặc OUT, hệ thống sẽ cập nhật bộ đếm.

Hiển thị kết quả lên video:

Số lượng người đang có trong khu vực = IN - OUT

Số người đã vào (IN)

Số người đã ra (OUT)

Giao diện trực quan, có nhãn “IN” và “OUT” rõ ràng (màu xanh / đỏ).

Mục tiêu:

Tạo hệ thống có thể ứng dụng trong giám sát an ninh, kiểm soát lượng người tại cửa ra vào, lớp học, siêu thị hoặc khu vực công cộng.

Có thể mở rộng sang thống kê theo giờ, lưu dữ liệu vào file CSV hoặc cơ sở dữ liệu.

Công nghệ sử dụng:

Python

OpenCV (xử lý video real-time, vẽ vạch và bounding box)

Ultralytics YOLOv8 (mô hình pretrained)

DeepSORT hoặc ByteTrack (tracking đối tượng)

Kết quả đầu ra mong muốn:

Hiển thị video real-time với bounding box quanh người.

Hai vạch IN/OUT hiện rõ trên video.

Hiển thị bộ đếm: “IN: x người, OUT: y người, Total: (IN - OUT) người trong khu vực”.