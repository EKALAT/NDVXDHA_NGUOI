# Hệ thống đếm số lượng người vào - ra theo thời gian thực

Hệ thống nhận diện và đếm số lượng người vào/ra theo thời gian thực sử dụng mô hình YOLOv8.

## Tính năng

- ✅ Phát hiện người trong thời gian thực bằng YOLOv8
- ✅ Tracking đối tượng để tránh đếm trùng
- ✅ Hai đường đếm: IN (vào) và OUT (ra)
- ✅ Hiển thị bounding box, ID tracking và thống kê
- ✅ Giao diện trực quan với màu sắc rõ ràng

## Yêu cầu hệ thống

- Python 3.8 trở lên
- Webcam hoặc file video để test

## Cài đặt

1. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

2. Đảm bảo file `yolov8n.pt` có trong thư mục dự án (đã có sẵn)

## Sử dụng

### Chạy với webcam:

```bash
python person_counter.py
```

### Chạy với file video:

Khi chạy chương trình, chọn option `2` và nhập đường dẫn file video.

## Hướng dẫn sử dụng

1. **Khởi động chương trình**: Chạy `person_counter.py`
2. **Chọn nguồn video**: 
   - Nhập `1` để sử dụng Camera (Webcam)
   - Nhập `2` để sử dụng file video (sau đó nhập đường dẫn file)
3. **Điều chỉnh vị trí đường đếm**: 
   - Đường OUT (màu đỏ): Mặc định ở 1/3 chiều rộng (bên trái) - đi ra khỏi siêu thị (từ phải sang trái)
   - Đường IN (màu xanh lá): Mặc định ở 2/3 chiều rộng (bên phải) - đi vào siêu thị (từ trái sang phải)
   - Có thể chỉnh sửa trong hàm `setup_lines()` trong code
   - **Lưu ý**: Hệ thống sử dụng đường đếm dọc (vertical lines) phù hợp với camera đặt trên bàn
4. **Các phím điều khiển**:
   - `q`: Thoát chương trình
   - `r`: Reset bộ đếm về 0
   - `s`: Bật/tắt ghi video (video sẽ được lưu vào thư mục `output_videos`)

## Cách hoạt động

1. **Phát hiện người**: YOLOv8 phát hiện tất cả người trong khung hình
2. **Tracking**: Hệ thống gán ID cho từng người và theo dõi di chuyển
3. **Đếm vào/ra**: 
   - Khi centroid của người vượt qua đường IN (từ trái sang phải) → tăng bộ đếm IN (đi vào siêu thị)
   - Khi centroid của người vượt qua đường OUT (từ phải sang trái) → tăng bộ đếm OUT (đi ra khỏi siêu thị)
4. **Hiển thị**: 
   - Số người đang trong khu vực = IN - OUT
   - Bounding box quanh từng người với ID và confidence
   - Thống kê real-time ở góc trên bên trái

## Giao diện

- **Đường OUT**: Màu đỏ (dọc), ở 1/3 từ trái (bên trái), đếm khi người đi từ phải sang trái qua vạch OUT (đi ra khỏi siêu thị)
- **Đường IN**: Màu xanh lá (dọc), ở 2/3 từ trái (bên phải), đếm khi người đi từ trái sang phải qua vạch IN (đi vào siêu thị)
- **Bounding box**: Màu xanh dương, hiển thị ID và confidence
- **Centroid**: Điểm vàng ở giữa bounding box
- **Trạng thái REC**: Hiển thị "REC" với chấm đỏ khi đang ghi video

## Tính năng ghi video

- Nhấn phím `s` để bắt đầu/dừng ghi video
- Video được lưu tự động vào thư mục `output_videos/`
- Tên file video có định dạng: `output_video_YYYYMMDD_HHMMSS.mp4`
- Khi đang ghi, màn hình sẽ hiển thị "REC" với chấm đỏ

## Mở rộng

Hệ thống có thể mở rộng thêm:
- Lưu dữ liệu vào CSV hoặc database
- Thống kê theo giờ/ngày
- Gửi cảnh báo khi vượt quá số lượng cho phép
- Tích hợp với hệ thống giám sát an ninh

## Lưu ý

- Đảm bảo ánh sáng đủ để phát hiện tốt
- Điều chỉnh vị trí đường đếm phù hợp với góc quay camera
- Có thể cần điều chỉnh ngưỡng confidence trong code nếu cần

## Tác giả

Hệ thống được xây dựng theo yêu cầu từ prompt.md

