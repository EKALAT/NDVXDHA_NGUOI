import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
from datetime import datetime
import os

class PersonCounter:
    def __init__(self, model_path='yolov8n.pt', video_source=0):
        """
        Khởi tạo hệ thống đếm người
        
        Args:
            model_path: Đường dẫn đến file mô hình YOLOv8 (.pt)
            video_source: Nguồn video (0 cho webcam, hoặc đường dẫn file video)
        """
        self.model = YOLO(model_path)
        self.video_source = video_source
        self.cap = None
        
        # Bộ đếm
        self.in_count = 0
        self.out_count = 0
        
        # Tracking dictionary để lưu trữ ID và vị trí trước đó của từng người
        self.tracks = {}  # {track_id: {'centroid': (x, y), 'prev_x': x, 'last_in': bool, 'last_out': bool, 'direction': None}}
        self.next_id = 0
        
        # Vị trí các đường đếm dọc (có thể điều chỉnh)
        self.line_in_x = None  # Đường IN (từ trái sang phải - đi vào)
        self.line_out_x = None  # Đường OUT (từ phải sang trái - đi ra)
        
        # Màu sắc
        self.color_in = (0, 255, 0)  # Xanh lá (IN)
        self.color_out = (0, 0, 255)  # Đỏ (OUT)
        self.color_box_left_right = (255, 0, 255)  # Tím (đi từ trái sang phải)
        self.color_box_right_left = (0, 165, 255)  # Cam (đi từ phải sang trái)
        self.color_box_default = (255, 0, 0)  # Xanh dương (mặc định)
        
        # Video recording
        self.video_writer = None
        self.is_recording = False
        self.output_filename = None
        
    def setup_lines(self, frame_width):
        """
        Thiết lập vị trí các đường đếm IN và OUT (dọc)
        - OUT: Ở 1/5 từ trái (bên trái), đếm khi người đi từ phải sang trái qua vạch OUT (ra khỏi siêu thị)
        - IN: Ở 0.7 từ trái (bên phải), đếm khi người đi từ trái sang phải qua vạch IN (vào siêu thị)
        """
        if self.line_out_x is None:
            # Đường OUT ở 1/5 từ trái (bên trái) - người đi ra từ phải sang trái
            self.line_out_x = int(frame_width * 1/5)
        if self.line_in_x is None:
            # Đường IN ở 0.7 từ trái (bên phải) - người đi vào từ trái sang phải
            self.line_in_x = int(frame_width * 0.7)
    
    def calculate_centroid(self, bbox):
        """
        Tính toán tâm của bounding box
        
        Args:
            bbox: [x1, y1, x2, y2]
        Returns:
            (x, y): Tọa độ centroid
        """
        x1, y1, x2, y2 = bbox
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)
        return (x, y)
    
    def update_tracks(self, results):
        """
        Cập nhật tracking và đếm người
        
        Args:
            results: Kết quả từ YOLO model (Results object)
        """
        current_centroids = {}
        
        # Lấy tất cả các detection là "person"
        boxes = results.boxes
        
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].cpu().numpy())
                
                # Class 0 là "person" trong COCO dataset
                if cls == 0:
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    
                    if conf > 0.6:  # Tăng ngưỡng confidence để bắt người chuẩn hơn
                        centroid = self.calculate_centroid(bbox)
                        current_centroids[i] = {
                            'centroid': centroid,
                            'bbox': bbox,
                            'conf': conf
                        }
        
        # Gán ID cho các detection mới dựa trên khoảng cách và IoU
        matched_ids = {}
        used_ids = set()
        
        # Tính IoU (Intersection over Union) giữa 2 bounding box
        def calculate_iou(bbox1, bbox2):
            x1_min, y1_min, x1_max, y1_max = bbox1
            x2_min, y2_min, x2_max, y2_max = bbox2
            
            # Tính diện tích intersection
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0
            
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            
            # Tính diện tích union
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = area1 + area2 - inter_area
            
            if union_area == 0:
                return 0.0
            
            return inter_area / union_area
        
        for new_id, new_data in current_centroids.items():
            best_match_id = None
            best_score = -1
            
            for old_id, old_data in self.tracks.items():
                if old_id in used_ids:
                    continue
                
                old_centroid = old_data['centroid']
                old_bbox = old_data['bbox']
                new_centroid = new_data['centroid']
                new_bbox = new_data['bbox']
                
                # Tính khoảng cách Euclidean
                distance = np.sqrt(
                    (old_centroid[0] - new_centroid[0])**2 + 
                    (old_centroid[1] - new_centroid[1])**2
                )
                
                # Tính IoU
                iou = calculate_iou(old_bbox, new_bbox)
                
                # Tính điểm số kết hợp (ưu tiên IoU cao và khoảng cách gần)
                # Nếu IoU > 0.3 hoặc khoảng cách < 150 pixels
                max_distance = 150  # Tăng ngưỡng để tracking tốt hơn
                
                if iou > 0.3:
                    # Nếu IoU cao, ưu tiên IoU
                    score = iou * 2
                elif distance < max_distance:
                    # Nếu khoảng cách gần, tính điểm dựa trên khoảng cách
                    score = (max_distance - distance) / max_distance
                else:
                    score = 0
                
                if score > best_score:
                    best_score = score
                    best_match_id = old_id
            
            if best_match_id is not None and best_score > 0.1:  # Ngưỡng tối thiểu
                matched_ids[new_id] = best_match_id
                used_ids.add(best_match_id)
            else:
                # Tạo ID mới
                matched_ids[new_id] = self.next_id
                self.next_id += 1
        
        # Cập nhật tracks và kiểm tra đường đếm
        updated_tracks = {}
        
        for new_id, new_data in current_centroids.items():
            track_id = matched_ids.get(new_id, new_id)
            centroid = new_data['centroid']
            bbox = new_data['bbox']
            conf = new_data['conf']
            
            prev_x = None
            last_in = False
            last_out = False
            direction = None
            
            if track_id in self.tracks:
                prev_data = self.tracks[track_id]
                prev_x = prev_data['prev_x']
                last_in = prev_data.get('last_in', False)
                last_out = prev_data.get('last_out', False)
                direction = prev_data.get('direction', None)
            
            # Kiểm tra vượt qua đường đếm (dọc)
            if prev_x is not None:
                current_x = centroid[0]
                
                # Tính khoảng cách di chuyển để đảm bảo là di chuyển thực sự
                movement_distance = abs(current_x - prev_x)
                
                # Kiểm tra đường IN (từ trái sang phải - đi vào siêu thị)
                # IN ở bên phải, đếm khi người đi từ trái sang phải qua vạch IN
                if prev_x < self.line_in_x and current_x >= self.line_in_x:
                    # Chỉ đếm nếu di chuyển đủ xa (tránh nhảy ID) và chưa đi vào hoặc đã đi ra
                    if movement_distance > 5 and (not last_in or last_out):
                        self.in_count += 1
                        direction = 'IN'
                        print(f"Người {track_id} đi vào! IN: {self.in_count}")
                    last_in = True
                    last_out = False
                
                # Kiểm tra đường OUT (từ phải sang trái - đi ra khỏi siêu thị)
                # OUT ở bên trái, đếm khi người đi từ phải sang trái qua vạch OUT
                if prev_x > self.line_out_x and current_x <= self.line_out_x:
                    # Đếm nếu di chuyển đủ xa và chưa đi ra (để tránh đếm trùng)
                    if movement_distance > 5 and not last_out:
                        self.out_count += 1
                        direction = 'OUT'
                        print(f"Người {track_id} đi ra! OUT: {self.out_count}")
                    last_out = True
                    last_in = False  # Sau khi đi ra, không còn trong khu vực
            
            # Xác định hướng di chuyển dựa trên vị trí trước và hiện tại
            movement_direction = None
            if prev_x is not None:
                if current_x > prev_x:
                    movement_direction = 'left_to_right'  # Trái sang phải
                elif current_x < prev_x:
                    movement_direction = 'right_to_left'  # Phải sang trái
                else:
                    # Giữ nguyên hướng di chuyển trước đó nếu có
                    if track_id in self.tracks:
                        old_movement = self.tracks[track_id].get('movement_direction', None)
                        movement_direction = old_movement
            
            updated_tracks[track_id] = {
                'centroid': centroid,
                'bbox': bbox,
                'prev_x': centroid[0],
                'last_in': last_in,
                'last_out': last_out,
                'direction': direction,
                'movement_direction': movement_direction,
                'conf': conf
            }
        
        self.tracks = updated_tracks
    
    def draw_lines(self, frame):
        """
        Vẽ các đường đếm IN và OUT lên frame (đường dọc)
        """
        h, w = frame.shape[:2]
        
        # Vẽ đường OUT (màu đỏ) - dọc, ở bên trái
        cv2.line(frame, (self.line_out_x, 0), (self.line_out_x, h), 
                 self.color_out, 3)
        cv2.putText(frame, "OUT", (self.line_out_x + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.color_out, 2)
        
        # Vẽ đường IN (màu xanh lá) - dọc, ở bên phải
        cv2.line(frame, (self.line_in_x, 0), (self.line_in_x, h), 
                 self.color_in, 3)
        cv2.putText(frame, "IN", (self.line_in_x + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.color_in, 2)
    
    def draw_detections(self, frame):
        """
        Vẽ bounding boxes và thông tin tracking lên frame
        Màu sắc bounding box dựa trên hướng di chuyển
        """
        for track_id, track_data in self.tracks.items():
            bbox = track_data['bbox']
            centroid = track_data['centroid']
            conf = track_data['conf']
            direction = track_data.get('direction', None)
            movement_direction = track_data.get('movement_direction', None)
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Chọn màu bounding box dựa trên hướng di chuyển
            if movement_direction == 'left_to_right':
                box_color = self.color_box_left_right  # Tím - trái sang phải
            elif movement_direction == 'right_to_left':
                box_color = self.color_box_right_left  # Cam - phải sang trái
            else:
                box_color = self.color_box_default  # Xanh dương - mặc định
            
            # Vẽ bounding box với màu theo hướng di chuyển
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            
            # Vẽ centroid
            cv2.circle(frame, centroid, 5, (0, 255, 255), -1)
            
            # Vẽ ID và confidence
            label = f"ID:{track_id} {conf:.2f}"
            if direction:
                label += f" [{direction}]"
            
            # Background cho text để dễ đọc
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 0, 0), -1)
            
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def draw_stats(self, frame):
        """
        Vẽ thống kê đếm lên frame (góc trên bên trái)
        """
        total_inside = self.in_count - self.out_count
        h, w = frame.shape[:2]
        
        # Tính toán vị trí góc trái trên
        box_width = 350
        box_height = 140
        x_start = 10
        y_start = 10
        
        # Vẽ background đen đậm với border trắng
        cv2.rectangle(frame, (x_start, y_start), (x_start + box_width, y_start + box_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (x_start, y_start), (x_start + box_width, y_start + box_height), (255, 255, 255), 2)
        
        # Vẽ text thống kê
        text_x = x_start + 15
        text_y = y_start + 35
        font_scale = 0.9
        thickness = 2
        
        # Vẽ tiêu đề
        cv2.putText(frame, "THONG KE", (text_x, text_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Vẽ IN với màu xanh lá
        cv2.putText(frame, f"IN: {self.in_count} nguoi", (text_x, text_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.color_in, thickness)
        
        # Vẽ OUT với màu đỏ
        cv2.putText(frame, f"OUT: {self.out_count} nguoi", (text_x, text_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.color_out, thickness)
        
        # Vẽ số người trong khu vực
        cv2.putText(frame, f"Trong khu vuc: {max(0, total_inside)}", (text_x, text_y + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)
        
        # Hiển thị trạng thái recording
        if self.is_recording:
            cv2.putText(frame, "REC", (text_x + 280, text_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # Vẽ chấm đỏ
            cv2.circle(frame, (text_x + 320, text_y - 5), 8, (0, 0, 255), -1)
    
    def run(self):
        """
        Chạy hệ thống đếm người
        """
        self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            print(f"Khong the mo video source: {self.video_source}")
            return
        
        # Đọc frame đầu tiên để lấy kích thước
        ret, frame = self.cap.read()
        if not ret:
            print("Khong the doc frame tu video")
            return
        
        self.setup_lines(frame.shape[1])  # frame.shape[1] là width cho đường dọc
        
        # Lấy thông tin video để tạo VideoWriter
        fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("Bat dau he thong dem nguoi...")
        print("Nhan 'q' de thoat")
        print("Nhan 'r' de reset bo dem")
        print("Nhan 's' de bat/tat ghi video")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Phát hiện đối tượng với YOLOv8
            results = self.model(frame, verbose=False)
            
            # Cập nhật tracking và đếm
            if len(results) > 0:
                self.update_tracks(results[0])
            
            # Vẽ các thành phần lên frame
            self.draw_lines(frame)
            self.draw_detections(frame)
            self.draw_stats(frame)
            
            # Ghi video nếu đang recording
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(frame)
            
            # Hiển thị frame
            cv2.imshow('He thong dem nguoi - IN/OUT Counter', frame)
            
            # Xử lý phím nhấn
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.in_count = 0
                self.out_count = 0
                self.tracks = {}
                print("Da reset bo dem!")
            elif key == ord('s'):
                self.toggle_recording(fps, width, height)
        
        # Dừng recording nếu đang ghi
        if self.is_recording:
            self.stop_recording()
        
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\nKet qua cuoi cung:")
        print(f"IN: {self.in_count} nguoi")
        print(f"OUT: {self.out_count} nguoi")
        print(f"Trong khu vuc: {max(0, self.in_count - self.out_count)} nguoi")
    
    def toggle_recording(self, fps, width, height):
        """
        Bật/tắt ghi video
        """
        if not self.is_recording:
            # Bắt đầu ghi
            self.start_recording(fps, width, height)
        else:
            # Dừng ghi
            self.stop_recording()
    
    def start_recording(self, fps, width, height):
        """
        Bắt đầu ghi video
        """
        if self.is_recording:
            return
        
        # Tạo tên file với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_filename = f"output_video_{timestamp}.mp4"
        
        # Tạo thư mục output nếu chưa có
        os.makedirs("output_videos", exist_ok=True)
        output_path = os.path.join("output_videos", self.output_filename)
        
        # Tạo VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.is_recording = True
        print(f"Bat dau ghi video: {output_path}")
    
    def stop_recording(self):
        """
        Dừng ghi video
        """
        if not self.is_recording:
            return
        
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        self.is_recording = False
        print(f"Da luu video: {self.output_filename}")
        self.output_filename = None


def main():
    """
    Hàm main để chạy hệ thống
    """
    print("=" * 50)
    print("HE THONG DEM NGUOI VAO/RA")
    print("=" * 50)
    print("\nChon nguon video:")
    print("1. Camera (Webcam)")
    print("2. File video")
    
    while True:
        choice = input("\nNhap lua chon (1 hoac 2): ").strip()
        
        if choice == '1':
            video_source = 0
            print("Da chon Camera!")
            break
        elif choice == '2':
            video_path = input("Nhap duong dan file video: ").strip()
            # Loại bỏ dấu ngoặc kép nếu có
            video_path = video_path.strip('"').strip("'")
            video_source = video_path
            print(f"Da chon file video: {video_path}")
            break
        else:
            print("Lua chon khong hop le! Vui long nhap 1 hoac 2.")
    
    # Đường dẫn đến file mô hình YOLOv8
    model_path = 'yolov8n.pt'
    
    print("\nDang khoi dong he thong...")
    
    # Tạo và chạy hệ thống
    counter = PersonCounter(model_path=model_path, video_source=video_source)
    counter.run()


if __name__ == '__main__':
    main()

