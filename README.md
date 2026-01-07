# Face Recognition Project

Dự án nhận diện khuôn mặt thời gian thực sử dụng webcam, dựa trên mô hình FaceNet và YOLO để phát hiện khuôn mặt.

## Mô tả

Dự án này bao gồm:
- Phát hiện khuôn mặt bằng YOLOv8
- Trích xuất đặc trưng khuôn mặt bằng Inception-ResNet V2 (FaceNet)
- Nhận diện khuôn mặt bằng cách so sánh embedding với cơ sở dữ liệu
- Giao diện thời gian thực qua webcam

## Yêu cầu hệ thống

- Python 3.8 trở lên
- Webcam (để chạy nhận diện thời gian thực)
- Hệ điều hành: Windows/Linux/MacOS

## Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd Face_Recog_Final
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv face_recog_env
# Trên Windows:
face_recog_env\Scripts\activate
# Trên Linux/Mac:
source face_recog_env/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install tensorflow opencv-python ultralytics pandas numpy scikit-learn scipy
```

### 4. Tải trọng số mô hình

Đảm bảo các file trọng số sau có trong thư mục `Weight/`:
- `facenet_keras_weights.h5`: Trọng số cho mô hình FaceNet
- `yolov8n-face.pt`: Trọng số cho mô hình YOLO face detection

## Chuẩn bị dữ liệu

### Cấu trúc dữ liệu

Tạo thư mục `Data/` với cấu trúc như sau:

```
Data/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── person2/
│   ├── image1.jpg
│   └── ...
└── ...
```

Mỗi thư mục con đại diện cho một người, chứa các ảnh khuôn mặt của họ.

### Tạo embeddings

Chạy script tạo embeddings từ dataset:

```bash
python create_embeds.py
```

Script này sẽ:
- Đọc ảnh từ thư mục `Data/`
- Phát hiện khuôn mặt
- Trích xuất embedding
- Lưu vào file `embedding_with_label.csv`

## Chạy dự án

### Nhận diện thời gian thực

Sau khi có file `embedding_with_label.csv`, chạy:

```bash
python main.py
```

- Webcam sẽ mở ra
- Hiển thị khung hình với khuôn mặt được phát hiện và nhận diện
- Nhấn 'q' để thoát

## Cách sử dụng

### Thêm người mới

Chạy script để chụp ảnh cho người mới:

```bash
python add_person.py
```

- Nhập tên người mới
- Webcam sẽ mở ra
- Nhấn SPACE để chụp ảnh (nên chụp 10-20 ảnh từ các góc khác nhau)
- Nhấn 'q' để thoát
- Ảnh sẽ được lưu tự động vào `Data/tên_người/`
- Sau đó chạy `python create_embeds.py` để cập nhật cơ sở dữ liệu

### Tùy chỉnh

- Ngưỡng nhận diện: Sửa trong `models/facenet.py` (hiện tại là 0.3)
- Độ tin cậy phát hiện: Sửa trong `utils/detect_face.py` (hiện tại là 0.4)

## Cấu trúc dự án

```
Face_Recog_Final/
├── main.py                 # Script chính cho nhận diện thời gian thực
├── create_embeds.py        # Script tạo embeddings từ dataset
├── add_person.py           # Script thêm người mới bằng webcam
├── embedding_with_label.csv # File chứa embeddings và nhãn
├── Data/                   # Thư mục chứa dataset ảnh
├── Weight/                 # Thư mục chứa trọng số mô hình
├── models/
│   ├── facenet.py          # Model facenet
│   └── inception_resnet.py # Model inception_resnetV2
├── tools/
│   └── detect_recog.py     # Service phát hiện và nhận diện
└── utils/
    ├── detect_face.py      # Utility phát hiện khuôn mặt
    └── embed.py            # Utility trích xuất embedding
```

## Lưu ý

- Đảm bảo webcam không bị chiếm bởi ứng dụng khác
- Ảnh trong dataset nên có độ phân giải cao và khuôn mặt rõ ràng
- Nếu gặp lỗi import, kiểm tra lại việc cài đặt dependencies
- Dự án sử dụng GPU nếu có, nếu không sẽ chạy trên CPU

## Troubleshooting

### Lỗi "Can't open webcam"
- Kiểm tra webcam có được kết nối và không bị sử dụng bởi app khác
- Thay đổi index camera trong `main.py`: `cv2.VideoCapture(1)` hoặc `cv2.VideoCapture(2)`

### Lỗi import module
- Đảm bảo đã activate môi trường ảo
- Cài lại dependencies: `pip install -r requirements.txt` (nếu có file requirements.txt)

### Không phát hiện khuôn mặt
- Kiểm tra ánh sáng và góc chụp
- Giảm ngưỡng confidence trong `detect_face.py`
