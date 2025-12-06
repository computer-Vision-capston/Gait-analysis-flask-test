# Human Picture - 독거노인 낙상 감지 및 보행 분석 시스템

AI 기반 저비용 원격 모니터링 시스템으로, 낙상 감지와 보행 이상 분석 기능을 제공합니다.

## 프로젝트 개요

- **목표**: 독거노인의 낙상 자동 감지 및 보행 이상 분석
- **핵심 기술**: MediaPipe Pose, LSTM, 규칙 기반 알고리즘
- **하드웨어**: 라즈베리파이 4, USB 웹캠, 아두이노, PIR 센서
- **예상 비용**: 약 15만원 (기존 상용 장비 대비 1/3 수준)

## 시스템 구조

```
[PIR 센서 감지] → [영상 녹화] → [키포인트 추출] → [낙상 판단] → [보행 분류] → [결과 저장]
```

## 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/your-username/human-picture.git
cd human-picture
```

### 2. 가상환경 생성 (권장)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. PyTorch 설치 (GPU 사용 시)

```bash
# CUDA 11.8 기준
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 폴더 구조

```
human-picture/
├── data/
│   ├── normal/              # 정상 보행 영상
│   └── special/             # 비정상 보행 영상
├── processed_data/
│   ├── raw_keypoints/       # 추출된 키포인트
│   └── preprocessed/        # 전처리된 데이터
├── models/                  # 학습된 모델
├── utils/
│   ├── extract_keypoints.py # 키포인트 추출
│   ├── preprocess.py        # 데이터 전처리
│   └── train.py             # 모델 학습
├── predict.py               # 추론
├── README.md
└── requirements.txt
```

## 실행 방법

### 1단계: 키포인트 추출

영상 데이터에서 MediaPipe를 사용하여 신체 키포인트를 추출합니다.

```bash
python utils/extract_keypoints.py
```

**입력**: `data/normal/`, `data/special/` 폴더의 영상 파일 (.mp4, .avi, .mov)

**출력**: `processed_data/raw_keypoints/` 폴더에 .npy 파일 저장

### 2단계: 데이터 전처리

추출된 키포인트를 학습에 적합한 형태로 전처리합니다.

```bash
python utils/preprocess.py
```

**전처리 과정**:
- 사람 등장 구간 추출 (visibility > 0.3)
- 400 프레임으로 리샘플링
- 데이터 증강 (4배)
- Train/Val 분할 (8:2)

**출력**: `processed_data/preprocessed/` 폴더에 X_train.npy, X_val.npy, y_train.npy, y_val.npy 저장

### 3단계: 모델 학습

LSTM 모델을 학습합니다.

```bash
python utils/train.py
```

**학습 설정**:
- Batch Size: 8
- Epochs: 80 (Early Stopping: 20)
- Learning Rate: 0.001
- Optimizer: Adam

**출력**: `models/` 폴더에 best_model.pt, final_model.pt 저장

### 4단계: 추론

학습된 모델로 새로운 영상을 분석합니다.

```bash
python predict.py --video path/to/video.mp4
```

## 모델 구조

### LSTM 보행 분류 모델

| Layer | 설명 |
|-------|------|
| Input | (400, 132) - 400프레임 × 33키포인트 × 4값 |
| LSTM | hidden_size=64, num_layers=2, dropout=0.5 |
| FC1 | 64 → 32, ReLU |
| FC2 | 32 → 1, Sigmoid |

### 규칙 기반 낙상 감지 알고리즘

1. **1단계**: 급격한 하강 감지 (임계값: 0.03)
2. **2단계**: 큰 높이 변화 확인 (임계값: 0.15)
3. **3단계**: 최종 자세 검증 (5가지 조건 중 2개 이상 충족)

## 데이터셋

- **출처**: [Mendeley Data](https://data.mendeley.com/datasets/44pfnysy89/1)
- **구성**: 정상 보행 + 비정상 보행 (관절염, 파킨슨병)
- **총 개수**: 약 155개 영상

## 라이선스

This project is licensed under the MIT License.

## 참고문헌

- [Mendeley Gait Dataset](https://data.mendeley.com/datasets/44pfnysy89/1)
- [IEEE Rule-based Fall Detection](https://ieeexplore.ieee.org/document/10543522)
- [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
