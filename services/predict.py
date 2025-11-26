import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# GPU 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GaitLSTM(nn.Module):
    """보행 분류 LSTM 모델"""
    def __init__(self, input_size=132, hidden_size=64, num_layers=2, 
                 dropout=0.5, bidirectional=False):
        super(GaitLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(lstm_output_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_n = h_n[-1]
        
        out = self.dropout(h_n)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out


class GaitPredictor:
    def __init__(self, model_path='models/best_model.pt', target_frames=400):
        print("="*60)
        print("🚀 보행 분석 시스템 초기화")
        print("="*60)
        
        # 모델 로드
        print(f"모델 로드 중: {model_path}")
        self.model = GaitLSTM(
            input_size=132,
            hidden_size=64,
            num_layers=2,
            dropout=0.5,
            bidirectional=False
        ).to(device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        self.target_frames = target_frames
        
        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("초기화 완료!")
    
    # 영상에서 키포인트 추출
    def extract_keypoints_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        keypoints_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility
                    ])
                keypoints_list.append(landmarks)
            else:
                keypoints_list.append([[0, 0, 0, 0]] * 33)
        
        cap.release()
        return np.array(keypoints_list)
    
    # 키포인트 전처리
    def preprocess_keypoints(self, keypoints):
        # 1. 사람 등장 시점부터 자르기
        avg_visibility = np.mean(keypoints[:, :, 3], axis=1)
        valid_frames = np.where(avg_visibility > 0.3)[0]
        
        if len(valid_frames) > 0:
            start_idx = valid_frames[0]
            end_idx = valid_frames[-1] + 1
            trimmed = keypoints[start_idx:end_idx]
        else:
            trimmed = keypoints
        
        # 2. 리샘플링 (400 프레임으로)
        resampled = self._resample_frames(trimmed, self.target_frames)
        
        # 3. Tensor 변환: (400, 33, 4) → (400, 132) → (1, 400, 132)
        flattened = resampled.reshape(resampled.shape[0], -1)
        tensor = torch.FloatTensor(flattened).unsqueeze(0).to(device)
        
        return tensor
    
    def _resample_frames(self, keypoints, target_frames):
        """프레임 리샘플링"""
        current_frames = len(keypoints)
        
        if current_frames == target_frames:
            return keypoints
        
        indices = np.linspace(0, current_frames - 1, target_frames)
        resampled = np.zeros((target_frames, keypoints.shape[1], keypoints.shape[2]))
        
        for kp_idx in range(keypoints.shape[1]):
            for coord_idx in range(keypoints.shape[2]):
                resampled[:, kp_idx, coord_idx] = np.interp(
                    indices,
                    np.arange(current_frames),
                    keypoints[:, kp_idx, coord_idx]
                )
        
        return resampled
    
    def predict(self, video_path):
        """영상 예측"""
        print(f"\n🔍 분석 중: {video_path}")
        
        # 키포인트 추출
        print("1. 키포인트 추출 중...")
        keypoints = self.extract_keypoints_from_video(video_path)
        print(f" -> {keypoints.shape[0]} 프레임 추출")
        
        # 디버그 정보
        avg_vis = np.mean(keypoints[:, :, 3])
        print(f" -> 평균 visibility: {avg_vis:.4f}")
        non_zero_frames = np.sum(np.mean(keypoints[:, :, 3], axis=1) > 0.3)
        print(f" -> 유효 프레임 수: {non_zero_frames}/{keypoints.shape[0]}")
        
        # 전처리
        print("2. 전처리 중...")
        processed = self.preprocess_keypoints(keypoints)
        print(f" -> Shape: {processed.shape}")
        
        # 디버그
        non_zero_ratio = (processed[0] != 0).float().mean().item()
        print(f" ->  Non-zero 비율: {non_zero_ratio:.4f}")
        print(f" ->  Min: {processed.min().item():.4f}, Max: {processed.max().item():.4f}")
        
        # 예측
        print("3. 예측 중...")
        with torch.no_grad():
            prediction_prob = self.model(processed)[0][0].item()
        
        print(f" -> Raw output: {prediction_prob:.6f}")
        
        prediction = 1 if prediction_prob > 0.5 else 0
        
        return prediction, prediction_prob
    
    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()


def main():
    """메인 실행"""
    print("보행 분석 시스템 시작")
    
    # 모델 경로 확인
    model_path = 'models/best_model.pt'
    if not Path(model_path).exists():
        print(f"모델을 찾을 수 없습니다: {model_path}")
        return
    
    # 예측기 생성
    predictor = GaitPredictor(model_path=model_path, target_frames=400)
    
    # 영상 경로 입력
    video_path = input("\n분석할 영상 경로를 입력하세요: ").strip()
    
    if not Path(video_path).exists():
        print(f"영상을 찾을 수 없습니다: {video_path}")
        return
    
    # 예측
    try:
        prediction, confidence = predictor.predict(video_path)
        
        # 결과 출력
        print("\n" + "="*60)
        print("분석 결과")
        print("="*60)
        
        if prediction == 0:
            result = "정상 보행 (Normal Gait)"
        else:
            result = "비정상 보행 (Abnormal Gait)"
        
        print(f"\n예측: {result}")
        print(f"확률: {confidence:.2%}")
        print(f"   - 정상: {(1-confidence)*100:.1f}%")
        print(f"   - 비정상: {confidence*100:.1f}%")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n프로그램 종료")


if __name__ == "__main__":
    main()