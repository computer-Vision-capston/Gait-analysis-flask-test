import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
import json

class KeypointExtractor:
    def __init__(self):
        """MediaPipe Pose 초기화"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # 0, 1, 2 (2가 가장 정확)
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_from_video(self, video_path):
        """
        영상에서 키포인트 추출
        
        Args:
            video_path: 영상 파일 경로
            
        Returns:
            keypoints: (num_frames, 33, 4) numpy array
                      4 = (x, y, z, visibility)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"영상을 열 수 없습니다: {video_path}")
        
        keypoints_list = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe 처리
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # 33개 랜드마크 추출
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
                # 감지 실패 시 0으로 채움
                keypoints_list.append([[0, 0, 0, 0]] * 33)
            
            frame_count += 1
        
        cap.release()
        
        # numpy array로 변환
        keypoints = np.array(keypoints_list)  # (num_frames, 33, 4)
        
        print(f"  └ 추출 완료: {frame_count} 프레임, shape={keypoints.shape}")
        
        return keypoints
    
    def process_dataset(self, data_dir, output_dir):
        """
        데이터셋 전체 처리
        
        Args:
            data_dir: 영상 데이터 폴더 (data/)
            output_dir: 키포인트 저장 폴더 (processed_data/)
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 클래스별 폴더
        classes = ['normal', 'spacia']
        
        all_info = []
        
        for class_name in classes:
            class_dir = data_path / class_name
            
            if not class_dir.exists():
                print(f"⚠️  폴더를 찾을 수 없습니다: {class_dir}")
                continue
            
            # 출력 폴더 생성
            output_class_dir = output_path / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            # 영상 파일 목록
            video_files = list(class_dir.glob('*.mp4')) + \
                         list(class_dir.glob('*.avi')) + \
                         list(class_dir.glob('*.mov'))
            
            print(f"\n{'='*60}")
            print(f"클래스: {class_name} ({len(video_files)}개 영상)")
            print(f"{'='*60}")
            
            for i, video_file in enumerate(video_files, 1):
                print(f"[{i}/{len(video_files)}] {video_file.name}")
                
                try:
                    # 키포인트 추출
                    keypoints = self.extract_from_video(str(video_file))
                    
                    # 저장
                    output_file = output_class_dir / f"{video_file.stem}.npy"
                    np.save(output_file, keypoints)
                    
                    # 정보 저장
                    all_info.append({
                        'filename': video_file.name,
                        'class': class_name,
                        'num_frames': len(keypoints),
                        'keypoint_file': str(output_file)
                    })
                    
                except Exception as e:
                    print(f"  └ ❌ 오류 발생: {e}")
                    continue
        
        # 전체 정보 JSON으로 저장
        info_file = output_path / 'dataset_info.json'
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(all_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"✅ 완료! 총 {len(all_info)}개 영상 처리")
        print(f"   저장 위치: {output_path}")
        print(f"   정보 파일: {info_file}")
        print(f"{'='*60}")
        
        return all_info
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'pose'):
            self.pose.close()


def main():
    """메인 실행"""
    # 경로 설정
    DATA_DIR = 'data'  # 원본 영상 폴더
    OUTPUT_DIR = 'processed_data/raw_keypoints'  # 키포인트 저장 폴더
    
    print("🎬 키포인트 추출 시작")
    print(f"입력: {DATA_DIR}")
    print(f"출력: {OUTPUT_DIR}\n")
    
    # 추출기 생성 및 실행
    extractor = KeypointExtractor()
    info = extractor.process_dataset(DATA_DIR, OUTPUT_DIR)
    
    # 간단한 통계
    if info:
        frame_counts = [item['num_frames'] for item in info]
        print(f"\n📊 프레임 수 통계:")
        print(f"   최소: {min(frame_counts)} 프레임")
        print(f"   최대: {max(frame_counts)} 프레임")
        print(f"   평균: {np.mean(frame_counts):.1f} 프레임")
        print(f"   중간값: {np.median(frame_counts):.1f} 프레임")


if __name__ == "__main__":
    main()