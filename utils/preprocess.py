import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle

class GaitDataPreprocessor:
    def __init__(self, raw_data_dir='processed_data/raw_keypoints', 
                 target_frames=400):
        """
        보행 데이터 전처리기
        
        Args:
            raw_data_dir: 원시 키포인트 폴더
            target_frames: 목표 프레임 수 (기본 400)
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.target_frames = target_frames
        
        # 전신 33개 키포인트 모두 사용
        
        # 정보 로드
        info_file = self.raw_data_dir / 'dataset_info.json'
        with open(info_file, 'r', encoding='utf-8') as f:
            self.dataset_info = json.load(f)
    
    def trim_to_person(self, keypoints, visibility_threshold=0.3):
        """
        사람이 등장하는 시점부터 자르기
        
        Args:
            keypoints: (frames, 33, 4) array
            visibility_threshold: 최소 visibility 기준
            
        Returns:
            trimmed_keypoints: 사람이 있는 프레임만
        """
        # 각 프레임의 평균 visibility 계산
        avg_visibility = np.mean(keypoints[:, :, 3], axis=1)
        
        # visibility가 threshold 이상인 프레임 찾기
        valid_frames = np.where(avg_visibility > visibility_threshold)[0]
        
        if len(valid_frames) == 0:
            # 유효한 프레임이 없으면 전체 반환
            return keypoints
        
        # 첫 등장부터 마지막 프레임까지
        start_idx = valid_frames[0]
        end_idx = valid_frames[-1] + 1
        
        return keypoints[start_idx:end_idx]
    
    def resample_frames(self, keypoints, target_frames):
        """
        프레임 수를 target_frames로 리샘플링
        
        Args:
            keypoints: (frames, 33, 4) array
            target_frames: 목표 프레임 수
            
        Returns:
            resampled: (target_frames, 33, 4) array
        """
        current_frames = len(keypoints)
        
        if current_frames == target_frames:
            return keypoints
        
        # 인덱스 생성 (선형 보간)
        indices = np.linspace(0, current_frames - 1, target_frames)
        
        # 각 키포인트별로 보간
        resampled = np.zeros((target_frames, keypoints.shape[1], keypoints.shape[2]))
        
        for kp_idx in range(keypoints.shape[1]):  # 33개 키포인트
            for coord_idx in range(keypoints.shape[2]):  # x, y, z, vis
                resampled[:, kp_idx, coord_idx] = np.interp(
                    indices, 
                    np.arange(current_frames), 
                    keypoints[:, kp_idx, coord_idx]
                )
        
        return resampled
    
    def preprocess_single_video(self, keypoints):
        """
        단일 영상 전처리 파이프라인
        
        Args:
            keypoints: (frames, 33, 4) 원시 키포인트
            
        Returns:
            processed: (target_frames, 33, 4) 전처리된 키포인트
        """
        # 1. 사람 등장 시점부터 자르기
        trimmed = self.trim_to_person(keypoints)
        
        # 2. 리샘플링 (400 프레임으로)
        resampled = self.resample_frames(trimmed, self.target_frames)
        
        return resampled
    
    def process_dataset(self, test_size=0.2, random_state=42):
        """
        전체 데이터셋 전처리
        
        Args:
            test_size: 검증 데이터 비율
            random_state: 랜덤 시드
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        print("="*60)
        print("🔧 데이터 전처리 시작")
        print("="*60)
        
        X_list = []
        y_list = []
        
        # 클래스 레이블 매핑
        class_to_label = {'normal': 0, 'spacia': 1}
        
        for i, item in enumerate(self.dataset_info, 1):
            print(f"[{i}/{len(self.dataset_info)}] {item['filename']}")
            
            try:
                # 원시 키포인트 로드
                keypoints = np.load(item['keypoint_file'])
                
                # 전처리
                processed = self.preprocess_single_video(keypoints)
                
                # 데이터 추가
                X_list.append(processed)
                y_list.append(class_to_label[item['class']])
                
                print(f"  └ {keypoints.shape} → {processed.shape}")
                
            except Exception as e:
                print(f"  └ ❌ 오류: {e}")
                continue
        
        # numpy 배열로 변환
        X = np.array(X_list)  # (samples, 400, 33, 4)
        y = np.array(y_list)  # (samples,)
        
        print(f"\n📦 최종 데이터 shape:")
        print(f"   X: {X.shape} (samples, frames, keypoints, coords)")
        print(f"   y: {y.shape}")
        
        # Train/Validation 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y  # 클래스 비율 유지
        )
        
        print(f"\n🔀 데이터 분할:")
        print(f"   Train: {X_train.shape[0]}개 (normal: {np.sum(y_train==0)}, spacia: {np.sum(y_train==1)})")
        print(f"   Val:   {X_val.shape[0]}개 (normal: {np.sum(y_val==0)}, spacia: {np.sum(y_val==1)})")
        
        # 저장
        output_dir = Path('processed_data/preprocessed')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / 'X_train.npy', X_train)
        np.save(output_dir / 'X_val.npy', X_val)
        np.save(output_dir / 'y_train.npy', y_train)
        np.save(output_dir / 'y_val.npy', y_val)
        
        # 메타 정보 저장
        meta = {
            'target_frames': self.target_frames,
            'num_keypoints': 33,
            'class_to_label': class_to_label,
            'train_size': len(X_train),
            'val_size': len(X_val)
        }
        
        with open(output_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"\n💾 저장 완료: {output_dir}")
        print("="*60)
        
        return X_train, X_val, y_train, y_val


def main():
    """메인 실행"""
    # 전처리기 생성 (target_frames=400)
    preprocessor = GaitDataPreprocessor(target_frames=400)
    
    # 전처리 실행
    X_train, X_val, y_train, y_val = preprocessor.process_dataset(
        test_size=0.2,
        random_state=42
    )
    
    print(f"\n✅ 전처리 완료!")
    print(f"   학습 데이터: {X_train.shape}")
    print(f"   검증 데이터: {X_val.shape}")


if __name__ == "__main__":
    main()