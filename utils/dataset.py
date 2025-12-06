import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split


class GaitDataPreprocessor:
    def __init__(self, raw_data_dir='processed_data/raw_keypoints', target_frames=400):
        self.raw_data_dir = Path(raw_data_dir)
        self.target_frames = target_frames
        
        # 좌우 반전 시 swap할 키포인트 쌍
        self.SWAP_PAIRS = [
            (1, 4), (2, 5), (3, 6),
            (7, 8), (9, 10), (11, 12),
            (13, 14), (15, 16), (17, 18),
            (19, 20), (21, 22), (23, 24),
            (25, 26), (27, 28), (29, 30), (31, 32),
        ]
        
        # 상체(0~22), 하체(23~32)
        self.UPPER_BODY = list(range(0, 23))
        self.LOWER_BODY = list(range(23, 33))
        
        # 데이터셋 정보 로드
        info_file = self.raw_data_dir / 'dataset_info.json'
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                self.dataset_info = json.load(f)
        else:
            self.dataset_info = []
    
    def trim_to_person(self, keypoints, visibility_threshold=0.3):
        # visibility 기반으로 사람 등장 구간만 추출
        avg_visibility = np.mean(keypoints[:, :, 3], axis=1)
        valid_frames = np.where(avg_visibility > visibility_threshold)[0]
        
        if len(valid_frames) == 0:
            return keypoints
        
        start_idx = valid_frames[0]
        end_idx = valid_frames[-1] + 1
        return keypoints[start_idx:end_idx]
    
    def resample_frames(self, keypoints, target_frames):
        # 선형 보간으로 프레임 수 통일
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
    
    # 데이터 증강 함수들
    def augment_time_shift(self, keypoints, shift_ratio=0.1):
        max_shift = int(len(keypoints) * shift_ratio)
        shift = np.random.randint(-max_shift, max_shift + 1)
        
        if shift == 0:
            return keypoints
        
        result = np.zeros_like(keypoints)
        if shift > 0:
            result[shift:] = keypoints[:-shift]
            result[:shift] = keypoints[0]
        else:
            result[:shift] = keypoints[-shift:]
            result[shift:] = keypoints[-1]
        
        return result
    
    def augment_horizontal_flip(self, keypoints):
        result = keypoints.copy()
        result[:, :, 0] = 1.0 - result[:, :, 0]
        
        for left_idx, right_idx in self.SWAP_PAIRS:
            temp = result[:, left_idx, :].copy()
            result[:, left_idx, :] = result[:, right_idx, :]
            result[:, right_idx, :] = temp
        
        return result
    
    def augment_scale(self, keypoints, scale_range=(0.95, 1.05)):
        result = keypoints.copy()
        scale = np.random.uniform(scale_range[0], scale_range[1])
        
        hip_center = (keypoints[:, 23, :2] + keypoints[:, 24, :2]) / 2
        
        for i in range(33):
            result[:, i, 0] = hip_center[:, 0] + (keypoints[:, i, 0] - hip_center[:, 0]) * scale
            result[:, i, 1] = hip_center[:, 1] + (keypoints[:, i, 1] - hip_center[:, 1]) * scale
        
        return result
    
    def augment_noise(self, keypoints, noise_std=0.008):
        result = keypoints.copy()
        noise = np.random.normal(0, noise_std, (keypoints.shape[0], 33, 3))
        result[:, :, :3] += noise
        return result
    
    def augment_upper_body_mask(self, keypoints, mask_ratio=0.15):
        # 하체는 보행 핵심이므로 마스킹 제외
        result = keypoints.copy()
        num_mask = int(len(self.UPPER_BODY) * mask_ratio)
        mask_indices = np.random.choice(self.UPPER_BODY, num_mask, replace=False)
        
        for idx in mask_indices:
            result[:, idx, :] = 0
        
        return result
    
    def apply_augmentation(self, keypoints):
        result = keypoints.copy()
        
        if np.random.random() < 0.5:
            result = self.augment_time_shift(result)
        if np.random.random() < 0.5:
            result = self.augment_horizontal_flip(result)
        if np.random.random() < 0.5:
            result = self.augment_scale(result)
        if np.random.random() < 0.5:
            result = self.augment_noise(result)
        if np.random.random() < 0.3:
            result = self.augment_upper_body_mask(result)
        
        return result
    
    def preprocess_single_video(self, keypoints):
        trimmed = self.trim_to_person(keypoints)
        resampled = self.resample_frames(trimmed, self.target_frames)
        return resampled
    
    def process_dataset(self, test_size=0.2, random_state=42, augment=True, augment_factor=3):
        X_list = []
        y_list = []
        class_to_label = {'normal': 0, 'special': 1}
        
        for item in self.dataset_info:
            try:
                keypoints = np.load(item['keypoint_file'])
                processed = self.preprocess_single_video(keypoints)
                X_list.append(processed)
                y_list.append(class_to_label[item['class']])
            except Exception as e:
                print(f"오류: {item['filename']} - {e}")
                continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Train/Val 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 학습 데이터만 증강
        if augment:
            X_aug_list = [X_train]
            y_aug_list = [y_train]
            
            for _ in range(augment_factor - 1):
                X_augmented = np.array([self.apply_augmentation(x) for x in X_train])
                X_aug_list.append(X_augmented)
                y_aug_list.append(y_train)
            
            X_train = np.concatenate(X_aug_list, axis=0)
            y_train = np.concatenate(y_aug_list, axis=0)
            
            shuffle_idx = np.random.permutation(len(X_train))
            X_train = X_train[shuffle_idx]
            y_train = y_train[shuffle_idx]
        
        # 저장
        output_dir = Path('processed_data/preprocessed')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / 'X_train.npy', X_train)
        np.save(output_dir / 'X_val.npy', X_val)
        np.save(output_dir / 'y_train.npy', y_train)
        np.save(output_dir / 'y_val.npy', y_val)
        
        meta = {
            'target_frames': self.target_frames,
            'num_keypoints': 33,
            'class_to_label': class_to_label,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'augmented': augment,
            'augment_factor': augment_factor if augment else 1
        }
        
        with open(output_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"전처리 완료 - Train: {len(X_train)}, Val: {len(X_val)}")
        return X_train, X_val, y_train, y_val


def main():
    preprocessor = GaitDataPreprocessor(target_frames=400)
    
    X_train, X_val, y_train, y_val = preprocessor.process_dataset(
        test_size=0.2,
        random_state=42,
        augment=True,
        augment_factor=4
    )


if __name__ == "__main__":
    main()