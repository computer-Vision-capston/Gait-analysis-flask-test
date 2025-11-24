import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

class FallDetector3Stage:
    def __init__(self):
        # MediaPipe 키포인트 인덱스
        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
        
        # === 1단계: 하강 임계값 ===
        self.MIN_DESCENT_VELOCITY = 0.03  # 정규화 좌표 기준
        self.CRITICAL_DESCENT_VELOCITY = 0.05
        
        # === 2단계: 높이 변화 임계값 ===
        self.MIN_HEIGHT_DROP = 0.15  # 화면의 15%
        self.CRITICAL_HEIGHT_DROP = 0.25  # 화면의 25%
        
        # === 3단계: 최종 자세 임계값 ===
        self.GROUND_THRESHOLD = 0.6  # y > 0.6 (화면 하단 40%)
        self.HORIZONTAL_THRESHOLD = 0.1  # 머리-엉덩이 차이
        self.LYING_ANGLE = 60  # 신체 기울기 (도)
        self.STATIONARY_THRESHOLD = 0.01  # 움직임 임계값
        self.MIN_STATIONARY_FRAMES = 30  # 1초 (30fps 기준)
    
    def detect(self, keypoints_sequence):
        if len(keypoints_sequence) < 30:
            return {
                'is_fall': False,
                'confidence': 0.0,
                'stage': -1,
                'reason': '영상이 너무 짧음 (최소 1초 필요)',
                'details': {}
            }
        
        print("\n" + "="*60)
        print("3단계 낙상 검증 시작")
        print("="*60)
        
        # 주요 키포인트 계산
        hip_center = (keypoints_sequence[:, self.LEFT_HIP, :2] + 
                     keypoints_sequence[:, self.RIGHT_HIP, :2]) / 2
        nose = keypoints_sequence[:, self.NOSE, :2]
        
        # ===== 1단계: 급격한 하강 감지 =====
        print("\n[1단계] 급격한 하강 감지 중")
        phase1 = self._phase1_rapid_descent(hip_center)
        
        print(f"최대 하강 속도: {phase1['max_velocity']:.4f}")
        print(f"임계값: {self.MIN_DESCENT_VELOCITY:.4f}")
        
        if not phase1['passed']:
            print(f"1단계 실패: {phase1['reason']}")
            return {
                'is_fall': False,
                'confidence': 0.0,
                'stage': 1,
                'reason': phase1['reason'],
                'details': phase1
            }
        print(f"1단계 통과!")
        
        # ===== 2단계: 큰 높이 변화 확인 =====
        print("\n[2단계] 큰 높이 변화 확인...")
        phase2 = self._phase2_height_drop(hip_center)
        
        print(f"초반 높이: {phase2['initial_height']:.3f}")
        print(f"후반 높이: {phase2['final_height']:.3f}")
        print(f"높이 변화: {phase2['height_drop']:.3f}")
        print(f"임계값: {self.MIN_HEIGHT_DROP:.3f}")
        
        if not phase2['passed']:
            print(f"2단계 실패: {phase2['reason']}")
            return {
                'is_fall': False,
                'confidence': 0.0,
                'stage': 2,
                'reason': phase2['reason'],
                'details': {**phase1, **phase2}
            }
        print(f"2단계 통과!")
        
        # ===== 3단계: 최종 자세 검증 =====
        print("\n[3단계] 최종 자세 검증 (5가지 조건)...")
        phase3 = self._phase3_final_posture(
            keypoints_sequence, hip_center, nose
        )
        
        print(f"1. 낮은 위치: {'OK' if phase3['is_on_ground'] else 'N'} (y={phase3['final_height']:.3f})")
        print(f"2. 수평 자세: {'OK' if phase3['is_horizontal'] else 'N'} (차이={phase3['head_hip_diff']:.3f})")
        print(f"3. 누운 각도: {'OK' if phase3['is_lying'] else 'N'} (각도={phase3['body_angle']:.1f}°)")
        print(f"4. 정지 상태: {'OK' if phase3['is_stationary'] else 'N'} (움직임={phase3['avg_movement']:.4f})")
        print(f"5. 지속 시간: {'OK' if phase3['sufficient_duration'] else 'N'} ({phase3['stationary_frames']}프레임)")
        print(f"통과: {phase3['checks_passed']}/5")
        
        if not phase3['passed']:
            print(f"3단계 실패: 최종 자세가 낙상과 다름")
            return {
                'is_fall': False,
                'confidence': 0.0,
                'stage': 3,
                'reason': '최종 자세가 낙상과 다름',
                'details': {**phase1, **phase2, **phase3}
            }
        print(f"3단계 통과!")
        
        # ===== 모든 단계 통과 → 낙상! =====
        confidence = self._calculate_confidence(phase1, phase2, phase3)
        
        print("\n" + "="*60)
        print(f"낙상 감지! (신뢰도: {confidence*100:.1f}%)")
        print("="*60)
        
        return {
            'is_fall': True,
            'confidence': confidence,
            'stage': 'COMPLETED',
            'reason': '모든 단계 통과',
            'details': {
                'phase1': phase1,
                'phase2': phase2,
                'phase3': phase3
            }
        }
    
    def _phase1_rapid_descent(self, hip_center):
        """1단계: 급격한 하강 감지"""
        y_positions = hip_center[:, 1]
        
        # 속도 계산 (프레임간 변화)
        velocities = np.diff(y_positions)
        
        # 가장 빠른 하강 (양수 = 하강)
        max_velocity = np.max(velocities)
        
        # 판정
        passed = max_velocity >= self.MIN_DESCENT_VELOCITY
        is_critical = max_velocity >= self.CRITICAL_DESCENT_VELOCITY
        
        reason = "급격한 하강 없음" if not passed else ""
        
        return {
            'passed': passed,
            'max_velocity': max_velocity,
            'is_critical': is_critical,
            'reason': reason
        }
    
    def _phase2_height_drop(self, hip_center):
        """2단계: 큰 높이 변화 확인"""
        y_positions = hip_center[:, 1]
        
        # 초반 평균 높이 (처음 15프레임)
        initial_height = np.mean(y_positions[:15])
        
        # 후반 평균 높이 (마지막 15프레임)
        final_height = np.mean(y_positions[-15:])
        
        # 높이 차이 (양수 = 하강)
        height_drop = final_height - initial_height
        
        # 판정
        passed = height_drop >= self.MIN_HEIGHT_DROP
        is_critical = height_drop >= self.CRITICAL_HEIGHT_DROP
        
        reason = "높이 변화 부족" if not passed else ""
        
        return {
            'passed': passed,
            'height_drop': height_drop,
            'initial_height': initial_height,
            'final_height': final_height,
            'is_critical': is_critical,
            'reason': reason
        }
    
    def _phase3_final_posture(self, keypoints_sequence, hip_center, nose):
        """3단계: 최종 자세 검증 (5가지 조건)"""
        # 마지막 30프레임 (1초) 분석
        last_frames = min(30, len(keypoints_sequence))
        last_kp = keypoints_sequence[-last_frames:]
        last_hip = hip_center[-last_frames:]
        last_nose = nose[-last_frames:]
        
        # === 조건 1: 낮은 위치 (바닥에 있음) ===
        final_height = np.mean(last_hip[:, 1])
        is_on_ground = final_height > self.GROUND_THRESHOLD
        
        # === 조건 2: 수평 자세 (머리와 엉덩이가 같은 높이) ===
        head_hip_diff = np.abs(last_nose[:, 1] - last_hip[:, 1])
        avg_head_hip_diff = np.mean(head_hip_diff)
        is_horizontal = avg_head_hip_diff < self.HORIZONTAL_THRESHOLD
        
        # === 조건 3: 신체 각도 (누운 자세) ===
        body_angles = []
        for kp in last_kp:
            angle = self._calculate_body_angle(kp)
            body_angles.append(angle)
        avg_body_angle = np.mean(body_angles)
        is_lying = avg_body_angle > self.LYING_ANGLE
        
        # === 조건 4: 정지 상태 ===
        movements = np.diff(last_hip, axis=0)
        movement_magnitude = np.linalg.norm(movements, axis=1)
        avg_movement = np.mean(movement_magnitude)
        is_stationary = avg_movement < self.STATIONARY_THRESHOLD
        
        # === 조건 5: 지속 시간 (1초 이상) ===
        stationary_frames = np.sum(movement_magnitude < self.STATIONARY_THRESHOLD * 1.5)
        sufficient_duration = stationary_frames >= self.MIN_STATIONARY_FRAMES
        
        # === 종합 판정 ===
        checks = [
            is_on_ground,
            is_horizontal,
            is_lying,
            is_stationary,
            sufficient_duration
        ]
        
        checks_passed = sum(checks)
        passed = checks_passed >= 2  # 5개 중 4개 이상
        
        return {
            'passed': passed,
            'checks_passed': checks_passed,
            'final_height': final_height,
            'is_on_ground': is_on_ground,
            'head_hip_diff': avg_head_hip_diff,
            'is_horizontal': is_horizontal,
            'body_angle': avg_body_angle,
            'is_lying': is_lying,
            'avg_movement': avg_movement,
            'is_stationary': is_stationary,
            'stationary_frames': stationary_frames,
            'sufficient_duration': sufficient_duration
        }
    
    def _calculate_body_angle(self, keypoints):
        """신체 기울기 계산 (어깨-엉덩이 라인)"""
        hip_center = (keypoints[self.LEFT_HIP, :2] + 
                     keypoints[self.RIGHT_HIP, :2]) / 2
        shoulder_center = (keypoints[self.LEFT_SHOULDER, :2] + 
                          keypoints[self.RIGHT_SHOULDER, :2]) / 2
        
        dy = abs(shoulder_center[1] - hip_center[1])
        dx = abs(shoulder_center[0] - hip_center[0])
        
        # 수직일 때 0도, 수평일 때 90도
        angle = np.degrees(np.arctan2(dx, dy + 1e-6))
        return angle
    
    def _calculate_confidence(self, phase1, phase2, phase3):
        """신뢰도 계산"""
        score = 0.0
        
        # 1단계 (30%)
        if phase1['is_critical']:
            score += 0.30
        else:
            score += 0.15
        
        # 2단계 (30%)
        if phase2['is_critical']:
            score += 0.30
        else:
            score += 0.15
        
        # 3단계 (40%)
        checks_ratio = phase3['checks_passed'] / 5.0
        score += 0.40 * checks_ratio
        
        return min(0.95, score)


if __name__ == "__main__":
    print("FallDetector3Stage 모듈 로드 완료")