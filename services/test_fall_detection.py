import cv2
import mediapipe as mp
import numpy as np
import os
from models.fall_detector_3stage import FallDetector3Stage

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints_from_video(video_path):
    """
    영상에서 MediaPipe 키포인트 추출
    
    Args:
        video_path: 영상 파일 경로
        
    Returns:
        numpy array: (frames, 33, 3) - 키포인트 시퀀스
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")
    
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    
    # 영상 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\n📹 영상 정보:")
    print(f"   파일: {os.path.basename(video_path)}")
    print(f"   FPS: {fps}")
    print(f"   총 프레임: {total_frames}")
    print(f"   길이: {duration:.1f}초")
    
    keypoints_list = []
    frame_count = 0
    
    print(f"\n⏳ 키포인트 추출 중...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # MediaPipe 처리
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            # 33개 키포인트 추출
            landmarks = results.pose_landmarks.landmark
            keypoints = np.array([
                [lm.x, lm.y, lm.z] for lm in landmarks
            ])
            keypoints_list.append(keypoints)
        
        frame_count += 1
        
        # 진행률 표시
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   진행: {progress:.1f}% ({frame_count}/{total_frames})")
    
    cap.release()
    pose.close()
    
    if len(keypoints_list) == 0:
        raise ValueError("키포인트를 추출할 수 없습니다.")
    
    print(f"✓ 키포인트 추출 완료: {len(keypoints_list)} 프레임")
    
    return np.array(keypoints_list)


def visualize_result(video_path, result):
    """
    결과를 영상에 표시
    
    Args:
        video_path: 원본 영상 경로
        result: 낙상 검출 결과
    """
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 영상 저장 설정
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = video_path.replace('.mp4', '_result.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\n🎬 결과 영상 생성 중...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # MediaPipe 처리
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # 랜드마크 그리기
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )
        
        # 결과 표시
        if result['is_fall']:
            # 낙상 감지
            cv2.putText(frame, 'FALL DETECTED!', (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame, f'Confidence: {result["confidence"]*100:.1f}%', 
                       (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 빨간 테두리
            cv2.rectangle(frame, (10, 10), (width-10, height-10), 
                         (0, 0, 255), 10)
        else:
            # 낙상 아님
            cv2.putText(frame, 'NO FALL', (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f'Stage {result["stage"]} Failed', 
                       (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    pose.close()
    
    print(f"✓ 결과 영상 저장: {output_path}")


def test_single_video(video_path, visualize=True):
    """단일 영상 테스트"""
    print("\n" + "="*70)
    print(f"🎯 테스트: {os.path.basename(video_path)}")
    print("="*70)
    
    # 키포인트 추출
    keypoints = extract_keypoints_from_video(video_path)
    
    # 낙상 검출
    detector = FallDetector3Stage()
    result = detector.detect(keypoints)
    
    # 최종 결과 출력
    print("\n" + "="*70)
    print("📊 최종 결과")
    print("="*70)
    
    if result['is_fall']:
        print(f"\n🚨 낙상 감지!")
        print(f"   신뢰도: {result['confidence']*100:.1f}%")
        print(f"   모든 단계 통과")
    else:
        print(f"\n✓ 낙상 아님")
        print(f"   실패 단계: {result['stage']}")
        print(f"   이유: {result['reason']}")
    
    print("="*70)
    
    # 시각화
    if visualize:
        visualize_result(video_path, result)
    
    return result


def test_multiple_videos(video_folder):
    """여러 영상 일괄 테스트"""
    if not os.path.exists(video_folder):
        print(f"❌ 폴더를 찾을 수 없습니다: {video_folder}")
        return
    
    # mp4 파일 찾기
    video_files = [f for f in os.listdir(video_folder) 
                   if f.endswith('.mp4') and not f.endswith('_result.mp4')]
    
    if len(video_files) == 0:
        print(f"❌ {video_folder}에 mp4 파일이 없습니다.")
        return
    
    print(f"\n📂 {len(video_files)}개 영상 테스트 시작")
    
    results = {}
    
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        result = test_single_video(video_path, visualize=False)
        results[video_file] = result
    
    # 전체 요약
    print("\n" + "="*70)
    print("📊 전체 요약")
    print("="*70)
    
    for video_file, result in results.items():
        status = "🚨 낙상" if result['is_fall'] else "✓ 정상"
        confidence = f"({result['confidence']*100:.1f}%)" if result['is_fall'] else f"(Stage {result['stage']})"
        print(f"{status} - {video_file} {confidence}")
    
    print("="*70)


def main():
    """메인 함수"""
    print("\n" + "="*70)
    print("🚀 3단계 낙상 검출 테스트")
    print("="*70)
    
    print("\n사용 방법을 선택하세요:")
    print("1. 단일 영상 테스트")
    print("2. 폴더 내 모든 영상 테스트")
    
    choice = input("\n선택 (1 or 2): ").strip()
    
    if choice == '1':
        video_path = input("영상 파일 경로: ").strip()
        test_single_video(video_path, visualize=True)
    
    elif choice == '2':
        folder_path = input("폴더 경로: ").strip()
        test_multiple_videos(folder_path)
    
    else:
        print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    main()