from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import numpy as np
import threading
import time
from datetime import datetime
import os
import sys


# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fall_detector_3stage import FallDetector3Stage
from services.predict import GaitPredictor
import mediapipe as mp

# Firebase API
try:
    from api.firebase_api import FirebaseAPI
    firebase = FirebaseAPI(
        cred_path='firebase-credentials.json',
        bucket_name='capstone-3d5ef.firebasestorage.app'  # 실제 프로젝트 ID로 변경
    )
    FIREBASE_ENABLED = True
    print("✅ Firebase 연동 활성화")
except Exception as e:
    firebase = None
    FIREBASE_ENABLED = False
    print(f"⚠️ Firebase 비활성화: {e}")

app = Flask(__name__)

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 폴더 생성
RECORDINGS_FOLDER = 'recordings/original'  # 원본 녹화 영상
RESULTS_FOLDER = 'recordings/analyzed'     # 분석 결과 영상
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# 전역 변수
camera = None
recording = False
countdown = 0  # 카운트다운 상태
keypoints_buffer = []
frames_buffer = []  # 원본 프레임 저장
analysis_result = None
pose_detector = None
result_video_path = None  # 결과 영상 경로

# 서비스 초기화
fall_detector = FallDetector3Stage()
gait_predictor = None  # 필요할 때 로드


class VideoCamera:
    """웹캠 관리 클래스"""
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, frame = self.video.read()
        return success, frame


def gen_frames():
    """비디오 스트림 생성"""
    global camera, recording, countdown, keypoints_buffer, frames_buffer, pose_detector
    
    if camera is None:
        camera = VideoCamera()
    
    if pose_detector is None:
        pose_detector = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    while True:
        success, frame = camera.get_frame()
        if not success:
            break
        
        # MediaPipe 처리 (항상 수행하지만 화면에는 표시 안함)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(frame_rgb)
        
        # 녹화 중일 때만 키포인트와 프레임 저장
        if recording and countdown == 0:
            # 프레임은 무조건 저장
            frames_buffer.append(frame.copy())
            
            # 키포인트가 있을 때만 저장
            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
                keypoints_buffer.append(landmarks)
            else:
                # 키포인트가 없으면 빈 데이터 추가
                keypoints_buffer.append([[0, 0, 0, 0]] * 33)
        
        # 카운트다운 표시
        if countdown > 0:
            # 반투명 배경
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), 
                         (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
            
            # 카운트다운 숫자
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(countdown)
            text_size = cv2.getTextSize(text, font, 5, 10)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            
            cv2.putText(frame, text, (text_x, text_y), font, 
                       5, (255, 255, 255), 10)
        
        # 녹화 상태 표시
        elif recording:
            cv2.putText(frame, f'RECORDING... ({len(keypoints_buffer)} frames)', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
        
        # JPEG 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/history')
def history_page():
    """이전 기록 페이지"""
    return render_template('history.html')


@app.route('/check_firebase', methods=['GET'])
def check_firebase():
    """Firebase 연결 상태 확인"""
    return jsonify({
        'enabled': FIREBASE_ENABLED,
        'message': 'Firebase is active' if FIREBASE_ENABLED else 'Firebase is not configured'
    })


def run_pipeline():
    """파이프라인 실행 (백그라운드)"""
    global keypoints_buffer, frames_buffer, analysis_result, gait_predictor, result_video_path
    
    print("\n" + "="*60)
    print("🔍 파이프라인 시작")
    print("="*60)
    
    # 데이터 검증
    if len(keypoints_buffer) == 0 or len(frames_buffer) == 0:
        print("❌ 오류: 수집된 데이터가 없습니다!")
        analysis_result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_frames': 0,
            'fall_detection': {
                'is_fall': False,
                'confidence': 0.0,
                'stage': -1,
                'reason': '수집된 데이터가 없습니다'
            },
            'gait_classification': {
                'error': '수집된 데이터가 없습니다'
            }
        }
        return
    
    # 타임스탬프
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 키포인트를 numpy 배열로 변환
    keypoints = np.array(keypoints_buffer)
    print(f"📊 수집된 프레임: {len(keypoints)}")
    print(f"📊 수집된 영상 프레임: {len(frames_buffer)}")
    
    result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_frames': len(keypoints)
    }
    
    # 원본 영상 저장
    original_video_path = os.path.join(RECORDINGS_FOLDER, f'recording_{timestamp}.mp4')
    save_original_video(frames_buffer, original_video_path)
    print(f"💾 원본 영상 저장: {original_video_path}")
    
    # === 1단계: 낙상 감지 ===
    print("\n[1단계] 낙상 감지 실행...")
    
    # 키포인트 형식 변환 (33, 4) -> (33, 3) - visibility 제외
    keypoints_xyz = keypoints[:, :, :3]
    
    fall_result = fall_detector.detect(keypoints_xyz)
    
    result['fall_detection'] = {
        'is_fall': fall_result['is_fall'],
        'confidence': fall_result['confidence'],
        'stage': fall_result['stage'],
        'reason': fall_result['reason']
    }
    
    if fall_result['is_fall']:
        print("🚨 낙상 감지! 결과 영상 생성...")
        result_video_path = create_result_video(
            frames_buffer, keypoints, 
            'fall', fall_result['confidence'],
            timestamp
        )
        result['gait_classification'] = None
        analysis_result = result
        return
    
    print("✅ 낙상 없음, 보행 분류로 진행...")
    
    # === 2단계: 보행 분류 ===
    print("\n[2단계] 보행 분류 실행...")
    
    # GaitPredictor 초기화 (처음 한 번만)
    if gait_predictor is None:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pt')
        gait_predictor = GaitPredictor(model_path=model_path, target_frames=400)
    
    # 키포인트로 직접 예측
    try:
        # 전처리
        processed = gait_predictor.preprocess_keypoints(keypoints)
        
        # 예측
        import torch
        with torch.no_grad():
            prediction_prob = gait_predictor.model(processed)[0][0].item()
        
        prediction = 1 if prediction_prob > 0.5 else 0
        
        result['gait_classification'] = {
            'prediction': prediction,  # 0: 정상, 1: 비정상
            'confidence': prediction_prob,
            'label': 'Normal' if prediction == 0 else 'Abnormal'
        }
        
        print(f"✅ 보행 분류 완료: {result['gait_classification']['label']}")
        
        # 결과 영상 생성
        print("🎬 결과 영상 생성 중...")
        result_video_path = create_result_video(
            frames_buffer, keypoints,
            'normal' if prediction == 0 else 'abnormal',
            prediction_prob,
            timestamp
        )
        
    except Exception as e:
        print(f"❌ 보행 분류 오류: {e}")
        result['gait_classification'] = {
            'error': str(e)
        }
    
    # === 최종 결과 저장 ===
    analysis_result = result
    
    # === Firebase 업로드 ===
    if FIREBASE_ENABLED:
        try:
            print("\n🔥 Firebase 업로드 시작...")
            
            # 결과 타입 결정
            if result['fall_detection']['is_fall']:
                result_type = 'fall'
            elif result['gait_classification']:
                result_type = 'normal' if result['gait_classification']['prediction'] == 0 else 'abnormal'
            else:
                result_type = 'unknown'
            
            # 1. 원본 영상 업로드
            original_url = firebase.upload_video(
                original_video_path,
                f'videos/original/{os.path.basename(original_video_path)}'
            )
            print(f"  ✅ 원본 영상 업로드: {original_url}")
            
            # 2. 분석 영상 업로드
            analyzed_url = firebase.upload_video(
                result_video_path,
                f'videos/analyzed/{os.path.basename(result_video_path)}'
            )
            print(f"  ✅ 분석 영상 업로드: {analyzed_url}")
            
            # 3. Firestore에 결과 저장
            firebase_data = {
                'timestamp': result['timestamp'],
                'result_type': result_type,
                'fall_detection': result['fall_detection'],
                'gait_classification': result['gait_classification'],
                'keypoints': keypoints.tolist(),  # numpy → list 변환
                'original_video_url': original_url,
                'analyzed_video_url': analyzed_url,
                'total_frames': result['total_frames']
            }
            
            doc_id = firebase.save_analysis_result(firebase_data)
            print(f"  ✅ Firestore 저장 완료: {doc_id}")
            
        except Exception as e:
            print(f"  ❌ Firebase 업로드 오류: {e}")
    
    print("\n" + "="*60)
    print("✅ 파이프라인 완료!")
    print("="*60)


def save_original_video(frames, output_path):
    """원본 영상 저장"""
    if len(frames) == 0:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()


def create_result_video(frames, keypoints, result_type, confidence, timestamp):
    """
    결과 영상 생성 (스켈레톤 + 색상 오버레이)
    
    Args:
        frames: 원본 프레임 리스트
        keypoints: 키포인트 배열
        result_type: 'fall', 'normal', 'abnormal'
        confidence: 신뢰도
        timestamp: 타임스탬프
    
    Returns:
        str: 저장된 영상 경로
    """
    if len(frames) == 0:
        return None
    
    # 색상 설정
    if result_type == 'fall':
        overlay_color = (0, 0, 255)  # 빨강
        text = 'FALL DETECTED'
    elif result_type == 'abnormal':
        overlay_color = (0, 165, 255)  # 주황
        text = 'ABNORMAL GAIT'
    else:  # normal
        overlay_color = (0, 255, 0)  # 녹색
        text = 'NORMAL GAIT'
    
    # 출력 경로
    output_path = os.path.join(RESULTS_FOLDER, f'result_{result_type}_{timestamp}.mp4')
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    # MediaPipe 포즈 초기화
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    for idx, frame in enumerate(frames):
        # 프레임 복사
        result_frame = frame.copy()
        
        # MediaPipe로 스켈레톤 그리기
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                result_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
        
        # 색상 오버레이 (반투명)
        overlay = result_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), overlay_color, -1)
        result_frame = cv2.addWeighted(result_frame, 0.85, overlay, 0.15, 0)
        
        # 테두리
        cv2.rectangle(result_frame, (5, 5), (width-5, height-5), overlay_color, 8)
        
        # 텍스트
        cv2.putText(result_frame, text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, overlay_color, 3)
        cv2.putText(result_frame, f'Confidence: {confidence*100:.1f}%', (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, overlay_color, 2)
        
        out.write(result_frame)
    
    out.release()
    pose.close()
    
    print(f"✅ 결과 영상 저장: {output_path}")
    return output_path


# ============ Flask 라우트 ============

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """비디오 스트림"""
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_recording', methods=['POST'])
def start_recording():
    """녹화 시작 (3초 카운트다운)"""
    global recording, countdown, keypoints_buffer, frames_buffer, analysis_result, result_video_path
    
    if recording or countdown > 0:
        return jsonify({'status': 'error', 'message': 'Already recording or counting down'})
    
    # 초기화
    keypoints_buffer = []
    frames_buffer = []
    analysis_result = None
    result_video_path = None
    
    # 카운트다운 + 자동 녹화 종료
    def countdown_and_record():
        global countdown, recording
        
        # 3초 카운트다운
        for i in range(3, 0, -1):
            countdown = i
            time.sleep(1)
        countdown = 0
        recording = True
        
        # 10초 녹화 후 자동 종료
        time.sleep(10)
        
        if recording:  # 아직 녹화 중이면
            recording = False
            # 자동으로 분석 시작
            run_pipeline()
    
    thread = threading.Thread(target=countdown_and_record)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'success', 'message': 'Countdown started'})


@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """녹화 중지 및 분석 시작"""
    global recording
    
    if not recording:
        return jsonify({'status': 'error', 'message': 'Not recording'})
    
    recording = False
    
    # 백그라운드에서 파이프라인 실행
    thread = threading.Thread(target=run_pipeline)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'success', 'message': 'Recording stopped, analysis started'})


@app.route('/get_result', methods=['GET'])
def get_result():
    """분석 결과 조회"""
    global analysis_result, result_video_path
    
    if analysis_result is None:
        return jsonify({'status': 'processing'})
    
    return jsonify({
        'status': 'completed',
        'result': analysis_result,
        'has_video': result_video_path is not None
    })


@app.route('/result_video_feed')
def result_video_feed():
    """결과 영상 스트리밍"""
    global result_video_path
    
    if result_video_path is None or not os.path.exists(result_video_path):
        return "No result video", 404
    
    def gen_result_video():
        cap = cv2.VideoCapture(result_video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # 영상 끝나면 처음부터 다시
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            time.sleep(0.033)  # 30fps
        
        cap.release()
    
    return Response(gen_result_video(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/reset', methods=['POST'])
def reset():
    """시스템 초기화"""
    global recording, countdown, keypoints_buffer, frames_buffer, analysis_result, result_video_path
    
    recording = False
    countdown = 0
    keypoints_buffer = []
    frames_buffer = []
    analysis_result = None
    result_video_path = None
    
    return jsonify({'status': 'success', 'message': 'System reset'})


# ============ Firebase 기록 조회 라우트 ============

@app.route('/get_history', methods=['GET'])
def get_history():
    """이전 분석 기록 조회"""
    if not FIREBASE_ENABLED:
        return jsonify({'status': 'error', 'message': 'Firebase not enabled'})
    
    filter_type = request.args.get('type', 'all')  # all, fall, normal, abnormal
    limit = int(request.args.get('limit', 50))
    
    try:
        if filter_type == 'all':
            records = firebase.get_all_records(limit=limit)
        else:
            records = firebase.get_records_by_type(filter_type, limit=limit)
        
        return jsonify({
            'status': 'success',
            'records': records,
            'count': len(records)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/get_record/<doc_id>', methods=['GET'])
def get_record(doc_id):
    """특정 기록 상세 조회"""
    if not FIREBASE_ENABLED:
        return jsonify({'status': 'error', 'message': 'Firebase not enabled'})
    
    try:
        record = firebase.get_record_by_id(doc_id)
        if record:
            return jsonify({'status': 'success', 'record': record})
        else:
            return jsonify({'status': 'error', 'message': 'Record not found'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/delete_record/<doc_id>', methods=['DELETE'])
def delete_record(doc_id):
    """기록 삭제"""
    if not FIREBASE_ENABLED:
        return jsonify({'status': 'error', 'message': 'Firebase not enabled'})
    
    try:
        firebase.delete_record(doc_id)
        return jsonify({'status': 'success', 'message': 'Record deleted'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 보행 분석 시스템 시작")
    print("="*60)
    print(f"\n📁 녹화 폴더: {RECORDINGS_FOLDER}")
    print(f"📁 결과 폴더: {RESULTS_FOLDER}")
    print("\n📱 브라우저에서 http://localhost:5000 접속\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)