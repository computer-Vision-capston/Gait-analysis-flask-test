from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import numpy as np
import threading
import time
from datetime import datetime
import os
import sys
import json
from raspberry_camera import RaspberryPiCamera

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
        bucket_name='capstone-3d5ef.firebasestorage.app'
    )
    FIREBASE_ENABLED = True
    print("Firebase 연동 활성화")
except Exception as e:
    firebase = None
    FIREBASE_ENABLED = False
    print(f"Firebase 비활성화: {e}")

app = Flask(__name__)

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 폴더 생성
RECORDINGS_FOLDER = 'recordings/original'
RESULTS_FOLDER = 'recordings/analyzed'
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# 전역 변수
camera = None
recording = False
countdown = 0
keypoints_buffer = []
frames_buffer = []
analysis_result = None
pose_detector = None
result_video_path = None
raspberry_camera = None
camera_source = 'local'

# 서비스 초기화
fall_detector = FallDetector3Stage()
gait_predictor = None


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
        
        # MediaPipe 처리
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(frame_rgb)
        
        # 녹화 중일 때만 키포인트와 프레임 저장
        if recording and countdown == 0:
            frames_buffer.append(frame.copy())
            
            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
                keypoints_buffer.append(landmarks)
            else:
                keypoints_buffer.append([[0, 0, 0, 0]] * 33)
        
        # 카운트다운 표시
        if countdown > 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(countdown)
            text_size = cv2.getTextSize(text, font, 5, 10)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            
            cv2.putText(frame, text, (text_x, text_y), font, 5, (255, 255, 255), 10)
        
        # 녹화 상태 표시
        elif recording:
            cv2.putText(frame, f'RECORDING... ({len(keypoints_buffer)} frames)', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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
    """파이프라인 실행"""
    global keypoints_buffer, frames_buffer, analysis_result, gait_predictor, result_video_path
    
    print("\n" + "="*60)
    print("파이프라인 시작")
    print("="*60)
    
    if len(frames_buffer) == 0:
        print("오류: 수집된 데이터가 없습니다!")
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
    if len(keypoints_buffer) == 0:
        print("프레임으로부터 키포인트 추출 중...")
        pose_detector = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        for frame in frames_buffer:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
                keypoints_buffer.append(landmarks)
            else:
                keypoints_buffer.append([[0, 0, 0, 0]] * 33)
        
        pose_detector.close()
        print(f"{len(keypoints_buffer)}개 키포인트 추출 완료")
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    keypoints = np.array(keypoints_buffer)
    
    print(f"수집된 프레임: {len(keypoints)}")
    print(f"수집된 영상 프레임: {len(frames_buffer)}")
    
    result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_frames': len(keypoints)
    }
    
    # 원본 영상 저장
    original_video_path = os.path.join(RECORDINGS_FOLDER, f'recording_{timestamp}.mp4')
    save_original_video(frames_buffer, original_video_path)
    print(f"원본 영상 저장: {original_video_path}")
    
    # === 1단계: 낙상 감지 ===
    print("\n[1단계] 낙상 감지 실행")
    keypoints_xyz = keypoints[:, :, :3]
    fall_result = fall_detector.detect(keypoints_xyz)
    
    result['fall_detection'] = {
        'is_fall': fall_result['is_fall'],
        'confidence': fall_result['confidence'],
        'stage': fall_result['stage'],
        'reason': fall_result['reason']
    }
    
    if fall_result['is_fall']:
        print("낙상 감지! 결과 영상 생성 중")
        result_video_path = create_result_video(
            frames_buffer, keypoints, 'fall', fall_result['confidence'], timestamp
        )
        result['gait_classification'] = None
        analysis_result = result
        
        # Firebase 업로드
        upload_to_firebase(result, keypoints, original_video_path, result_video_path)
        return
    
    print("낙상 없음, 보행 분류로 진행")
    
    print("\n[2단계] 보행 분류 실행")
    
    if gait_predictor is None:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pt')
        gait_predictor = GaitPredictor(model_path=model_path, target_frames=400)
    
    try:
        processed = gait_predictor.preprocess_keypoints(keypoints)
        
        import torch
        with torch.no_grad():
            prediction_prob = gait_predictor.model(processed)[0][0].item()
        
        prediction = 1 if prediction_prob > 0.5 else 0
        
        result['gait_classification'] = {
            'prediction': prediction,
            'confidence': prediction_prob,
            'label': 'Normal' if prediction == 0 else 'Abnormal'
        }
        
        print(f"보행 분류 완료: {result['gait_classification']['label']}")
        
        print("결과 영상 생성 중")
        result_video_path = create_result_video(
            frames_buffer, keypoints,
            'normal' if prediction == 0 else 'abnormal',
            prediction_prob, timestamp
        )
        
    except Exception as e:
        print(f"보행 분류 오류: {e}")
        result['gait_classification'] = {'error': str(e)}
    
    analysis_result = result
    
    # Firebase 업로드
    upload_to_firebase(result, keypoints, original_video_path, result_video_path)
    
    print("\n" + "="*60)
    print("파이프라인 완료!")
    print("="*60)


def upload_to_firebase(result, keypoints, original_video_path, result_video_path):
    """Firebase 업로드"""
    if FIREBASE_ENABLED:
        try:
            print("\n🔥 Firebase 업로드 시작")
            
            # 결과 타입 결정
            if result['fall_detection']['is_fall']:
                result_type = 'fall'
            elif result['gait_classification'] and 'error' not in result['gait_classification']:
                result_type = 'normal' if result['gait_classification']['prediction'] == 0 else 'abnormal'
            else:
                result_type = 'unknown'
            
            # 원본 영상 업로드
            original_url = firebase.upload_video(
                original_video_path,
                f'videos/original/{os.path.basename(original_video_path)}'
            )
            
            # 분석 영상 업로드
            analyzed_url = firebase.upload_video(
                result_video_path,
                f'videos/analyzed/{os.path.basename(result_video_path)}'
            )
            
            # Firestore에 결과 저장
            firebase_data = {
                'timestamp': result['timestamp'],
                'result_type': result_type,
                'fall_detection': result['fall_detection'],
                'gait_classification': result['gait_classification'],
                'keypoints_json': json.dumps(keypoints.tolist()),  # JSON 문자열로 변환
                'original_video_url': original_url,
                'analyzed_video_url': analyzed_url,
                'total_frames': result['total_frames']
            }
            
            firebase.save_analysis_result(firebase_data)
            print("Firestore 저장 완료!")
            
        except Exception as e:
            print(f"Firebase 업로드 오류: {e}")


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
    """결과 영상 생성"""
    if len(frames) == 0:
        return None
    
    # 색상 설정
    if result_type == 'fall':
        overlay_color = (0, 0, 255)
        text = 'FALL DETECTED'
    elif result_type == 'abnormal':
        overlay_color = (0, 165, 255)
        text = 'ABNORMAL GAIT'
    else:
        overlay_color = (0, 255, 0)
        text = 'NORMAL GAIT'
    
    output_path = os.path.join(RESULTS_FOLDER, f'result_{result_type}_{timestamp}.mp4')
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    for idx, frame in enumerate(frames):
        result_frame = frame.copy()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                result_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
        
        # 색상 오버레이
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
    
    print(f"결과 영상 저장: {output_path}")
    return output_path


# ============ Flask 라우트 ============

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """비디오 스트림"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_recording', methods=['POST'])
def start_recording():
    """녹화 시작 (로컬 또는 라즈베리파이)"""
    global recording, countdown, keypoints_buffer, frames_buffer, analysis_result, result_video_path
    global camera_source
    
    if camera_source == 'raspberry':
        # 라즈베리파이 카메라 녹화
        if raspberry_camera and raspberry_camera.connected:
            success, result = raspberry_camera.start_recording()
            if success:
                return jsonify({'status': 'success', 'message': 'Raspberry Pi recording started', 'source': 'raspberry'})
            else:
                return jsonify({'status': 'error', 'message': result.get('error', 'Unknown error')})
        else:
            return jsonify({'status': 'error', 'message': 'Raspberry Pi camera not connected'})
    
    else:
        # 로컬 카메라 녹화 (기존 코드)
        if recording or countdown > 0:
            return jsonify({'status': 'error', 'message': 'Already recording or counting down'})
        
        keypoints_buffer = []
        frames_buffer = []
        analysis_result = None
        result_video_path = None
        
        def countdown_only():
            global countdown, recording
            
            for i in range(3, 0, -1):
                countdown = i
                time.sleep(1)
            countdown = 0
            recording = True
        
        thread = threading.Thread(target=countdown_only)
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'success', 'message': 'Local recording started', 'source': 'local'})



@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """녹화 중지 및 분석 (로컬 또는 라즈베리파이)"""
    global recording, camera_source, frames_buffer, keypoints_buffer
    
    if camera_source == 'raspberry':
        # 라즈베리파이에서 프레임 받아와서 분석
        if raspberry_camera and raspberry_camera.connected:
            success, frames, error = raspberry_camera.stop_recording_and_get_frames()
            
            if not success:
                return jsonify({'status': 'error', 'message': error or 'Failed to get frames'})
            
            if not frames or len(frames) == 0:
                return jsonify({'status': 'error', 'message': 'No frames received'})
            
            # frames_buffer를 라즈베리파이에서 받은 프레임으로 설정
            frames_buffer = frames
            keypoints_buffer = []  # 비워두고 run_pipeline에서 추출
            
            # 백그라운드에서 분석 시작
            thread = threading.Thread(target=run_pipeline)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'status': 'success', 
                'message': f'Received {len(frames)} frames from Raspberry Pi',
                'source': 'raspberry'
            })
        else:
            return jsonify({'status': 'error', 'message': 'Raspberry Pi camera not connected'})
    
    else:
        # 로컬 카메라 녹화 중지 (기존 코드)
        if not recording:
            return jsonify({'status': 'error', 'message': 'Not recording'})
        
        recording = False
        
        thread = threading.Thread(target=run_pipeline)
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'success', 'message': 'Local recording stopped', 'source': 'local'})

@app.route('/get_recording_status', methods=['GET'])
def get_recording_status():
    """녹화 상태 조회"""
    global recording, analysis_result
    
    return jsonify({
        'is_recording': recording,
        'is_analyzing': not recording and analysis_result is None and len(keypoints_buffer) > 0
    })


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
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            time.sleep(0.033)
        
        cap.release()
    
    return Response(gen_result_video(), mimetype='multipart/x-mixed-replace; boundary=frame')


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

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """동영상 파일 업로드"""
    global frames_buffer, keypoints_buffer, analysis_result, result_video_path
    
    print("=== upload_video 함수 시작 ===")
    
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file'})
    
    file = request.files['video']
    print(f"파일명: {file.filename}")
    
    if not file.filename:
        return jsonify({'status': 'error', 'message': 'No file selected'})
    
    # 임시 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_path = os.path.join(RECORDINGS_FOLDER, f'temp_{timestamp}.mp4')
    file.save(temp_path)
    print(f"임시 저장: {temp_path}")
    
    # 프레임 추출
    cap = cv2.VideoCapture(temp_path)
    temp_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        temp_frames.append(frame.copy())
    
    cap.release()
    os.remove(temp_path)
    
    print(f"추출된 프레임 수: {len(temp_frames)}")
    
    if not temp_frames:
        return jsonify({'status': 'error', 'message': 'Cannot read video'})
    
    # 전역 변수 업데이트
    frames_buffer = temp_frames
    keypoints_buffer = []
    
    print(f"frames_buffer에 저장됨: {len(frames_buffer)} 프레임")
    print(f"run_pipeline 호출 직전 확인: {len(frames_buffer)} 프레임")
    
    # run_pipeline 호출
    thread = threading.Thread(target=run_pipeline)
    thread.daemon = True
    thread.start()
    
    print("=== upload_video 함수 종료 ===")
    
    return jsonify({
        'status': 'success',
        'message': f'Uploaded {len(frames_buffer)} frames'
    })
# ============ Firebase 기록 조회 라우트 ============

# 이전 분석 기록 조회
@app.route('/get_history', methods=['GET'])
def get_history():
    if not FIREBASE_ENABLED:
        return jsonify({'status': 'error', 'message': 'Firebase not enabled'})
    
    filter_type = request.args.get('type', 'all')
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

# 특정 기록 상세 조회
@app.route('/get_record/<doc_id>', methods=['GET'])
def get_record(doc_id):
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

# 기록 삭제
@app.route('/delete_record/<doc_id>', methods=['DELETE'])
def delete_record(doc_id):
    if not FIREBASE_ENABLED:
        return jsonify({'status': 'error', 'message': 'Firebase not enabled'})
    
    try:
        firebase.delete_record(doc_id)
        return jsonify({'status': 'success', 'message': 'Record deleted'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# 라즈베리파이 연결
@app.route('/raspberry/connect', methods=['POST'])
def raspberry_connect():
    global raspberry_camera
    
    data = request.get_json()
    raspberry_ip = data.get('ip', '').strip()
    
    if not raspberry_ip:
        return jsonify({'status': 'error', 'message': 'IP 주소를 입력하세요'})
    
    # http:// 자동 추가
    if not raspberry_ip.startswith('http'):
        raspberry_ip = f'http://{raspberry_ip}:8000'
    elif ':8000' not in raspberry_ip:
        raspberry_ip = f'{raspberry_ip}:8000'
    
    try:
        raspberry_camera = RaspberryPiCamera(raspberry_ip)
        
        if raspberry_camera.connected:
            return jsonify({
                'status': 'success',
                'message': '라즈베리파이 연결 성공!',
                'url': raspberry_ip
            })
        else:
            raspberry_camera = None
            return jsonify({
                'status': 'error',
                'message': '라즈베리파이 연결 실패. IP 주소와 서버 실행 상태를 확인하세요.'
            })
    
    except Exception as e:
        raspberry_camera = None
        return jsonify({'status': 'error', 'message': f'연결 오류: {str(e)}'})

# 카메라 소스 선택
@app.route('/camera/select', methods=['POST'])
def select_camera():
    global camera_source
    
    data = request.get_json()
    source = data.get('source', 'local')
    
    if source not in ['local', 'raspberry']:
        return jsonify({'status': 'error', 'message': 'Invalid camera source'})
    
    if source == 'raspberry' and (raspberry_camera is None or not raspberry_camera.connected):
        return jsonify({'status': 'error', 'message': 'Raspberry Pi camera not connected'})
    
    camera_source = source
    print(f"카메라 소스 변경: {source}")
    
    return jsonify({
        'status': 'success',
        'source': camera_source,
        'message': f"Camera switched to {source}"
    })

# 현재 카메라 소스 및 상태 조회
@app.route('/camera/status', methods=['GET'])
def camera_status():
    global camera_source
    
    status = {
        'current_source': camera_source,
        'local': {
            'available': True,
            'name': 'PC 웹캠'
        },
        'raspberry': {
            'available': raspberry_camera is not None and raspberry_camera.connected,
            'name': '라즈베리파이 카메라',
            'url': raspberry_camera.url if raspberry_camera else None
        }
    }
    
    # 라즈베리파이 상태 확인
    if raspberry_camera and raspberry_camera.connected:
        success, pi_status = raspberry_camera.get_status()
        if success:
            status['raspberry']['recording'] = pi_status.get('recording', False)
            status['raspberry']['frame_count'] = pi_status.get('frame_count', 0)
    
    return jsonify(status)

# 라즈베리파이 비디오 스트림 프록시
@app.route('/raspberry/video_feed')
def raspberry_video_feed():
    if raspberry_camera and raspberry_camera.connected:
        import requests
        
        def generate():
            try:
                stream = requests.get(raspberry_camera.get_video_feed_url(), stream=True, timeout=10)
                for chunk in stream.iter_content(chunk_size=1024):
                    yield chunk
            except Exception as e:
                print(f"라즈베리파이 스트림 오류: {e}")
        
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return "Raspberry Pi camera not available", 404

# 아두이노 PIR 센서에서 자동 녹화 트리거
@app.route('/trigger_auto_recording', methods=['POST'])
def trigger_auto_recording():
    global recording, countdown, keypoints_buffer, frames_buffer, analysis_result, result_video_path
    global camera_source
    
    print("\n" + "="*60)
    print("PIR 센서 트리거 감지!")
    print("="*60)
    
    # 이미 녹화 중이거나 카운트다운 중이면 무시
    if recording or countdown > 0:
        print("(경고) 이미 녹화 중 - 트리거 무시")
        return jsonify({
            'status': 'busy',
            'message': 'Already recording or counting down'
        })
    
    print(f"카메라 소스: {camera_source}")
    
    # 초기화
    keypoints_buffer = []
    frames_buffer = []
    analysis_result = None
    result_video_path = None
    
    # 자동 녹화 시퀀스 (3초 카운트다운 + 10초 녹화)
    def auto_recording_sequence():
        global countdown, recording
        
        print("3초 카운트다운 시작!")
        # 3초 카운트다운
        for i in range(3, 0, -1):
            countdown = i
            print(f"{i}...")
            time.sleep(1)
        
        countdown = 0
        recording = True
        print("녹화 시작!")
        
        # 10초 녹화
        time.sleep(10)
        print("녹화 종료 (10초 경과)")
        
        # 녹화 중지 및 분석
        if recording:
            recording = False
            print("분석 시작")
            run_pipeline()
    
    # 백그라운드에서 실행
    thread = threading.Thread(target=auto_recording_sequence)
    thread.daemon = True
    thread.start()
    
    print("자동 녹화 스레드 시작됨")
    print("="*60 + "\n")
    
    return jsonify({
        'status': 'success',
        'message': 'Auto recording triggered by PIR sensor',
        'source': camera_source,
        'duration': '13 seconds (3s countdown + 10s recording)'
    })






if __name__ == '__main__':
    print("\n" + "="*60)
    print("보행 분석 시스템 시작")
    print("="*60)
    print(f"\n녹화 폴더: {RECORDINGS_FOLDER}")
    print(f"결과 폴더: {RESULTS_FOLDER}")
    print("\n브라우저에서 http://localhost:5000 접속\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)