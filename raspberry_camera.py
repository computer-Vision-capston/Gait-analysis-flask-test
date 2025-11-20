"""
라즈베리파이 카메라 제어 모듈
PC에서 라즈베리파이 카메라를 원격 제어
"""
import requests
import base64
import numpy as np
import cv2


class RaspberryPiCamera:
    def __init__(self, raspberry_pi_url):
        """
        Args:
            raspberry_pi_url: 라즈베리파이 주소 (예: 'http://192.168.0.50:8000')
        """
        self.url = raspberry_pi_url
        self.connected = False
        self.check_connection()
    
    def check_connection(self):
        """라즈베리파이 연결 확인"""
        try:
            response = requests.get(f"{self.url}/", timeout=3)
            if response.status_code == 200:
                self.connected = True
                print(f"✅ 라즈베리파이 연결됨: {self.url}")
                return True
        except Exception as e:
            self.connected = False
            print(f"⚠️ 라즈베리파이 연결 실패: {e}")
        return False
    
    def start_recording(self):
        """라즈베리파이에 녹화 시작 명령"""
        try:
            response = requests.post(f"{self.url}/start_recording", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"📹 라즈베리파이 녹화 시작: {data}")
                return True, data
            return False, {'error': f'Status code: {response.status_code}'}
        except Exception as e:
            print(f"❌ 녹화 시작 실패: {e}")
            return False, {'error': str(e)}
    
    def stop_recording_and_get_frames(self):
        """라즈베리파이 녹화 중지 및 프레임 수신"""
        try:
            print("⏹️ 라즈베리파이 녹화 중지 및 프레임 수신 중...")
            response = requests.post(f"{self.url}/stop_recording", timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] != 'success':
                    return False, None, data.get('message', 'Unknown error')
                
                encoded_frames = data.get('frames', [])
                frame_count = data.get('frame_count', 0)
                
                print(f"📥 {frame_count}개 프레임 수신됨")
                
                # base64 디코딩하여 numpy 배열로 변환
                frames = []
                for encoded in encoded_frames:
                    img_data = base64.b64decode(encoded)
                    nparr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        frames.append(frame)
                
                print(f"✅ {len(frames)}개 프레임 디코딩 완료")
                return True, frames, None
            
            return False, None, f'Status code: {response.status_code}'
        
        except Exception as e:
            print(f"❌ 프레임 수신 실패: {e}")
            return False, None, str(e)
    
    def get_status(self):
        """라즈베리파이 상태 조회"""
        try:
            response = requests.get(f"{self.url}/status", timeout=3)
            if response.status_code == 200:
                return True, response.json()
            return False, {'error': f'Status code: {response.status_code}'}
        except Exception as e:
            return False, {'error': str(e)}
    
    def reset(self):
        """라즈베리파이 카메라 리셋"""
        try:
            response = requests.post(f"{self.url}/reset", timeout=5)
            if response.status_code == 200:
                return True, response.json()
            return False, {'error': f'Status code: {response.status_code}'}
        except Exception as e:
            return False, {'error': str(e)}
    
    def get_video_feed_url(self):
        """라즈베리파이 비디오 스트림 URL 반환"""
        return f"{self.url}/video_feed"