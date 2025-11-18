"""
Firebase API 관리
- Storage: 영상 업로드 (원본, 분석)
- Firestore: 분석 결과 및 키포인트 저장
"""
import firebase_admin
from firebase_admin import credentials, storage, firestore
from datetime import datetime
import json
import os

class FirebaseAPI:
    def __init__(self, cred_path='capstone-3d5ef-firebase-adminsdk-fbsvc-72fa6f3c7f.json', bucket_name='your-project.appspot.com'):
        """
        Firebase 초기화
        
        Args:
            cred_path: Firebase 서비스 계정 키 JSON 파일 경로
            bucket_name: Firebase Storage 버킷 이름
        """
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                'storageBucket': bucket_name
            })
        
        self.bucket = storage.bucket()
        self.db = firestore.client()
    
    
    def upload_video(self, local_path, storage_path):
        """
        영상을 Firebase Storage에 업로드
        
        Args:
            local_path: 로컬 파일 경로
            storage_path: Storage에 저장될 경로 (예: 'videos/original/xxx.mp4')
        
        Returns:
            str: 다운로드 URL
        """
        blob = self.bucket.blob(storage_path)
        blob.upload_from_filename(local_path)
        
        # 공개 URL 생성
        blob.make_public()
        return blob.public_url
    
    
    def save_analysis_result(self, data):
        """
        분석 결과를 Firestore에 저장
        
        Args:
            data: dict {
                'timestamp': str,
                'result_type': str ('fall', 'normal', 'abnormal'),
                'fall_detection': dict,
                'gait_classification': dict,
                'keypoints': list,  # numpy array를 list로 변환
                'original_video_url': str,
                'analyzed_video_url': str,
                'total_frames': int
            }
        
        Returns:
            str: 문서 ID
        """
        doc_ref = self.db.collection('gait_analysis').document()
        doc_ref.set(data)
        
        print(f"✅ Firestore 저장 완료: {doc_ref.id}")
        return doc_ref.id
    
    
    def get_all_records(self, limit=50):
        """
        모든 분석 기록 조회 (최신순)
        
        Args:
            limit: 조회할 최대 개수
        
        Returns:
            list: [{id, data}, ...]
        """
        docs = self.db.collection('gait_analysis')\
                      .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                      .limit(limit)\
                      .stream()
        
        results = []
        for doc in docs:
            results.append({
                'id': doc.id,
                'data': doc.to_dict()
            })
        
        return results
    
    
    def get_records_by_type(self, result_type, limit=50):
        """
        특정 타입의 분석 기록 조회
        
        Args:
            result_type: 'fall', 'normal', 'abnormal'
            limit: 조회할 최대 개수
        
        Returns:
            list: [{id, data}, ...]
        """
        docs = self.db.collection('gait_analysis')\
                      .where('result_type', '==', result_type)\
                      .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                      .limit(limit)\
                      .stream()
        
        results = []
        for doc in docs:
            results.append({
                'id': doc.id,
                'data': doc.to_dict()
            })
        
        return results
    
    
    def get_record_by_id(self, doc_id):
        """
        특정 ID의 분석 기록 조회
        
        Args:
            doc_id: 문서 ID
        
        Returns:
            dict: 분석 결과 데이터
        """
        doc = self.db.collection('gait_analysis').document(doc_id).get()
        
        if doc.exists:
            return doc.to_dict()
        else:
            return None
    
    
    def delete_record(self, doc_id):
        """
        분석 기록 삭제 (Firestore + Storage 영상)
        
        Args:
            doc_id: 문서 ID
        """
        # Firestore에서 데이터 가져오기
        doc = self.db.collection('gait_analysis').document(doc_id).get()
        
        if doc.exists:
            data = doc.to_dict()
            
            # Storage에서 영상 삭제
            try:
                # URL에서 경로 추출하여 삭제
                if 'original_video_url' in data:
                    self._delete_from_url(data['original_video_url'])
                if 'analyzed_video_url' in data:
                    self._delete_from_url(data['analyzed_video_url'])
            except Exception as e:
                print(f"⚠️ Storage 삭제 오류: {e}")
            
            # Firestore 문서 삭제
            self.db.collection('gait_analysis').document(doc_id).delete()
            print(f"✅ 기록 삭제 완료: {doc_id}")
        else:
            print(f"❌ 기록 없음: {doc_id}")
    
    
    def _delete_from_url(self, public_url):
        """URL에서 Storage 경로 추출하여 삭제"""
        # URL 형식: https://storage.googleapis.com/bucket-name/path/to/file.mp4
        path = public_url.split(self.bucket.name + '/')[-1]
        blob = self.bucket.blob(path)
        blob.delete()