"""
Firebase API 관리
- Storage: 영상 업로드 (원본, 분석)
- Firestore: 분석 결과 및 키포인트 저장
"""
import firebase_admin
from firebase_admin import credentials, storage, firestore
import os

class FirebaseAPI:
    def __init__(self, cred_path='firebase-credentials.json', bucket_name='capstone-3d5ef.firebasestorage.app'):
        """Firebase 초기화"""
        if not firebase_admin._apps:
            if not os.path.exists(cred_path):
                raise FileNotFoundError(f"Firebase 인증 파일을 찾을 수 없습니다: {cred_path}")
            
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                'storageBucket': bucket_name
            })
            print(f"✅ Firebase 초기화 완료 - Bucket: {bucket_name}")
        
        self.bucket = storage.bucket()
        self.db = firestore.client()
    
    def upload_video(self, local_path, storage_path):
        """영상을 Firebase Storage에 업로드"""
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"업로드할 파일을 찾을 수 없습니다: {local_path}")
        
        blob = self.bucket.blob(storage_path)
        blob.upload_from_filename(local_path)
        blob.make_public()
        
        print(f"  ✅ Storage 업로드: {storage_path}")
        return blob.public_url
    
    def save_analysis_result(self, data):
        """분석 결과를 Firestore에 저장"""
        doc_ref = self.db.collection('gait_analysis').document()
        doc_ref.set(data)
        
        print(f"  ✅ Firestore 저장: {doc_ref.id}")
        return doc_ref.id
    
    def get_all_records(self, limit=50):
        """모든 분석 기록 조회 (최신순)"""
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
        """특정 타입의 분석 기록 조회"""
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
        """특정 ID의 분석 기록 조회"""
        doc = self.db.collection('gait_analysis').document(doc_id).get()
        return doc.to_dict() if doc.exists else None
    
    def delete_record(self, doc_id):
        """분석 기록 삭제 (Firestore 문서 + Storage 영상 파일)"""
        doc = self.db.collection('gait_analysis').document(doc_id).get()
        
        if not doc.exists:
            print(f"❌ 기록 없음: {doc_id}")
            return
        
        data = doc.to_dict()
        
        # Storage에서 영상 삭제
        try:
            if 'original_video_url' in data:
                self._delete_from_url(data['original_video_url'])
            if 'analyzed_video_url' in data:
                self._delete_from_url(data['analyzed_video_url'])
        except Exception as e:
            print(f"⚠️ Storage 삭제 오류: {e}")
        
        # Firestore 문서 삭제
        self.db.collection('gait_analysis').document(doc_id).delete()
        print(f"✅ 기록 삭제: {doc_id}")
    
    def _delete_from_url(self, public_url):
        """Storage URL에서 파일 삭제"""
        path = public_url.split(self.bucket.name + '/')[-1]
        blob = self.bucket.blob(path)
        blob.delete()
        print(f"  - Storage 파일 삭제: {path}")


if __name__ == '__main__':
    try:
        firebase = FirebaseAPI()
        print("\n✅ Firebase 연결 성공!")
        records = firebase.get_all_records(limit=5)
        print(f"📊 저장된 기록: {len(records)}개")
    except Exception as e:
        print(f"\n❌ Firebase 연결 실패: {e}")