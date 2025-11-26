// 히스토리 js 

let currentFilter = 'all';

// 페이지 로드 시 실행
window.addEventListener('DOMContentLoaded', () => {
    checkFirebaseConnection();
    loadHistory('all');
});


// Firebase 연결 확인


function checkFirebaseConnection() {
    const statusDiv = document.getElementById('firebase-status');
    
    fetch('/check_firebase')
        .then(response => response.json())
        .then(data => {
            if (data.enabled) {
                statusDiv.className = 'firebase-status connected';
                statusDiv.innerHTML = `
                    <span class="status-icon">✅</span>
                    <span class="status-text">Firebase 연결됨</span>
                `;
            } else {
                statusDiv.className = 'firebase-status disconnected';
                statusDiv.innerHTML = `
                    <span class="status-icon">❌</span>
                    <span class="status-text">Firebase 비활성화</span>
                `;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            statusDiv.className = 'firebase-status disconnected';
            statusDiv.innerHTML = `
                <span class="status-icon">⚠️</span>
                <span class="status-text">연결 오류</span>
            `;
        });
}

// 기록 로드

function filterHistory(type) {
    currentFilter = type;
    
    // 버튼 active 클래스 변경
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    loadHistory(type);
}

function loadHistory(type) {
    const listDiv = document.getElementById('history-list');
    listDiv.innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <p>기록을 불러오는 중...</p>
        </div>
    `;
    
    fetch(`/get_history?type=${type}&limit=50`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                displayHistory(data.records);
            } else {
                listDiv.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">⚠️</div>
                        <h3>오류 발생</h3>
                        <p>${data.message}</p>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            listDiv.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">❌</div>
                    <h3>로딩 실패</h3>
                    <p>네트워크 오류가 발생했습니다</p>
                </div>
            `;
        });
}

function displayHistory(records) {
    const listDiv = document.getElementById('history-list');
    
    if (records.length === 0) {
        listDiv.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📂</div>
                <h3>기록이 없습니다</h3>
                <p>분석된 기록이 없습니다. 메인 페이지에서 분석을 시작하세요.</p>
            </div>
        `;
        return;
    }
    
    let html = '<div class="history-grid">';
    
    records.forEach(record => {
        const data = record.data;
        const badge = getResultBadge(data.result_type);
        
        // 썸네일 아이콘 (타입별)
        let thumbnailIcon = '🎥';
        if (data.result_type === 'fall') thumbnailIcon = '🚨';
        else if (data.result_type === 'abnormal') thumbnailIcon = '⚠️';
        else if (data.result_type === 'normal') thumbnailIcon = '✅';
        
        html += `
            <div class="history-card" onclick="viewDetail('${record.id}')">
                <div class="history-thumbnail" style="display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-size: 4em;">
                    ${thumbnailIcon}
                </div>
                <div class="history-info">
                    <div class="history-badge">${badge}</div>
                    <div class="history-time">📅 ${data.timestamp || 'N/A'}</div>
                    <div class="history-frames">🎞️ ${data.total_frames || 0} frames</div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    listDiv.innerHTML = html;
}

function getResultBadge(type) {
    if (type === 'fall') return '<span class="badge danger">🔴 낙상</span>';
    if (type === 'abnormal') return '<span class="badge warning">🟠 비정상</span>';
    if (type === 'normal') return '<span class="badge success">🟢 정상</span>';
    return '<span class="badge">알 수 없음</span>';
}

// ============================================
// 상세 보기
// ============================================

function viewDetail(docId) {
    document.getElementById('detail-modal').style.display = 'block';
    document.body.style.overflow = 'hidden';
    
    fetch(`/get_record/${docId}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                displayDetail(data.record, docId);
            } else {
                alert('기록을 불러올 수 없습니다: ' + data.message);
                closeDetail();
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('기록 로딩 실패');
            closeDetail();
        });
}

function closeDetail() {
    document.getElementById('detail-modal').style.display = 'none';
    document.body.style.overflow = 'auto';
}

function displayDetail(record, docId) {
    const contentDiv = document.getElementById('detail-content');
    
    const fall = record.fall_detection || {};
    const gait = record.gait_classification || {};
    
    let html = `
        <div class="detail-section">
            <h3>분석 영상</h3>
            <video controls width="100%" style="max-width: 640px; border-radius: 10px;">
                <source src="${record.analyzed_video_url || ''}" type="video/mp4">
                브라우저가 비디오를 지원하지 않습니다.
            </video>
        </div>
        
        <div class="detail-section">
            <h3>분석 결과</h3>
            <div class="result-item">
                <div class="result-detail">
                    <strong>분석 시각:</strong>
                    <span>${record.timestamp || 'N/A'}</span>
                </div>
                <div class="result-detail">
                    <strong>총 프레임:</strong>
                    <span>${record.total_frames || 0} frames</span>
                </div>
                <div class="result-detail">
                    <strong>결과 타입:</strong>
                    ${getResultBadge(record.result_type)}
                </div>
            </div>
            
            <div class="result-item">
                <h4>(1) 낙상 감지</h4>
                <div class="result-detail">
                    <strong>낙상 여부:</strong>
                    <span>${fall.is_fall ? '낙상 감지' : '낙상 없음'}</span>
                </div>
                <div class="result-detail">
                    <strong>신뢰도:</strong>
                    <span>${((fall.confidence || 0) * 100).toFixed(1)}%</span>
                </div>
                <div class="result-detail">
                    <strong>상세:</strong>
                    <span>${fall.reason || 'N/A'}</span>
                </div>
            </div>
    `;
    
    if (gait && !gait.error && gait.prediction !== undefined) {
        html += `
            <div class="result-item">
                <h4>(2) 보행 분류</h4>
                <div class="result-detail">
                    <strong>결과:</strong>
                    <span>${gait.prediction === 0 ? '정상 보행' : '비정상 보행'}</span>
                </div>
                <div class="result-detail">
                    <strong>비정상 확률:</strong>
                    <span>${((gait.confidence || 0) * 100).toFixed(1)}%</span>
                </div>
                <div class="result-detail">
                    <strong>정상 확률:</strong>
                    <span>${((1 - (gait.confidence || 0)) * 100).toFixed(1)}%</span>
                </div>
            </div>
        `;
    } else if (gait && gait.error) {
        html += `
            <div class="result-item">
                <h4>(2) 보행 분류</h4>
                <div class="result-detail">
                    <strong>오류:</strong>
                    <span class="badge warning">${gait.error}</span>
                </div>
            </div>
        `;
    }
    
    html += `
        </div>
        
        <div class="detail-section">
            <h3>다운로드</h3>
            <div class="download-links">
                <a href="${record.original_video_url || '#'}" target="_blank" class="btn-download" ${!record.original_video_url ? 'style="pointer-events:none;opacity:0.5"' : ''}>
                    원본 영상
                </a>
                <a href="${record.analyzed_video_url || '#'}" target="_blank" class="btn-download" ${!record.analyzed_video_url ? 'style="pointer-events:none;opacity:0.5"' : ''}>
                    분석 영상
                </a>
            </div>
        </div>
        
        <div class="detail-section">
            <button class="btn-delete" onclick="deleteRecord('${docId}')">
                기록 삭제
            </button>
        </div>
    `;
    
    contentDiv.innerHTML = html;
}

function deleteRecord(docId) {
    if (!confirm('이 기록을 삭제하시겠습니까?\n(영상 파일도 함께 삭제됩니다)')) {
        return;
    }
    
    fetch(`/delete_record/${docId}`, { method: 'DELETE' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                alert('기록이 삭제되었습니다');
                closeDetail();
                loadHistory(currentFilter);
            } else {
                alert('삭제 실패: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('삭제 실패');
        });
}


// 기타

function goBack() {
    window.location.href = '/';
}

// 모달 외부 클릭 시 닫기
window.onclick = function(event) {
    const modal = document.getElementById('detail-modal');
    if (event.target === modal) {
        closeDetail();
    }
}

// ESC 키로 모달 닫기
document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
        closeDetail();
    }
});