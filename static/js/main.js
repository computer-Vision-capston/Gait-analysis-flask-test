// ============================================
// 보행 분석 시스템 JavaScript
// ============================================

let isRecording = false;
let checkInterval = null;
let originalVideoSrc = null;
let recordingTimer = null;
let secondsLeft = 10;
let recordingCheckInterval = null;

// ============================================
// 녹화 제어 함수
// ============================================

function startRecording() {
    fetch('/start_recording', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                isRecording = true;
                updateUI('countdown');
                
                // 3초 후 녹화 시작 → 10초 타이머 시작
                setTimeout(() => {
                    if (isRecording) {
                        startRecordingTimer();
                        startRecordingCheck();
                    }
                }, 3000);
            }
        })
        .catch(error => console.error('Error:', error));
}

function startRecordingTimer() {
    secondsLeft = 10;
    updateUI('recording');
    
    recordingTimer = setInterval(() => {
        secondsLeft--;
        updateRecordingStatus();
        
        if (secondsLeft <= 0) {
            clearInterval(recordingTimer);
        }
    }, 1000);
}

function startRecordingCheck() {
    // 서버 상태를 주기적으로 체크 (0.5초마다)
    recordingCheckInterval = setInterval(() => {
        fetch('/get_recording_status')
            .then(response => response.json())
            .then(data => {
                // 서버에서 녹화가 종료되고 분석이 시작되었는지 확인
                if (!data.is_recording && data.is_analyzing) {
                    clearInterval(recordingCheckInterval);
                    clearInterval(recordingTimer);
                    
                    isRecording = false;
                    updateUI('analyzing');
                    startCheckingResult();
                }
            })
            .catch(error => console.error('Error:', error));
    }, 500);
}


function stopRecording() {
    // 수동 중지
    clearInterval(recordingTimer);
    clearInterval(recordingCheckInterval);
    
    fetch('/stop_recording', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                isRecording = false;
                updateUI('analyzing');
                startCheckingResult();
            }
        })
        .catch(error => console.error('Error:', error));
}

function resetSystem() {
    // 모든 타이머 정리
    clearInterval(recordingTimer);
    clearInterval(recordingCheckInterval);
    clearInterval(checkInterval);
    
    fetch('/reset', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                isRecording = false;
                secondsLeft = 10;
                updateUI('idle');
                showResultSection('waiting');
                
                // 비디오 피드를 실시간 카메라로 복구
                const videoFeed = document.getElementById('video-feed');
                if (originalVideoSrc) {
                    videoFeed.src = originalVideoSrc;
                    console.log('✅ 실시간 카메라로 복구됨');
                } else {
                    // 원본이 없으면 기본 경로로 재설정
                    const timestamp = new Date().getTime();
                    videoFeed.src = `/video_feed?t=${timestamp}`;
                    console.log('✅ 비디오 피드 재시작됨');
                }
            }
        })
        .catch(error => console.error('Error:', error));
}

// ============================================
// 결과 확인
// ============================================

function startCheckingResult() {
    checkInterval = setInterval(() => {
        fetch('/get_result')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'completed') {
                    clearInterval(checkInterval);
                    
                    // 결과 영상이 있으면 비디오 피드를 analyzed 영상으로 전환
                    if (data.has_video) {
                        switchToResultVideo();
                    }
                    
                    // 결과 표시
                    displayResult(data.result);
                    updateUI('completed');
                }
            })
            .catch(error => console.error('Error:', error));
    }, 1000);
}

function switchToResultVideo() {
    const videoFeed = document.getElementById('video-feed');
    
    // 원본 소스 저장 (처음 한 번만)
    if (!originalVideoSrc) {
        originalVideoSrc = videoFeed.src;
    }
    
    // 결과 영상으로 전환
    const timestamp = new Date().getTime();
    videoFeed.src = `/result_video_feed?t=${timestamp}`;
    
    console.log('✅ 결과 영상으로 전환됨');
}

// ============================================
// 결과 표시
// ============================================

function displayResult(result) {
    const displayDiv = document.getElementById('result-display');
    
    let html = `
        <div class="result-item">
            <h3>⏱️ 분석 정보</h3>
            <div class="result-detail">
                <strong>분석 시각:</strong>
                <span>${result.timestamp}</span>
            </div>
            <div class="result-detail">
                <strong>총 프레임:</strong>
                <span>${result.total_frames} frames</span>
            </div>
        </div>
    `;

    // 낙상 감지 결과
    const fall = result.fall_detection;
    const fallBadge = fall.is_fall 
        ? `<span class="badge danger">🚨 낙상 감지!</span>`
        : `<span class="badge success">✅ 낙상 없음</span>`;
    
    html += `
        <div class="result-item">
            <h3>1️⃣ 낙상 감지</h3>
            <div class="result-detail">
                <strong>결과:</strong>
                ${fallBadge}
            </div>
            <div class="result-detail">
                <strong>신뢰도:</strong>
                <span>${(fall.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="result-detail">
                <strong>상세:</strong>
                <span>${fall.reason}</span>
            </div>
        </div>
    `;

    // 보행 분류 결과
    if (result.gait_classification) {
        const gait = result.gait_classification;
        
        if (gait.error) {
            html += `
                <div class="result-item">
                    <h3>2️⃣ 보행 분류</h3>
                    <div class="result-detail">
                        <strong>오류:</strong>
                        <span class="badge warning">${gait.error}</span>
                    </div>
                </div>
            `;
        } else {
            const gaitBadge = gait.prediction === 0
                ? `<span class="badge success">정상 보행</span>`
                : `<span class="badge warning">비정상 보행</span>`;
            
            html += `
                <div class="result-item">
                    <h3>2️⃣ 보행 분류</h3>
                    <div class="result-detail">
                        <strong>결과:</strong>
                        ${gaitBadge}
                    </div>
                    <div class="result-detail">
                        <strong>비정상 확률:</strong>
                        <span>${(gait.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div class="result-detail">
                        <strong>정상 확률:</strong>
                        <span>${((1 - gait.confidence) * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `;
        }
    } else {
        html += `
            <div class="result-item">
                <h3>2️⃣ 보행 분류</h3>
                <div class="result-detail">
                    <span class="badge info">낙상이 감지되어 보행 분류를 실행하지 않았습니다</span>
                </div>
            </div>
        `;
    }

    displayDiv.innerHTML = html;
    showResultSection('display');
}

// ============================================
// UI 업데이트
// ============================================

function updateUI(state) {
    const statusDiv = document.getElementById('status');
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    const btnReset = document.getElementById('btn-reset');

    if (state === 'countdown') {
        statusDiv.className = 'status recording';
        statusDiv.textContent = '⏱️ 3초 후 녹화 시작...';
        btnStart.disabled = true;
        btnStop.disabled = true;
        btnReset.disabled = true;
        
        // 3초 후 자동으로 녹화 상태로 변경
        setTimeout(() => {
            if (isRecording) {
                updateUI('recording');
            }
        }, 3000);
    } else if (state === 'recording') {
        statusDiv.className = 'status recording';
        statusDiv.textContent = '🔴 녹화 중...';
        btnStart.disabled = true;
        btnStop.disabled = false;
        btnReset.disabled = true;
    } else if (state === 'analyzing') {
        statusDiv.className = 'status analyzing';
        statusDiv.textContent = '⏳ 분석 중...';
        btnStart.disabled = true;
        btnStop.disabled = true;
        btnReset.disabled = true;
        showResultSection('processing');
    } else if (state === 'completed') {
        statusDiv.className = 'status completed';
        statusDiv.textContent = '✅ 분석 완료! (결과 영상 재생 중)';
        btnStart.disabled = false;
        btnStop.disabled = true;
        btnReset.disabled = false;
    } else { // idle
        statusDiv.className = 'status idle';
        statusDiv.textContent = '대기 중 - 녹화 시작 버튼을 누르세요';
        btnStart.disabled = false;
        btnStop.disabled = true;
        btnReset.disabled = false;
    }
}


function showResultSection(section) {
    document.getElementById('result-waiting').classList.remove('show');
    document.getElementById('result-processing').classList.remove('show');
    document.getElementById('result-display').classList.remove('show');
    
    document.getElementById(`result-${section}`).classList.add('show');
}

let currentCameraSource = 'local';
let raspberryConnected = false;

// 페이지 로드 시
window.addEventListener('DOMContentLoaded', () => {
    // 초기 상태 설정
    document.getElementById('camera-source').value = 'local';
    document.getElementById('raspberry-ip-section').style.display = 'none';
    document.getElementById('camera-status').className = 'camera-status connected';
    document.getElementById('camera-status').textContent = '✅ 사용 중';
    
    updateCameraStatus();
    setInterval(updateCameraStatus, 5000);
});

function updateCameraStatus() {
    fetch('/camera/status')
        .then(response => response.json())
        .then(data => {
            currentCameraSource = data.current_source;
            
            const statusSpan = document.getElementById('camera-status');
            const select = document.getElementById('camera-source');
            
            // select 값 동기화
            select.value = data.current_source;
            
            if (data.current_source === 'local') {
                statusSpan.className = 'camera-status connected';
                statusSpan.textContent = '✅ 사용 중';
            } else if (data.current_source === 'raspberry') {
                if (data.raspberry.available) {
                    statusSpan.className = 'camera-status connected';
                    statusSpan.textContent = '✅ 연결됨';
                    raspberryConnected = true;
                } else {
                    statusSpan.className = 'camera-status disconnected';
                    statusSpan.textContent = '❌ 연결 끊김';
                    raspberryConnected = false;
                    // 라즈베리파이 끊기면 자동으로 local로
                    switchCamera('local');
                }
            }
        })
        .catch(error => console.error('Error checking camera status:', error));
}

function onCameraSourceChange() {
    const select = document.getElementById('camera-source');
    const newSource = select.value;
    const ipSection = document.getElementById('raspberry-ip-section');
    const statusSpan = document.getElementById('camera-status');
    
    if (newSource === 'raspberry') {
        ipSection.style.display = 'flex';
        
        if (raspberryConnected) {
            switchCamera('raspberry');
        } else {
            statusSpan.className = 'camera-status disconnected';
            statusSpan.textContent = '⚠️ 연결 필요';
        }
    } else {
        ipSection.style.display = 'none';
        switchCamera('local');
    }
}

function connectRaspberry() {
    const ipInput = document.getElementById('raspberry-ip');
    const ip = ipInput.value.trim();
    const statusSpan = document.getElementById('camera-status');
    const connectBtn = document.querySelector('.btn-connect');
    
    if (!ip) {
        alert('IP 주소를 입력하세요');
        return;
    }
    
    statusSpan.className = 'camera-status connecting';
    statusSpan.textContent = '⏳ 연결 중...';
    connectBtn.disabled = true;
    
    fetch('/raspberry/connect', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ip: ip})
    })
    .then(response => response.json())
    .then(data => {
        connectBtn.disabled = false;
        
        if (data.status === 'success') {
            statusSpan.className = 'camera-status connected';
            statusSpan.textContent = '✅ 연결 성공';
            raspberryConnected = true;
            switchCamera('raspberry');
        } else {
            statusSpan.className = 'camera-status disconnected';
            statusSpan.textContent = '❌ 연결 실패';
            alert(data.message);
        }
    })
    .catch(error => {
        connectBtn.disabled = false;
        statusSpan.className = 'camera-status disconnected';
        statusSpan.textContent = '❌ 오류';
        alert('연결 오류: ' + error);
    });
}

function switchCamera(source) {
    const videoFeed = document.getElementById('video-feed');
    const statusSpan = document.getElementById('camera-status');
    
    console.log('Switching to:', source);
    
    fetch('/camera/select', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({source: source})
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            currentCameraSource = source;
            
            // 기존 영상 완전히 끊기
            videoFeed.src = '';
            
            // 짧은 딜레이 후 새 소스 설정
            setTimeout(() => {
                const timestamp = new Date().getTime();
                if (source === 'local') {
                    videoFeed.src = '/video_feed?' + timestamp;
                    statusSpan.className = 'camera-status connected';
                    statusSpan.textContent = '✅ 사용 중';
                } else {
                    videoFeed.src = '/raspberry/video_feed?' + timestamp;
                    statusSpan.className = 'camera-status connected';
                    statusSpan.textContent = '✅ 연결됨';
                }
            }, 100);
            
            console.log('Camera switched to:', source);
        } else {
            alert('카메라 전환 실패: ' + data.message);
            document.getElementById('camera-source').value = currentCameraSource;
        }
    })
    .catch(error => {
        console.error('Error switching camera:', error);
        alert('카메라 전환 중 오류 발생');
        document.getElementById('camera-source').value = currentCameraSource;
    });
}