/*
 * PIR 센서 + WiFi + Flask 자동 녹화 (Arduino UNO R4 WiFi 전용)
 * 
 * 연결:
 * PIR VCC -> 5V
 * PIR GND -> GND
 * PIR OUT -> D2
 */

#include <WiFiS3.h>  // UNO R4 WiFi 전용 라이브러리

// ============================================
// 설정 부분 (여기만 수정하세요!)
// ============================================

// WiFi 정보
char ssid[] = "U+NetACB3";           // 와이파이 이름
char password[] = "4000019202";    // 와이파이 비밀번호

// Flask 서버 정보
const char* serverIP = "192.168.219.112";  // PC의 IP 주소 (ipconfig로 확인)
const int serverPort = 5000;              // Flask 포트

// PIR 센서 핀
const int PIR_PIN = 2;

// ============================================
// 변수 선언
// ============================================

WiFiClient client;

// PIR 상태
int pirState = LOW;
int lastState = LOW;

// 쿨다운
unsigned long lastTriggerTime = 0;
const unsigned long COOLDOWN = 15000;  // 15초

// 트리거 카운트
int triggerCount = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n\n========================================");
  Serial.println("  PIR 센서 + Flask 자동 녹화 시스템");
  Serial.println("  (Arduino UNO R4 WiFi)");
  Serial.println("========================================\n");
  
  // PIR 센서 초기화
  pinMode(PIR_PIN, INPUT);
  Serial.println("[1/3] PIR 센서 초기화 완료");
  
  // WiFi 연결
  Serial.print("[2/3] WiFi 연결 중: ");
  Serial.println(ssid);
  
  int status = WL_IDLE_STATUS;
  
  // WiFi 연결 시도
  while (status != WL_CONNECTED) {
    Serial.print("   연결 시도 중...");
    status = WiFi.begin(ssid, password);
    delay(5000);  // 5초 대기
  }
  
  Serial.println("\n✅ WiFi 연결 성공!");
  Serial.print("   IP 주소: ");
  Serial.println(WiFi.localIP());
  Serial.print("   Flask 서버: ");
  Serial.print(serverIP);
  Serial.print(":");
  Serial.println(serverPort);
  
  // PIR 센서 안정화
  Serial.println("[3/3] PIR 센서 안정화 중... (30초)");
  for (int i = 30; i > 0; i--) {
    Serial.print("   ");
    Serial.print(i);
    Serial.println("초 남음...");
    delay(1000);
  }
  
  Serial.println("\n========================================");
  Serial.println("  ✓ 시스템 준비 완료!");
  Serial.println("  사람 감지 시 Flask 서버에 요청합니다.");
  Serial.println("========================================\n");
}

void loop() {
  // WiFi 연결 확인
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("⚠️ WiFi 연결 끊김 - 재연결 시도...");
    WiFi.begin(ssid, password);
    delay(5000);
    return;
  }
  
  // PIR 센서 읽기
  pirState = digitalRead(PIR_PIN);
  unsigned long currentTime = millis();
  
  // LOW → HIGH (사람 감지)
  if (pirState == HIGH && lastState == LOW) {
    
    // 쿨다운 체크
    if (currentTime - lastTriggerTime > COOLDOWN) {
      Serial.println("\n========================================");
      Serial.println("🚨 사람 감지!");
      Serial.println("========================================");
      
      triggerCount++;
      Serial.print("트리거 횟수: ");
      Serial.println(triggerCount);
      
      // Flask 서버에 요청
      triggerFlaskRecording();
      
      // 쿨다운 시작
      lastTriggerTime = currentTime;
      Serial.println("\n⏳ 15초 쿨다운 시작...");
      Serial.println("========================================\n");
      
    } else {
      // 쿨다운 중
      unsigned long remaining = (COOLDOWN - (currentTime - lastTriggerTime)) / 1000;
      Serial.print("⏳ 쿨다운 중... (");
      Serial.print(remaining);
      Serial.println("초 남음)");
    }
    
    lastState = HIGH;
    
  } else if (pirState == LOW && lastState == HIGH) {
    Serial.println("✓ 감지 종료\n");
    lastState = LOW;
  }
  
  // 상태 표시 (10초마다)
  static unsigned long lastStatusTime = 0;
  if (currentTime - lastStatusTime > 10000) {
    if (currentTime - lastTriggerTime > COOLDOWN) {
      Serial.println("[ 대기 중 - 감지 가능 ]");
    } else {
      unsigned long remaining = (COOLDOWN - (currentTime - lastTriggerTime)) / 1000;
      Serial.print("[ 쿨다운: ");
      Serial.print(remaining);
      Serial.println("초 남음 ]");
    }
    lastStatusTime = currentTime;
  }
  
  delay(100);
}

void triggerFlaskRecording() {
  Serial.println("📡 Flask 서버에 요청 전송 중...");
  
  // 서버에 연결
  Serial.print("   연결 시도: ");
  Serial.print(serverIP);
  Serial.print(":");
  Serial.println(serverPort);
  
  if (client.connect(serverIP, serverPort)) {
    Serial.println("   ✓ 서버 연결 성공");
    
    // HTTP POST 요청 생성
    client.println("POST /trigger_auto_recording HTTP/1.1");
    client.print("Host: ");
    client.println(serverIP);
    client.println("Content-Type: application/json");
    client.println("Content-Length: 2");
    client.println("Connection: close");
    client.println();
    client.println("{}");  // 빈 JSON
    
    // 서버 응답 대기
    unsigned long timeout = millis();
    while (client.available() == 0) {
      if (millis() - timeout > 5000) {
        Serial.println("   ❌ 응답 타임아웃");
        client.stop();
        return;
      }
    }
    
    // 응답 읽기
    Serial.println("   서버 응답:");
    while (client.available()) {
      String line = client.readStringUntil('\r');
      Serial.print("   ");
      Serial.println(line);
    }
    
    client.stop();
    Serial.println("✅ 요청 완료!");
    
  } else {
    Serial.println("   ❌ 서버 연결 실패");
    Serial.println("   Flask 서버가 실행 중인지 확인하세요.");
  }
}
