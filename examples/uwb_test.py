import serial
import threading

uwb = serial.Serial('/dev/serial0', baudrate=115200, timeout=0.5)

def read_from_uwb():
    while True:
        if uwb.in_waiting:
            #0xff 같이 이상한 데이터 읽히는 거 방지
            data = uwb.readline().decode(errors='ignore').strip()
            if data:
                print(f"UWB 응담 : {data}")

def write_to_uwb():
    while True:
        try:
            cmd = input(">>> ")
            if cmd.strip():
                uwb.write((cmd + '\r').encode())
        except KeyboardInterrupt:
            print("키보드 연결 오류. 프로그램 종료")
            break

if __name__ == "__main__":
    print("--UWB 시리얼 출력 시작--")
    threading.Thread(target=read_from_uwb, daemon=True).start()
    write_to_uwb()