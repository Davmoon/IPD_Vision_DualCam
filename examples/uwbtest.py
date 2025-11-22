import serial
ser = serial.Serial('/dev/serial0', 115200, timeout=1)
ser.write(b'lec\r\n')
print(ser.readline())#