import RPi.GPIO as GPIO
import time
import sys, termios, tty

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

 
IN1 = 31   # pin11
IN2 = 33
IN3 = 35
IN4 = 37
 
def setStep(w1, w2, w3, w4):
	GPIO.output(IN1, w1)
	GPIO.output(IN2, w2)
	GPIO.output(IN3, w3)
	GPIO.output(IN4, w4)
 
def stop():
	setStep(0, 0, 0, 0)
 
def forward(delay, steps):  
	for i in range(0, steps):
		setStep(1, 0, 0, 0)
		time.sleep(delay)
		setStep(0, 1, 0, 0)
		time.sleep(delay)
		setStep(0, 0, 1, 0)
		time.sleep(delay)
		setStep(0, 0, 0, 1)
		time.sleep(delay)
 
def backward(delay, steps):  
	for i in range(0, steps):
		setStep(0, 0, 0, 1)
		time.sleep(delay)
		setStep(0, 0, 1, 0)
		time.sleep(delay)
		setStep(0, 1, 0, 0)
		time.sleep(delay)
		setStep(1, 0, 0, 0)
		time.sleep(delay)
 
def setup():
	GPIO.setwarnings(False)
	GPIO.setmode(GPIO.BOARD)       # Numbers GPIOs by physical location
	GPIO.setup(IN1, GPIO.OUT)      # Set pin's mode is output
	GPIO.setup(IN2, GPIO.OUT)
	GPIO.setup(IN3, GPIO.OUT)
	GPIO.setup(IN4, GPIO.OUT)
 
def loop():
	while True:
		print ("backward...")
		backward(0.0025, 512)  # 512 steps --- 360 angle
		
		print ("stop...")
		stop()                 # stop
		time.sleep(3)          # sleep 3s
		
		print ("forward...")
		forward(0.0025, 512)
		
		print ("stop...")
		stop()
		time.sleep(3)
 
def destroy():
	GPIO.cleanup()             # Release resource
 
if __name__ == '__main__':     # Program start from here
    setup()
    try:
        while True:
            print("Press any key to continue...")
            dir = getch()
            print("You pressed:", dir)
            if dir == 'd':
                backward(0.003, 16)
            elif dir == 'a':
                forward(0.003, 16)
            elif dir == 'q':
                destroy()
                break
            #loop()
    except KeyboardInterrupt:  # When 'Ctrl+C' is pressed, the child function destroy() will be  executed.
        destroy()




