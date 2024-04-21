# 360-degree real-time rotational camera
## Stepper Motor Control by Raspberry Pi GPIO
### Hardware requirements:
1. Stepper motor
2. Raspberry Pi
3. Dupont cable
4. Driver module

### Connecting a Raspberry Pi to a Stepper Motor
The following figure shows the connection diagram for the Raspberry Pi to control a stepper motor:

![alt text](figures/raspberry-pi-pico-stepper-motor-circuit-diagram.png)

**Pin31, pin32, pin35, and pin37 of the Raspberry Pi are connected to IN1, IN2, IN3, and IN4 of the driver, respectively.**

The positive and negative terminals of the power supply on the driver are connected to +5V on the Raspberry Pi and to ground, respectively

![alt text](figures/GPIO-Pinout-Diagram-2.png)

### Running Python Code on a Raspberry Pi
```python
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

 
IN1 = 31    # pin31
IN2 = 33    # pin33
IN3 = 35    # pin35
IN4 = 37    # pin37
 
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
		backward(0.003, 512)  # 512 steps --- 360 angle
		
		print ("stop...")
		stop()                 # stop
		time.sleep(3)          # sleep 3s
		
		print ("forward...")
		forward(0.003, 512)
		
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
```  

**Note** that here a **MAC** system is used to run the python program on the Raspberry Pi remotely. Since we need to use the keyboard to control the direction of rotation of the motor, this function here is a control command for the **MAC OS** system:
```python
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
```

If it needed to run real-time keyboard input in a Windows environment, you need to replace this function with:
```python
import msvcrt

def getch():
    """Get a single keypress without echoing it to the console."""
    print("Press a key:", end="", flush=True)  # Prompt the user to press a key without a newline.
    ch = msvcrt.getch()  # Read a single keypress without echoing.
    print("\nYou pressed:", ch.decode())  # Print the keypress after decoding it to a string.
    return ch

# Call the getch function
getch()

```
## Run the Program
When you ssh the raspberry Pi and run the program:
```bash
ssh user_name@IP_Address
python3 stepper_motor.py
```

While the program running, you should find some prompts in your command window:
```bash
Press any key to continue...
```
Then you just need to press **"A"** or **"D"** bottom to make camera **LEFT** or **RIGHT** Trun
```bash
You pressed: a
```
```bash
You pressed: d
```
It will detect the bottom of your press and wait for your next press.

### Results Showcase
![Sample GIF](figures/Motor.gif)

### System Shown in Autodron
![Sample GIF](figures/system.gif)
