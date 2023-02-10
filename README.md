<h1 align="center">
<b>MECE Fall 2023 Capstone project - Autodrone | Rice University</b>

![Autonomous Drone in Artificial Pollination](https://github.com/Rice-MECE-Capstone-Projects/Autodrone/blob/main/Photos/Autonomous%20Drone%20in%20Artificial%20Pollination.png)

-----------------------------------------------------------------------------------------------

## :computer:Hardware (Embedded systems)
### Getting start with Jetson Nano 2GB Developer Kit:

To begin with Jetson Nano 2GB, go to [this link](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit#write) write image to the microSD card (Before write image to your SD card, make sure your [JetPack version](https://developer.nvidia.com/embedded/jetpack-archive), notice that **Jetson Nano only supports up to JetPack 4.6.3**)

After inserting the microSD card, you can connect the power supply, which will automatically boot up the system.

When you boot the system for the first time, you'll be taken through some initial setup, including:

- Review and accept NVIDIA Jetson software EULA
- Select system language, keyboard layout, and time zone
- Create username, password, and computer name
- Log in

After the initial setup, you should see the following screen:
![Initial screen](https://gilberttanner.com/content/images/2020/08/initial_screen.png)

After successfully logging into the desktop, we recommend completing the following steps for later use:

#### 1. Increasing swap memory:

```
git clone https://github.com/JetsonHacksNano/resizeSwapMemory
cd resizeSwapMemory
 ./setSwapMemorySize.sh -g 4
```

After executing the above command, you'll have to **reboot** the Jetson Nano for the changes to take effect.

#### 2. Installing prerequisites and configuring your Python environment:

Now that the Jetson Nano is ready to go, we will create a deep learning environment. We will start by installing all prerequisites and configuring a Python environment, and how to code remote using VSCode Remote SSH.

```
sudo apt-get update
sudo apt-get upgrade

sudo apt-get install git cmake python3-dev nano

sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev
```

#### 3. Configuring your Python environment:

Next, we will configure our Python environment. This includes downloading pip3 and virtualenv.

```
sudo apt-get install python3-pip
sudo pip3 install -U pip testresources setuptools
sudo pip install virtualenv virtualenvwrapper
```

To get virtualenv to work, we need to add the following lines to the **~/.bashrc** file:

```
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
```

To activate the changes, the following command must be executed:

```
source ~/.bashrc
```

Now we can create a virtual environment using the **mkvirtualenv** command.

```
mkvirtualenv ml -p python3
workon ml
```

#### 4. Coding remote with Visual Studio Code:

If you are like me and hate writing long scripts in nano or vim, the VSCode remote development plugin is for you. It allows you to develop remotely inside VSCode by establishing an SSH remote connection.
To use VSCode remote development, you'll first have to install the remote development plugin. After that, you need to create an SSH-Key on your local machine and then copy it over to the Jetson Nano.

```
# Create Key
ssh-keygen -t rsa
# Copy key to jetson nano
cat ~/.ssh/id_rsa.pub | ssh user@hostname 'cat >> .ssh/authorized_keys'
```

Now you only need to add the SSH Host. Ctrl + Shift + P -> Remote SSH: Connect to Host.

![SSH Host](https://gilberttanner.com/content/images/2020/03/grafik-5.png)
![VSCode SSH](https://gilberttanner.com/content/images/2020/08/vscode_remote_control.PNG)

#### 5. Install jetson-stats:

**jetson-stats** is a package for **monitoring** and **control** your [NVIDIA Jetson](https://developer.nvidia.com/buy-jetson) [Orin, Xavier, Nano, TX] series.

jetson-stats is a powerful tool to analyze your board, you can use with a stand alone application with `jtop` or import in your python script, the main features are:

- Decode hardware, architecture, L4T and NVIDIA Jetpack
- Monitoring, CPU, GPU, Memory, Engines, fan
- Control NVP model, fan speed, jetson_clocks
- Importable in a python script
- Dockerizable in a container
- Do not need super user
- Tested on many different hardware configurations
- Works with all NVIDIA Jetpack

```
sudo pip3 install -U jetson-stats
```

_Don't forget to **logout/login** or **reboot** your board_

Start jtop it's pretty simple just write `jtop`!

```
jtop
```

#### 6. Install OpenCV:
Installing OpenCV on the Jetson Nano can be a bit more complicated, but frankly, [JetsonHacks.com](https://jetsonhacks.com/) has a great guide, or see the tutorial from [Q-engineering](https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html).

-----------------------------------------------------------------------------------------------
## :helicopter:Hardware/Firmware (Drone kit)
### Getting start with Drone kit:
We using [QWinOut 4-Aixs RC Drone](https://www.amazon.com/dp/B082PN8C98?ref_=cm_sw_r_cp_ud_dp_FE0D8ZMAWQRE5JXRX8X8) for this project, and the controller is based on [Ardupilot](https://ardupilot.org/dev/docs/learning-ardupilot-introduction.html). 

![QWinOut 4-Aixs RC Dron](https://m.media-amazon.com/images/I/61ZRX0IbxFL._AC_SL1000_.jpg)

Download [Mission Planner](https://ardupilot.org/planner/docs/mission-planner-installation.html) to connect to your [Flight Controller](https://a.co/d/29JsbCW) or [Pixhawk PX4 Flight Controller](https://a.co/d/iWNnGU8), then we need to install [MAVProxy](https://pypi.org/project/MAVProxy/) and [DroneKit-Python](https://github.com/dronekit/dronekit-python/) on Jetson Nano, **MAVProxy** is a powerful command-line based “developer” ground station software, and **DroneKit-Python** allows you to control ArduPilot using the Python.

![APM2.8 Flight Controller](https://i.ebayimg.com/images/g/-5EAAOSwpOxhHMsS/s-l500.jpg)

-----------------------------------------------------------------------------------------------
## :sunflower:Deep Learning (YOLOv5)

-----------------------------------------------------------------------------------------------
## :rocket:Software (Drone control)

-----------------------------------------------------------------------------------------------
## :flying_saucer:Software (Simulation in the loop (SIL)

-----------------------------------------------------------------------------------------------
## :iphone:Software (UI)
