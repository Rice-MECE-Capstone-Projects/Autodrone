# Simulation:

This folder includes codes that were used to help with the simulation of the drone operations like hovering and fly to certain locations. Read the comments in the codes for more details about each of them and how to use them.

## Prerequisites

All of the codes are conducted under:

Dronekit 2.9.2


ArduCopter 4.0.4


MAVProxy 1.8.17


Pymavlink 2.4.10


Before using the code in this repository, you are suggested to install Python and Ardupilot on your system.

## Installing Python

To install Python on your system, use the following command:

(`$ sudo apt-get install python3`)


## Installing Ardupilot

To install Ardupilot on your system, follow these steps:

1. First, make sure you have the necessary dependencies installed. You can use the following command to install them:


(`$ sudo apt-get install python-pip python-dev libxml2-dev libxslt1-dev zlib1g-dev`)


2. Next, clone the Ardupilot repository from GitHub using the following command:


(`$ git clone https://github.com/ArduPilot/ardupilot.git`)



3. Change into the directory where Ardupilot was cloned, and then use the `pip` command to install the required Python packages:


(`$ cd ardupilot`)


(`$ pip install -r requirements.txt`)




4. Finally, you can use the `make` command to build and install Ardupilot on your system:


(`$ make configure`)


(`$ make px4-clean`)


(`$ make px4`)



After following these steps, Ardupilot should be installed and ready to use on your system.
