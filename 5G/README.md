# MECE Spring 2024 Capstone project - Autodrone | Rice University
## 5G

## :clipboard: Overall System Design
### Test Environment
- Linux VM 
  - Ubuntu 20.04.6 LTS  
  - Uses Quectel CM with QMI WWAN Driver
  - Power: USB-C (Modem) to AC Adapter
  - Data: USB-A (Modem) to USB-C (Nano)
### Autodrone
- Jetson Nano
  - Uses Quectel CM with GobiNet Driver
  - Power: USB-C (Modem) to USB-A (Nano)
  - Data: USB-A (Modem) to USB-C (Nano)


-----------------------------------------------------------------------------------------------

## :computer:Hardware (Embedded systems)
### Download Software/Drivers (see folder) 

- Quectel CM
- QMI 
- Gobi
- USB to Serial
- UICC

https://docs.sixfab.com/docs/sixfab-5g-modem-kit-documentation

https://open-cells.com/index.php/uiccsim-programing/

#### Install ATCOM

```
sudo apt install python3-pip
pip3 install atcom
```


#### Install Minicom
Install Minicom 
```
Sudo apt install minicom
```
To launch minicom when UE is connected:
```
Sudo minicom -D /dev/ttyUSB2
```
#### Program SIM
Install UICC
```
make
```
If dependencies fail, use the following:
```
make -B
```
To program a SIM card, use a card reader. Below is an example configuration based on OpenAir (https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/NR_SA_Tutorial_COTS_UE.md#22-sim-card)

```
sudo ./program_uicc --adm 12345678 --imsi 001010000000001 --isdn 00000001 --acc 0001 --key fec86ba6eb707ed08905757b1bb44b8f --opc C42449363BBAD02B66D16BC975D77CC1 -spn "OpenAirInterface" --authenticate![image](https://github.com/Rice-MECE-Capstone-Projects/Autodrone/assets/143918914/3e79efae-3978-4fbd-ae9d-5d88c713858e)

```
-----------------------------------------------------------------------------------------------
## 📱 Initial Operations 
### Basic ATCOM Commands
Please see ATCOM Manual under References folder.
```
AT+CFUN
AT+CGDCONT
AT+CGACT
AT+CPIN?
AT+COPS
AT+5g
```
### Check Host Recognizes Modem
Use either command to look for USB0, USB1, USB2, and USB3 connected
```
ls /dev/tty*
sudo dmesg | grep tty
```
### Start Minicom to send AT Commands
```
Sudo minicom -D /dev/ttyUSB2
```
### Configure UE for OpenAir
```
AT+CGDCONT=1,"IP","oai","0.0.0.0",0,0,0,0
AT+QCFG="usbnet",0

```
### Connect UE and Establish Data Session
- Turn off RF on the UE
```
AT+CFUN=4
```
- Provide PDP Context and set to activate
```
AT+CGDCONT=1,"IP","oai","0.0.0.0",0,0,0,0
AT+CGACT=1,1
```
- Navigate to directory where QCM is compiled and run
```
sudo ./quectel-CM -s oai -4
```
- Turn on 5G network
- Turn on RF on UE
```
AT+CFUN=1
```
- Verify on QCM that the interface received an IP address
- Can also verify through the command line and test with ping
```
ifconfig
ping -I wwan0 8.8.8.8
```
- Verify with AT Commands that network information matches expected values based on 5G network implmenetation
```
AT+QNWINFO
AT+CREG?
AT+C5GREG?
AT+QSPN
```
