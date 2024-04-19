# MECE Spring 2024 Capstone project - Autodrone | Rice University
## 5G

## :clipboard: Overall System Design
### Test Environment

### Autodrone
- Jetson Nano
  - Uses Quectel CM with GobiNet Driver
  - Power: USB-C (Modem) to USB-A (Nano)
  - Data: USB-A (Modem) to USB-C (Nano)
- Linux VM
  - Uses Quectel CM with QMI WWAN Driver
  - Power: USB-C (Modem) to AC Adapter
  - Data: USB-A (Modem) to USB-C (Nano)


-----------------------------------------------------------------------------------------------

## :computer:Hardware (Embedded systems)
### Download Software/Drivers (https://docs.sixfab.com/docs/sixfab-5g-modem-kit-documentation)
- Quectel CM
- QMI 
- Gobi
- USB to Serial

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

Download UICC from https://open-cells.com/index.php/uiccsim-programing/
Install 
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
AT+CGDCONT=1,"IP","oai"
AT+QCFG="usbnet",0

```
