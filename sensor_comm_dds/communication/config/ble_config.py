from enum import Enum
from dataclasses import dataclass


class SensorUuid(Enum):
    #  Base 00000000-0000-1000-8000-00805f9b34fb with 16b short UUID as fixed in Arduino firmware
    DATA_CHAR_SMARTTEX = "0000C130-0000-1000-8000-00805F9B34FB"
    DATA_CHAR_ACCELNET = "0000C131-0000-1000-8000-00805F9b34fb"
    DATA_CHAR_CAPSENSE = "0000C132-0000-1000-8000-00805F9b34fb"
    DATA_CHAR_CAPTOUCH = "0000C133-0000-1000-8000-00805F9b34FB"
    DATA_CHAR_IRTOUCH = "0000C134-0000-1000-8000-00805F9b34FB"
    DATA_CHAR_POLYPIEZO = "0000C135-0000-1000-8000-00805F9b34FB"
    DATA_CHAR_MAGTOUCH = "0000C136-0000-1000-8000-00805F9b34FB"
    DATA_CHAR_CLOTHESHANGER = "0000C137-0000-1000-8000-00805F9b34FB"
    DATA_CHAR_PRESSURE = "0000C138-0000-1000-8000-00805F9b34FB"
    DATA_CHAR_SWITCH = "0000C139-0000-1000-8000-00805F9b34FB"


class DeviceMAC(Enum):
    PCB_BLACK = "D3:21:46:1B:B9:A0"
    PCB_RED = "DE:C5:4A:D4:1E:CD"
    PCB_GREEN = "E4:33:DA:53:42:57"
    PCBX = "A0:94:AD:34:E5:AD"
    PCBX2 = "EB:9D:08:1A:E8:FD"
    ARDUINO1 = "C2:55:80:51:13:5C"
    ARDUINO2 = "89:12:69:E2:5C:5B"
    ARDUINO3 = "DA:D5:2B:7D:FC:10"
    ARDUINO4 = "AF:CE:EE:D6:34:58"
    ARDUINO5 = "1E:CD:73:DD:60:66"
    HALBERD1 = "29:BE:81:5B:2F:37"
    HALBERD2 = "25:4D:86:ED:58:93"
    HALBERD3 = "67:62:B4:D1:E0:2A"


@dataclass
class BleConfig:
    """
    uuid: the UUID of the BLE characteristic to which data is written
    device_mac: the hard-coded bluetooth MAC address of the device that you wish to communicate with
    hci: this indicates the bluetooth interface to use. On the workstations, the internal bluetooth often
    has bad connection, so a dongle is needed. Run hciconfig in a terminal to see available bluetooth interfaces.
    """
    uuid: SensorUuid
    device_mac: DeviceMAC
    hci: str = "hci0"
