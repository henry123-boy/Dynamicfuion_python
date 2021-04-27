import sys
import os


def get_mac_address():
    if sys.platform == 'win32':
        for line in os.popen("ipconfig /all"):
            if line.lstrip().startswith('Physical Address'):
                mac_address = line.split(':')[1].strip().replace('-', ':')
                break
    else:
        for line in os.popen("/sbin/ifconfig"):
            if line.find('Ether') > -1:
                mac_address = line.split()[1]
                break
    return mac_address
