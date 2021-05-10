import sys
import os


def get_mac_address():
    if sys.platform == 'win32':
        for line in os.popen("ipconfig /all"):
            if line.lstrip().startswith('Physical Address'):
                mac_address = line.split(':')[1].strip().replace('-', ':')
                break
    else:
        ifconfig_lines = ""
        for line in os.popen("/sbin/ifconfig"):
            ifconfig_lines += line
        interface_infos = [info.strip() for info in ifconfig_lines.split("collisions 0")]
        for info in interface_infos:
            # don't use docker or virtual ethernet Mac addresses -- they tend to change
            if not (info.startswith("docker") or info.startswith("veth") or len(info) == 0):
                for line in info.split("\n"):
                    if line.find('ether') > -1:
                        mac_address = line.split()[1]
                        break
    return mac_address
