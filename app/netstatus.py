"""Live network status measured from the OS (psutil), shown on the dashboard.

What is measured (and labelled honestly in the UI):
  * online: whether any non-loopback interface is up with an IP address;
  * bytes_sent: bytes sent on all non-loopback interfaces since the app started
    (machine-wide: includes other programs, so 0 is a strong, conservative result);
  * external_sockets: open sockets of this app process to non-loopback addresses.
"""
from __future__ import annotations

import ipaddress
import os
import socket

import psutil

_LOOP = ("lo", "lo0")


def _nonloop_counters() -> tuple[int, int]:
    s = r = 0
    for nic, c in psutil.net_io_counters(pernic=True).items():
        if nic in _LOOP or nic.startswith("lo"):
            continue
        s += c.bytes_sent; r += c.bytes_recv
    return s, r


def baseline() -> tuple[int, int]:
    return _nonloop_counters()


def online() -> bool:
    stats = psutil.net_if_stats()
    for nic, addrs in psutil.net_if_addrs().items():
        if nic.startswith("lo") or not stats.get(nic) or not stats[nic].isup:
            continue
        for a in addrs:
            if a.family in (socket.AF_INET, socket.AF_INET6):
                try:
                    ip = ipaddress.ip_address(a.address.split("%")[0])
                except ValueError:
                    continue
                if not (ip.is_loopback or ip.is_link_local):
                    return True
    return False


def external_sockets() -> int:
    """Outbound connections from this process to non-loopback hosts.

    Inbound connections to a port this process listens on (the browser talking to the app) are not
    counted: behind a container port-forward (podman/pasta) they arrive from a non-loopback address.
    """
    n = 0
    try:
        conns = psutil.Process(os.getpid()).net_connections(kind="inet")
    except (psutil.AccessDenied, AttributeError):
        return -1
    listening = {c.laddr.port for c in conns if c.status == psutil.CONN_LISTEN}
    for c in conns:
        if not c.raddr or c.laddr.port in listening:
            continue
        try:
            if not ipaddress.ip_address(c.raddr.ip.split("%")[0]).is_loopback:
                n += 1
        except ValueError:
            pass
    return n


def status(base: tuple[int, int]) -> dict:
    s, r = _nonloop_counters()
    return {"online": online(), "bytes_sent": max(0, s - base[0]), "bytes_recv": max(0, r - base[1]),
            "external_sockets": external_sockets()}
