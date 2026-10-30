"""Networking fix for this host: it resolves IPv6 (AAAA) records but has no
working IPv6 route, so httpx/huggingface_hub intermittently stalls or fails
with SSL EOF / name-resolution errors when a client picks the IPv6 address
first. Forcing IPv4 at the socket layer makes Hugging Face downloads reliable
without touching /etc/hosts or requiring root.

Import and call ``force_ipv4()`` before any Hub download in a script.
"""

from __future__ import annotations

import socket

_original_getaddrinfo = socket.getaddrinfo


def force_ipv4() -> None:
    """Make socket.getaddrinfo return IPv4 addresses only (idempotent)."""

    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002
        return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = getaddrinfo
