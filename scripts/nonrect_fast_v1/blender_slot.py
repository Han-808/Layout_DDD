#!/usr/bin/env python3
"""Take one Blender slot, run Blender, and release the slot when it exits.

This process stays alive for the whole render. Blender closes inherited
file descriptors, so the lock cannot be handed to it with exec.
"""
import fcntl
import os
import signal
import subprocess
import sys
import time

REAL = '/Applications/Blender.app/Contents/MacOS/Blender'
SLOTS = int(os.environ.get('NONRECT_BLENDER_SLOTS', '3'))
SLOT_DIR = os.environ.get(
    'NONRECT_BLENDER_SLOT_DIR',
    '/Users/han_mohan/.codex/worktrees/3b6f/Layout_DDD/Support/nonrect_fast_v1/handoff_remaining12_20260925/blender_slots',
)


def acquire():
    os.makedirs(SLOT_DIR, mode=0o700, exist_ok=True)
    while True:
        for index in range(SLOTS):
            fd = os.open(os.path.join(SLOT_DIR, str(index)), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            return fd
        time.sleep(0.5)


def main():
    fd = acquire()
    proc = subprocess.Popen([REAL, *sys.argv[1:]])

    def forward(signum, _frame):
        proc.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    rc = proc.wait()
    os.close(fd)
    raise SystemExit(rc if rc >= 0 else 128 + abs(rc))


if __name__ == '__main__':
    main()
