
from __future__ import annotations
import sys, time, contextlib

class SimpleLogger:
    def __init__(self, name:str="log"):
        self.name = name
    def info(self, msg:str):
        print(f"[{self.name}][{time.strftime('%H:%M:%S')}] {msg}")
    def warn(self, msg:str):
        print(f"[{self.name}][{time.strftime('%H:%M:%S')}] WARNING: {msg}", file=sys.stderr)

log = SimpleLogger("topocn")
