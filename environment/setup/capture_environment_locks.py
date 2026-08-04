#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    parser=argparse.ArgumentParser(description='Capture environment lock files from local virtual environments.'); parser.add_argument('--write', action='store_true'); args=parser.parse_args(); print('lock capture is local-environment dependent'); return 0
if __name__=='__main__': raise SystemExit(main())
