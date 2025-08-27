#!/usr/bin/env python3

import argparse
import sys
from typing import List, Optional

import pandas as pd


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args(argv)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
