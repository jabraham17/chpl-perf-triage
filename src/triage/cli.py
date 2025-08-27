#!/usr/bin/env python3

import argparse
import sys
from typing import List, Optional
import glob
from pathlib import Path
import datetime as dt
from triage.find_anomalies import find_anomalies


def find_files(paths: List[str], recurse: bool = True) -> List[Path]:
    files = []
    for f in paths:
        p = Path(f)
        if p.is_dir():
            if not recurse:
                files.extend(glob.glob(f + "/*.dat", recursive=False))
            else:
                files.extend(glob.glob(f + "/**/*.dat", recursive=True))
        elif p.is_file():
            files.append(f)
        else:
            raise FileNotFoundError(f"File {f} not found")
    return files


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files", nargs="*", help="Directory containing the data files"
    )
    parser.add_argument("--no-recurse", action="store_true", default=False)
    def valid_date(s):
        try:
            return dt.datetime.strptime(s, "%m/%d/%y")
        except ValueError:
            msg = f"Not a valid date: '{s}'."
            raise argparse.ArgumentTypeError(msg)

    parser.add_argument(
        "--start-date",
        help="Start date for analysis - 'MM/DD/YY'",
        type=valid_date,
        default=None,
    )
    args = parser.parse_args(argv)


    files = find_files(args.files, recurse=not args.no_recurse)
    for f in files:
        try:
            find_anomalies(f, start_date=args.start_date)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
