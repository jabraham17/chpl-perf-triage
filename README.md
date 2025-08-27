# chpl-perf-triage

A command line tool for performance triage of Chapel performance data from https://chapel-lang.org/perf

## Installation

You can install this package by cloning the repository and using pip.

```bash
git clone https://github.com/jabraham17/chpl-perf-triage.git
cd chpl-perf-triage
python3 -m pip install .
```

If you are planning to contribute to the development of this package, you can install it with development dependencies:

```bash
pip install ".[dev]"
```

## Usage

After installation, you can use one of the installed commands:

* `triage`: Main command line tool for performance triage.

## Development

### Running tests

```bash
pytest
```

### Code formatting

```bash
black src/ tests/
```
