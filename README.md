# APSDPS (Automatic Pathway Searching based on Discrete Path Sampling)

> [!IMPORTANT]
> This README file is still under development and the latest content will be updated dynamicallyfont.

## Overview

APSDPS is a pathway searching startegy specially developed for complex processes.

## Requirements

### Python libs
* rpyc
* numpy
* scipy
* matplotlib
* rdkit
* joblib
* networkx

### Other
* ORCA v6.1.0
* xtb v6.7.1
* (optimal) Gotify

## Installation

1. copy the files in src to your machine, and check the requirement.
2. setup in `config.py`.
3. copy the `config.py` and `calc_node.py` to the computations nodes. (Using Ramdisk as scratch dir will show better performance)

## Usage

just run the `master_node.py`.

## Update

> [!TIP]
> This project is in refactoring and developing to improve usability and extensibility.

## License

This project is distributed under the MIT License.
