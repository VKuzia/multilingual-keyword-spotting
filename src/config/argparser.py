import argparse
from typing import Any


class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Run script using given config json.')
        self.parser.add_argument('-c', '--config_path', type=str, nargs='?',
                                 help='relative path to configuration script')

    def parse_args(self) -> Any:
        return self.parser.parse_args()
