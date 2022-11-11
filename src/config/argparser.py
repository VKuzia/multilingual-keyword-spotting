import argparse
from typing import Any

from src.paths import PATH_TO_SAVED_MODELS


class ArgParser:
    """Parses command line arguments passed to the script.
    Has only one use for now: launching specified script"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Run script using given config json.')
        self.parser.add_argument('-c', '--config_path', type=str, nargs='?',
                                 help='relative path to configuration script')

    def parse_args(self) -> Any:
        return self.parser.parse_args()


class TrainArgParser(ArgParser):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('-o', '--checkpoint_output', type=str, nargs='?',
                                 help='relative path to directory to save models to',
                                 default=PATH_TO_SAVED_MODELS)
