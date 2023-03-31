import os
import subprocess
from abc import abstractmethod
from collections import defaultdict

import keyboard
import pyautogui

class Action:
    def __init__(self, config):
        pass

    @abstractmethod
    def do(self):
        pass


class KeyboardAction(Action):
    def __init__(self, config):
        super().__init__(config)
        self.content = config['content']
        self.method = config.get('method', 'press')

    def do(self):
        if self.method == 'write':
            pyautogui.write(self.content)
        elif self.method == 'press':
            pyautogui.press(self.content)
        else:
            raise ValueError(f'Unknown method "{self.method}"')


class CommandAction(Action):

    def __init__(self, config):
        super().__init__(config)
        self.command = config['command']

    def do(self):
        subprocess.Popen(self.command, shell=True)


def get_actions(actions_config):
    result = defaultdict(list)
    for name, config in actions_config.items():
        if config['type'] == 'keyboard':
            result[name].append(KeyboardAction(config))
        elif config['type'] == 'command':
            result[name].append(CommandAction(config))
        else:
            raise ValueError(f'Unknown type "{config["type"]}"')
    return result
