import argparse
import json
from typing import List, Tuple

import queue
import threading

import numpy as np
import torch
from tqdm import tqdm

import sys

sys.path.append("./")

import src.utils as utils
import src.models as models
import src.inference.tools as inference_tools

import matplotlib.pyplot as plt

MAX_TIME_BOUND_SEC = 10000.0
EXIT_SIGNAL = '__@#exit#@__'
VERBOSE = True


def _log(content):
    if VERBOSE:
        print(content)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType('r'))
    parser.add_argument('--saved-models')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--verbose', default=True)
    return parser.parse_args()


def load_heads(heads_configs, model_io, root):
    result = []
    _log('Loading model heads')
    for config in tqdm(heads_configs):
        if config['type'] == 'binary_labeled':
            result.append(inference_tools.BinaryLabeledHead(config, model_io, root))
    return result


def get_models(config, model_io, saved_models_path, device) -> Tuple[
    models.Module, List[models.Module]]:
    _log('Loading model embedding')
    embedding: models.Module = model_io.build_module(config['model']['embedding'],
                                                     saved_models_path).to(device)
    heads = load_heads(config['model']['heads'], model_io, saved_models_path)
    heads = [x.to(device) for x in heads]
    embedding.eval()
    for head in heads:
        head.eval()
    return embedding, heads


def run_actions_thread(trigger_queue, actions):
    while True:
        triggered = trigger_queue.get()
        if isinstance(triggered, str) and triggered == EXIT_SIGNAL:
            break
        # if len(triggered) > 0:
        #     _log(triggered)
        for head_name in triggered:
            if head_name in actions.keys():
                for doable in actions[head_name]:
                    doable.do()

#python3 src/inference/infer_heads.py --config src/inference/configs_23_03/config_infer_01.json --saved-models inference_test_heads --device cuda:0
def run_stream_thread(trigger_queue, streamer, transformer, embedding, heads, device, cooldown):
    current_time = 0.0
    heads_last_trigger_times = {x.name: -MAX_TIME_BOUND_SEC for x in heads}

    for frame in streamer.stream():
        # try:
        current_time += streamer.refresh_rate_sec
        input_features = transformer(frame).to(device)
        embedding_features = embedding(input_features)
        triggered = []
        for head in heads:
            is_triggered = head.infer(embedding_features)
            if is_triggered:
                if current_time - heads_last_trigger_times[head.name] > cooldown:
                    triggered.append(head.name)
                heads_last_trigger_times[head.name] = current_time
        if len(triggered) > 0:
            streamer.on_trigger()
        trigger_queue.put(triggered)
        if current_time + streamer.refresh_rate_sec > MAX_TIME_BOUND_SEC:
            current_time -= MAX_TIME_BOUND_SEC
            for key, value in heads_last_trigger_times.items():
                heads_last_trigger_times[key] = value - MAX_TIME_BOUND_SEC
        # except:
        #     trigger_queue.put(EXIT_SIGNAL)
    trigger_queue.put(EXIT_SIGNAL)


def main():
    global VERBOSE
    args = _parse_args()
    VERBOSE = args.verbose
    config = json.loads(args.config.read())
    model_io: models.ModelIO = models.ModelIO(args.saved_models)
    embedding, heads = get_models(config, model_io, args.saved_models, args.device)
    transformer = inference_tools.get_transformer(config['transformer'])
    streamer = inference_tools.AudioStreamer(config['audio_streamer'])
    actions = inference_tools.get_actions(config['actions'])
    trigger_queue = queue.Queue()
    streaming_thread = threading.Thread(target=run_stream_thread,
                                        args=(trigger_queue,
                                              streamer,
                                              transformer,
                                              embedding,
                                              heads,
                                              args.device,
                                              config['cooldown_in_sec']))
    actions_thread = threading.Thread(target=run_actions_thread, args=(trigger_queue, actions))
    with torch.no_grad():
        _log('STARTING ACTIONS THREAD')
        actions_thread.start()
        _log("STARTING STREAMING THREAD")
        streaming_thread.start()
        _log("RUNNING")
        actions_thread.join()
        streaming_thread.join()
        _log("END")


if __name__ == "__main__":
    main()
