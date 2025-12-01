import sys
import os
import subprocess
import socket
import json
import threading
import time
import random
import traceback
import yaml
from datetime import datetime
import importlib
import argparse
from pathlib import Path
from collections import deque

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs._GLOBAL_CONFIGS import CONFIGS_PATH

import numpy as np
from typing import Any
import base64


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder extension for numpy types, includes reconstruction metadata"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Determine dtype for reconstruction
            if obj.dtype == np.float32:
                dtype = 'float32'
            elif obj.dtype == np.float64:
                dtype = 'float64'
            elif obj.dtype == np.int32:
                dtype = 'int32'
            elif obj.dtype == np.int64:
                dtype = 'int64'
            else:
                dtype = str(obj.dtype)
            # Encode array bytes as base64
            return {
                '__numpy_array__': True,
                'data': base64.b64encode(obj.tobytes()).decode('ascii'),
                'dtype': dtype,
                'shape': obj.shape
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_to_json(data: Any) -> str:
    """Serialize Python data (including numpy arrays) to JSON string"""
    return json.dumps(data, cls=NumpyEncoder)


def json_to_numpy(json_str: str) -> Any:
    """Deserialize JSON string back to Python objects, reconstructing numpy arrays"""
    def object_hook(dct):
        if '__numpy_array__' in dct:
            raw = base64.b64decode(dct['data'])
            return np.frombuffer(raw, dtype=dct['dtype']).reshape(dct['shape'])
        return dct
    return json.loads(json_str, object_hook=object_hook)


# --------------------- Model Server Implementation ---------------------
class ModelServer:
    def __init__(self, model, host='localhost', port=None,
                 enable_batching=False, batch_window=0.5, max_batch=64):
        self.model = model
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.wait_interval = 10
        self.client_threads = []

        # Batching configuration
        self.enable_batching = bool(enable_batching)
        self.batch_window = float(batch_window)
        self.max_batch = int(max_batch)

        # Batching data structures
        self.queue = deque()
        self.queue_lock = threading.Lock()
        self.queue_cv = threading.Condition(self.queue_lock)
        self.batch_thread = None
        self.stop_event = threading.Event()

        # Per-socket send locks to avoid interleaved writes
        self.socket_locks = {}
        self.socket_locks_lock = threading.Lock()

    def start(self):
        """Start the model server and listen for incoming client connections"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.settimeout(self.wait_interval)
        self.server_socket.listen(5)
        self.running = True

        print(f"🚀 Model server started on {self.host}:{self.port}")
        print("🔄 Server is waiting for client connections...")

        # Start optional batching worker
        if self.enable_batching:
            self.batch_thread = threading.Thread(target=self._batch_worker, daemon=True)
            self.batch_thread.start()

        self._accept_connections()

    def stop(self):
        """Stop the server and clean up resources gracefully"""
        self.running = False
        self.stop_event.set()
        # wake batch thread if waiting
        with self.queue_cv:
            self.queue_cv.notify_all()
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        for t in self.client_threads:
            t.join(timeout=1)
        if self.batch_thread:
            self.batch_thread.join(timeout=1)
        print("🛑 Server has been stopped")

    def _accept_connections(self):
        """Accept and handle new client connections"""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                print(f"✅ Client connected from {addr}")
                # Handle each client in a separate thread
                t = threading.Thread(target=self._handle_client,
                                     args=(client_socket,), daemon=True)
                t.start()
                self.client_threads.append(t)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"⚠️ Error accepting connection: {e}")
                break

    def _handle_client(self, client_socket):
        """Process requests from a single client"""
        # Register socket lock
        self._register_socket(client_socket)
        with client_socket:
            while self.running:
                try:
                    # Read message length header (4 bytes, big-endian)
                    len_bytes = client_socket.recv(4)
                    if not len_bytes:
                        print("🔌 Client disconnected")
                        break
                    msg_length = int.from_bytes(len_bytes, 'big')

                    # Read the full message based on length
                    chunks = []
                    remaining = msg_length
                    while remaining > 0:
                        chunk = client_socket.recv(min(remaining, 4096))
                        if not chunk:
                            raise ConnectionError("Incomplete data received")
                        chunks.append(chunk)
                        remaining -= len(chunk)
                    raw_msg = b''.join(chunks).decode('utf-8')

                    # Deserialize JSON to Python, reconstruct any numpy arrays
                    data = json_to_numpy(raw_msg)

                    # Extract command and observation
                    cmd = data.get("cmd")
                    obs = data.get("obs")  # None if not provided

                    # Find corresponding model method
                    method = getattr(self.model, cmd, None)
                    if not callable(method):
                        raise AttributeError(f"No model method named '{cmd}'")

                    # If batching enabled and we have an observation, enqueue
                    if self.enable_batching and obs is not None:
                        req = {
                            'cmd': cmd,
                            'obs': obs,
                            'socket': client_socket
                        }
                        with self.queue_cv:
                            self.queue.append(req)
                            # notify batch worker
                            self.queue_cv.notify()
                        # Do not send response here; batch worker will send it
                        continue

                    # Otherwise call method immediately
                    result = method(obs) if obs is not None else method()
                    response = {"res": result}

                    # Serialize response and send back with length header (protected)
                    self._send_response(client_socket, response)

                except (ConnectionResetError, BrokenPipeError):
                    print("🔌 Client connection lost")
                    break
                except Exception as e:
                    err = f"Error handling request: {e}"
                    print(f"⚠️ {err}")
                    tb = traceback.format_exc()
                    error_resp = numpy_to_json({"error": err, "traceback": tb}).encode('utf-8')
                    try:
                        client_socket.sendall(len(error_resp).to_bytes(4, 'big'))
                        client_socket.sendall(error_resp)
                    except Exception:
                        pass
                    break

        # Unregister socket lock when client handler exits
        self._unregister_socket(client_socket)

    def _register_socket(self, sock):
        try:
            key = sock.fileno()
        except Exception:
            return
        with self.socket_locks_lock:
            if key not in self.socket_locks:
                self.socket_locks[key] = threading.Lock()

    def _unregister_socket(self, sock):
        try:
            key = sock.fileno()
        except Exception:
            return
        with self.socket_locks_lock:
            if key in self.socket_locks:
                # do not remove immediately; leave to GC; but remove to avoid leak
                del self.socket_locks[key]

    def _get_socket_lock(self, sock):
        try:
            key = sock.fileno()
        except Exception:
            return threading.Lock()
        with self.socket_locks_lock:
            return self.socket_locks.setdefault(key, threading.Lock())

    def _send_response(self, sock, response):
        try:
            resp_bytes = numpy_to_json(response).encode('utf-8')
            lock = self._get_socket_lock(sock)
            with lock:
                sock.sendall(len(resp_bytes).to_bytes(4, 'big'))
                sock.sendall(resp_bytes)
        except Exception:
            # best-effort: ignore send errors
            pass

    def _batch_worker(self):
        """Worker that periodically flushes queue and runs batched inference."""
        while not self.stop_event.is_set():
            with self.queue_cv:
                # wait until there is something or timeout
                if not self.queue:
                    self.queue_cv.wait(timeout=self.batch_window)
                # after wake, build a batch up to max_batch
                batch = []
                while self.queue and len(batch) < self.max_batch:
                    batch.append(self.queue.popleft())

            if not batch:
                continue

            # Group by command name
            groups = {}
            for req in batch:
                groups.setdefault(req['cmd'], []).append(req)

            # Process each command group
            for cmd, reqs in groups.items():
                obs_list = [r['obs'] for r in reqs]
                batch_method_name = f"batch_{cmd}"
                try:
                    if hasattr(self.model, batch_method_name):
                        batch_method = getattr(self.model, batch_method_name)
                        results = batch_method(obs_list)
                    else:
                        # fallback to individual calls
                        results = []
                        method = getattr(self.model, cmd, None)
                        if not callable(method):
                            raise AttributeError(f"No model method named '{cmd}'")
                        for obs in obs_list:
                            results.append(method(obs))

                    # Send back results in order
                    for r, res in zip(reqs, results):
                        response = {"res": res}
                        try:
                            self._send_response(r['socket'], response)
                        except Exception:
                            pass
                except Exception as e:
                    tb = traceback.format_exc()
                    err_msg = {"error": str(e), "traceback": tb}
                    for r in reqs:
                        try:
                            self._send_response(r['socket'], err_msg)
                        except Exception:
                            pass

# --------------------- Utility Decorators ---------------------
def class_decorator(task_name):
    """Instantiate environment class for given task"""
    envs_module = importlib.import_module(f"envs.{task_name}")
    if not hasattr(envs_module, task_name):
        raise SystemExit("Task not found")
    return getattr(envs_module, task_name)()


def eval_function_decorator(policy_name, model_name, conda_env=None):
    """Load a specified function (e.g., get_model) from a policy module"""
    module = importlib.import_module(policy_name)
    return getattr(module, model_name)


def get_camera_config(camera_type):
    """Load camera configuration from YAML file"""
    cfg_path = os.path.join(os.path.dirname(__file__), "../task_config/_camera_config.yml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError("Camera config file not found")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if camera_type not in cfg:
        raise KeyError(f"Camera type '{camera_type}' is not defined")
    return cfg[camera_type]


def get_embodiment_config(robot_file):
    """Load robot embodiment configuration from YAML"""
    path = os.path.join(robot_file, "config.yml")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main(usr_args):
    """Main entry: load model, start server, run indefinitely"""
    # Extract basic arguments
    policy_name = usr_args['policy_name']
    port = usr_args.get('port')

    # Instantiate model
    get_model = eval_function_decorator(policy_name, 'get_model')
    model = get_model(usr_args)

    # Start server in background thread
    enable_batching = usr_args.get('enable_batching')
    batch_window = usr_args.get('batch_window')
    max_batch = usr_args.get('max_batch')
    server = ModelServer(model, port=port,
                         enable_batching=enable_batching,
                         batch_window=batch_window,
                         max_batch=max_batch)
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()

    # Keep main thread alive until KeyboardInterrupt
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        server.stop()
        thread.join()


def parse_args_and_config():
    """Parse CLI args and YAML config, merge overrides"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, help='Port for ModelServer (optional)')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--enable_batching', type=bool, action='store_true', help='Enable batching for ModelServer')
    parser.add_argument('--batch_window', type=float, default=0.5, help='Batch window duration for ModelServer')
    parser.add_argument('--max_batch', type=int, default=64, help='Maximum batch size for ModelServer')
    parser.add_argument('--overrides', nargs=argparse.REMAINDER,
                        help='Override config values')
    args = parser.parse_args()

    # Load base config
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['port'] = args.port

    # Parse overrides: --key value pairs
    if args.overrides:
        it = iter(args.overrides)
        for key in it:
            val = next(it)
            cfg[key.lstrip('--')] = eval(val) if val.isnumeric() else val
    return cfg


if __name__ == '__main__':
    usr_args = parse_args_and_config()
    main(usr_args)
