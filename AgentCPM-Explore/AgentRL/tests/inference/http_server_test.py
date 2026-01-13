import socket
import random

import os
import time
import subprocess
import logging
import sys
from contextlib import contextmanager
import torch


if __name__ == "__main__":
    model_name_or_path = "/data3/workhome/luyaxi/ARL/output/resume-sft"

    logger = logging.getLogger(__name__)

    BASE_PORT = 21000
    port = random.randint(BASE_PORT,65535)
    
    env = os.environ.copy()
    env.pop("TORCHELASTIC_USE_AGENT_STORE",None)
    env.pop("RANK", None)
    env.pop("LOCAL_RANK", None)
    env.pop("WORLD_SIZE", None)
    env.pop("MASTER_ADDR", None)
    env.pop("MASTER_PORT", None)

    import multiprocessing
    from sglang.srt.entrypoints.http_server import launch_server, ServerArgs
    
    def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
        p = multiprocessing.Process(target=launch_server, args=(server_args,))
        p.start()

        base_url = server_args.url()
        timeout = 300.0  # Increased timeout to 5 minutes for downloading large models
        start_time = time.perf_counter()

        with requests.Session() as session:
            while time.perf_counter() - start_time < timeout:
                try:
                    headers = {
                        "Content-Type": "application/json; charset=utf-8",
                        "Authorization": f"Bearer {server_args.api_key}",
                    }
                    response = session.get(f"{base_url}/health_generate", headers=headers)
                    if response.status_code == 200:
                        return p
                except requests.RequestException:
                    pass

                if not p.is_alive():
                    raise Exception("Server process terminated unexpectedly.")

                time.sleep(2)

        p.terminate()
        raise TimeoutError("Server failed to start within the timeout period.")
    