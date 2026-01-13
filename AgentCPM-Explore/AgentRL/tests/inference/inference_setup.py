import subprocess
import os
import random
import sys
# local_visible_devices = ["0","1"]

env = os.environ.copy()
# env["CUDA_VISIBLE_DEVICES"] = ",".join(local_visible_devices)

inference_proc = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path", "/data3/workhome/luyaxi/ARL/output/resume-sft",
        "--port", str(random.randint(10000, 20000)),
        "--tp", str(2),
        "--dp", str(1),
        "--mem-fraction-static", str(0.9),
        "--trust-remote-code",
        ],
    env=env,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    bufsize=1,
    universal_newlines=True,
    user=os.environ.get("USER", "default_user"),
    cwd=os.getcwd()
)
import time
# 监控服务启动输出
uvicorn_started = False
timeout = 600  # 600秒超时
start_time = time.time()
while not uvicorn_started:
    # 检查超时
    if time.time() - start_time > timeout:
        inference_proc.terminate()
        raise TimeoutError("Inference service failed to start within timeout period")
    
    # 检查进程是否意外退出
    if inference_proc.poll() is not None:
        # 获取剩余输出
        output, _ = inference_proc.communicate()
        raise RuntimeError("Inference service process exited unexpectedly")
    
    # 非阻塞读取一行
    line = inference_proc.stderr.readline()
    if not line:
        time.sleep(0.1)  # 如果没有输出，稍等一下再检查
        continue
    print(line[:-1])
    if "Uvicorn running on" in line:
        print("Detected Uvicorn startup message")
        uvicorn_started = True
    
    if "address already in use" in line:
        raise RuntimeError(f"Address already in use.\n{line}")


inference_proc.terminate()
inference_proc.kill()

inference_proc.wait()

print("Inference service killed successfully.")

time.sleep(30)