import asyncio
from beanie import init_beanie,UpdateResponse
import sys
sys.path.append("./")  
sys.path.append("./src")
from src.models import models_to_initialize, DistributedLock, DistributedCounter
import threading
from pymongo import ReturnDocument
from beanie.operators import In

loop = asyncio.get_event_loop()

def run():
    loop = asyncio.new_event_loop()
    loop.run_until_complete(DistributedLock.create("Test"))
    print("Semaphore created in thread:", threading.current_thread().name)
    

loop.run_until_complete(init_beanie(
    connection_string="mongodb://10.0.1.8:27021,10.0.1.9:27021,10.0.1.10:27021,10.0.1.11:27021/math-state",
    document_models=models_to_initialize,
    multiprocessing_mode=True
))

min_sem = loop.run_until_complete(DistributedCounter.find(
                    In(DistributedCounter.name, [f"train-fetch-{i}" for i in range(8)])
                ).min(DistributedCounter.n))
import pdb
pdb.set_trace()

test_thd = threading.Thread(target=run, daemon=True)
test_thd.start()

test_thd.join()