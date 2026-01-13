from typing import Optional, List, Any, Set, Literal
import time
import asyncio  
from asyncio import AbstractEventLoop
import datetime
from enum import Enum
from pydantic import BaseModel, Field
from beanie import Document, Indexed, Link, init_beanie, PydanticObjectId, UpdateResponse
import threading
from queue import Queue as ThreadQueue
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import InferenceService, DispatchedSamplingTask, Task, Record, models_to_initialize
from torch.utils.data import IterableDataset, DataLoader
 
import asyncio
from beanie import UpdateResponse, init_beanie
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from queue import Queue, Empty
import threading
from src.log import logger

from dataclasses import dataclass, field
@dataclass
class GRPOTrainingConfig:
    db_connection_string: str = field(
        default="mongodb://10.0.1.8:27018,10.0.1.9:27018,10.0.1.11:27018/TestRun",
        metadata={"help": "Database connection string for storing trajectories."}
    )
    
    db_cache_size: int = field(
        default=10,
        metadata={"help": "cache 中要获取的 record的个数 （cache 存的是 每个record 下采样的 samples 列表，包含其所有ready trajs）"}
    )

class DBIterableDataset(IterableDataset):
    def __init__(
        self,
        tp_group: dist.ProcessGroup = None,
        config: GRPOTrainingConfig = None
    ):
        super().__init__()
        self.tp_group = tp_group
        self.tp_rank = dist.get_rank(self.tp_group) if self.tp_group else 0
        
        self.args = config
        self.cache_size = self.args.db_cache_size if self.args.db_cache_size is not None else 20
        self.cache = Queue(self.cache_size)
        self.stop_event = threading.Event()
        self.db_initialized = threading.Event()
        self._thread = None
        
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._start_cache_loop,
                name="DBIterableDatasetCacheThread",
                daemon=True
            )
            self._thread.start()
            logger.critical(f"DBIterableDatasetCacheThread started.")  
            
    def _start_cache_loop(self):
        logger.critical(f"_start_cache_loop 执行")  
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.loop = asyncio.get_event_loop() 
        self.loop.run_until_complete(self._cache_producer())
        # self.loop.run_until_complete(self._run_cache_producer())
    
    # async def _run_cache_producer(self):
        # Mongo CB Client Initialize
        # if not self.db_initialized.is_set():
        #     await init_beanie(
        #         connection_string=self.args.db_connection_string if self.args.db_connection_string is not None else  "mongodb://10.0.1.9:27018/myapp",
        #         document_models=models_to_initialize
        #     )
        #     self.db_initialized.set()
        # logger.critical(f"子线程 db 初始化完毕")  
        # Cache Monitor
        # await self._cache_producer()
    
    async def _cache_producer(self):
        logger.critical(f"进入 _cache_producer")  
        while not self.stop_event.is_set():
            try:
                if self.cache.qsize() >= self.cache_size:
                    await asyncio.sleep(0.1)
                    continue
                
                if self.tp_rank == 0:
                    for _ in range(self.cache_size):
                        # logger.critical(f"get a record")  
                        record = await Record.find_one(Record.trained_count < 1, Record.status == Record.Status.READY).inc({"trained_count": -1}, response_type=UpdateResponse.NEW_DOCUMENT)
                        
                        if not record: 
                            await asyncio.sleep(0.2)
                            continue
                        
                        task = await record.task.fetch(fetch_links=True)
                        # scores = task.scores
                        scores = [1]

                        samples = []
                        for item in record.traj:
                            sample = await item.fetch(fetch_links=True)
                            data = {
                                "prompt": sample.request["messages"],
                                "completion": sample.response["choices"][0]["message"]["content"],
                                "reward": record.score,
                                "advantage": record.score - sum(scores) / len(scores) if scores else 0,
                            }

                            if sample.response["choices"][0]["logprobs"]:
                                old_per_token_logps = torch.tensor(
                                    [item["logprob"] for item in sample.response["choices"][0]["logprobs"]["content"]]
                                )
                                data["old_per_token_logps"] = old_per_token_logps
                                completion = [item["token"] for item in sample.response["choices"][0]["logprobs"]["content"]]
                                data["completion"] = completion
                                
                            samples.append(data)
                        self.cache.put(samples)
            except Exception as e:
                import traceback
                logger.error(f"[Producer Error] {e}\n{traceback.format_exc()}")
                await asyncio.sleep(1)
                        
        
    def __iter__(self):
        # obtain records from the database
        try:
            while True:
                try:
                    if self.tp_rank == 0:
                        samples = self.cache.get()
                    else:
                        samples = None
                        
                    if self.tp_group:
                        gathered_samples = [None for _ in range(dist.get_world_size(self.tp_group))]
                        dist.all_gather_object(gathered_samples, samples, group=self.tp_group)
                        samples = gathered_samples[0]
                        logger.critical(f"[Rank {self.tp_rank}] got samples of length {len(samples)}, len(gathered_samples): {len(gathered_samples)}")

                    if samples is not None:
                        for data in samples:
                            yield data
                except Empty:
                    logger.warning(f"Cache is empty, yield None")
                    yield None
        except Exception as e:
            print(f"Error in DBIterableDataset: {e}")
            raise e
    
    def __del__(self):
        self.stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)
    
    async def estimate_length(self):
        pipeline = [
            {
                "$match": {
                    "trained_count": 0,
                    "status": Record.Status.READY
                }
            },
            {
                "$group": {
                    "_id": 0,
                    "total_traj_length": { 
                        "$sum": { "$size": "$traj" }
                    }
                }
            }
        ]
        result = await Record.aggregate(pipeline).to_list()
        if result:
            return result[0]["total_traj_length"]
        return 0
    
    
    

def single_device_test():
    
    model_path = "/share_data/data1/models/Qwen/Qwen3-8B"
    connection_string = "mongodb://10.0.1.8:27018,10.0.1.9:27018,10.0.1.11:27018/myapp"
    document_models = models_to_initialize
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(init_beanie(
            connection_string=connection_string,
            document_models=document_models,
            multiprocessing_mode=True
        ))
        
        dataset = DBIterableDataset(
            tp_group=None,
            config=GRPOTrainingConfig()
        )
        
        dataloader = DataLoader(dataset, batch_size=None)  # IterableDataset 不能设置 batch_size > 1

        sample_cnt = 0
        for sample in dataloader:
            if sample is None:
                logger.warning("Got empty sample.")
                continue

            # logger.info(f"Sample: {sample}")
            sample_cnt += 1
            # if sample_cnt >= 3:  # 控制测试样本数量
            #     break
            logger.info(f"Sample_cnt: {sample_cnt}")
        
        
    finally:
        if loop.is_running():
            loop.stop()
        loop.close()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12359"

def set_up(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def distributed_device_test(rank, world_size):
    torch.cuda.set_device(rank)

    try:
        set_up(rank, world_size)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(init_beanie(
            connection_string="mongodb://10.0.1.8:27018,10.0.1.9:27018,10.0.1.11:27018/myapp",
            document_models=models_to_initialize,
            multiprocessing_mode=True
        ))

        tp_group = dist.new_group(ranks=list(range(world_size)))
        dataset = DBIterableDataset(tp_group=tp_group, config=GRPOTrainingConfig())
        dataloader = DataLoader(dataset, batch_size=None)

        sample_cnt = 0
        for sample in dataloader:
            if sample is None:
                logger.warning(f"[Rank {rank}] Got empty sample.")
                continue
            sample_cnt += 1
            logger.info(f"[Rank {rank}] Sample_cnt: {sample_cnt}")
    except Exception as e:
        logger.error(f"[Rank {rank}] Exception: {e}")
    finally:
        cleanup()
        loop.close()

def main():
    world_size = 2
    mp.spawn(distributed_device_test, args=(world_size,), nprocs=world_size, join=True, start_method="spawn")

if __name__ == "__main__":
    # single_device_test()
     main()