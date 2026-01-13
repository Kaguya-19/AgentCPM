import os
import sys
import json
import uuid
import random
import asyncio
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any
sys.path.append(str(Path(__file__).parent.parent))

import tiktoken
from rich.theme import Theme
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from beanie import init_beanie, PydanticObjectId

from sampler import Record, DispatchedSamplingTask, Task, InferenceService

@dataclass
class UploadConfig(object):

    model_name:str = field(
        metadata={
            "help": "The model name in the whole completion data."
        }
    )

    data_files: List[str] = field(
        metadata={
            "help": "All the data files you want to upload. support `.json` and `.jsonl`"
        }
    )

    chunk_size:int = field(
        default=128,
        metadata={
            "help": "the size of chunk, all the data will be chunked into small sub-data to process."
        }
    )

    task_description: str = field(
        default = "",
        metadata={
            "help":"The description set in the `Task` document."
        }
    )

    mongo_connection_url: str = field(
        default="",
        metadata={
            "help": "the connection url in mongodb, formatting as `mongodb://[username:password@]host1[:port1][,host2[:port2],...]`"
        }
    )

    mongo_db_name: str = field(
        default = "GUI_Agent_Trajectory",
        metadata={
            "help": "the db name of your data."
        }
    )


class Uploader(ABC):

    def __init__(self, config:UploadConfig):
        """
        Read files and process the data. Each trajectory considered as an item.
        """
        self.config = config

        try:
            self.encoding = tiktoken.encoding_for_model(config.model_name)
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.sem_mongo = asyncio.Semaphore(16)

        # rich
        self.console = Console(theme = Theme(
            {
                "info": "cyan",
                "warning": "yellow",
                "error": "bold red"
                }
            ))
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.overall_task = None
        self.sub_task = None

        # prepare statistics.
        self.max_req_tokens = 0
        self.max_resp_tokens = 0
        self.max_all_tokens = 0
        self.overall_task_num = 0

        # prepare data load.
        self.datas = []
        for data_path in self.config.data_files:

            if not os.path.exists(data_path):
                self.console.print(f"File {data_path} does not exist, skip this file.", style = "warning")

            match str(data_path):
                case _jsonline if _jsonline.endswith(".jsonl"):
                    with open(_jsonline, "r", encoding="utf-8") as f:
                        for line in f.readlines():
                            self.datas.append(json.loads(line))

                case _json if _json.endswith(".json"):
                    with open(_json, "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                        self.datas.extend(file_data)

                case __:
                    self.console.print(f"File type {Path(data_path).suffix} is not supported. please use only json format.", style="warning")

        random.shuffle(self.datas)

    async def init_env(self):
        """
        Initialize mongoDB
        """

        self.console.print("Connecting to mongoDB", style = "info")

        connection_string = f"{self.config.mongo_connection_url}/{self.config.mongo_db_name}?authSource=admin"
        await init_beanie(
            connection_string=connection_string,
            document_models=[Record, DispatchedSamplingTask, Task, InferenceService]
        )

        self.task = Task(
            split="train",
            description=self.config.task_description,
            num_samples=0,
            epoch=0,
            added_step=0,
            scores=[random.random()]
        )
        await self.task.insert()
        self.console.print(f"Created task: {self.task.id}", style = "info")

    @abstractmethod
    async def req_and_resp(self, data: Any) -> List[Tuple[Dict, Dict]]:
        """
        Convert one data item into some req and resp that DispatchedTasks requires.
        """
        raise NotImplementedError("Sub-class should implement this function to get request and response items.")

    async def upload_one(self, data: Any, data_idx:int) -> bool:
        """
        Upload one single data.
        """
        origin_dispatched: List[DispatchedSamplingTask] = []
        dispatched_tasks = []

        reqs_and_resps = await self.req_and_resp(data)

        sub_req_max = 0
        sub_resp_max = 0
        sub_total_max = 0

        for req, resp in reqs_and_resps:

            prompt_tokens = self.approx_token(req)
            completion_tokens = self.approx_token(resp)

            if sub_req_max < prompt_tokens:
                sub_req_max = prompt_tokens
            if sub_resp_max < completion_tokens:
                sub_resp_max = completion_tokens
            if sub_total_max < prompt_tokens + completion_tokens:
                sub_total_max = prompt_tokens + completion_tokens

            dispatched_task = DispatchedSamplingTask(
                task=self.task,
                traj_id=data_idx,
                req_type="chatcompletions",
                request=req,
                response={
                    "id": uuid.uuid4().hex,
                    "choices": [{
                        "finish_reason": "stop",
                        "index": 0,
                        "logprobs": None,
                        "message": resp
                    }],
                    "created": int(datetime.now().timestamp()),
                    "model": self.config.model_name,
                    "object": "chat.completion",
                    "usage": {
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                },
                score=random.random(),
                status=DispatchedSamplingTask.Status.COMPLETED,
                finish_time=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                creat_time=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                priority=0,
            )

            if self.validate_task(dispatched_task):
                origin_dispatched.append(dispatched_task)

            else:
                return False

        async with self.sem_mongo:
            for dispatched_task in origin_dispatched:
                await dispatched_task.insert()
                doc_id = dispatched_task.id
                task_doc = await DispatchedSamplingTask.get(PydanticObjectId(doc_id))
                dispatched_tasks.append(task_doc)

        record = Record(
            task=self.task,
            split="train",
            traj_id=data_idx,
            score=random.random(),
            status=Record.Status.READY,
            trained_count=0,
            last_trained_step=-1,
            traj=dispatched_tasks
        )
        await record.insert()

        # update data stats.
        if sub_req_max > self.max_req_tokens:
            self.max_req_tokens = sub_req_max

        if sub_resp_max > self.max_resp_tokens:
            self.max_resp_tokens = sub_resp_max

        if sub_total_max > self.max_all_tokens:
            self.max_all_tokens = sub_total_max

        self.overall_task_num += len(dispatched_tasks)
        self.progress.update(self.sub_task, advance = 1)

        return True


    def validate_task(self, dispatched_task:DispatchedSamplingTask) -> bool:
        return True

    async def upload(self):
        """
        Upload all data.
        """
        await self.init_env()

        length = len(self.datas)
        results = []

        with self.progress:

            self.overall_task = self.progress.add_task("[green] processing chunks", total = length // self.config.chunk_size + 1)
            self.sub_task = self.progress.add_task("[blue] Processing chunk datas:", total = self.config.chunk_size)

            for idx in range(0, length, self.config.chunk_size):

                sub_datas = self.datas[idx:idx + self.config.chunk_size]
                coros = [self.upload_one(data, data_idx= idx + jdx) for jdx, data in enumerate(sub_datas)]
                res = await asyncio.gather(*coros, return_exceptions=True)

                for r in res:
                    if not isinstance(res, bool):
                        self.console.print(r, style="error")
                        break

                results += res
                self.progress.advance(self.overall_task, advance=1)
                self.progress.reset(self.sub_task)

        self.export_stats(results)

    def export_stats(self, results:List[Union[bool, BaseException]]):

        table = Table(title = "Statistics")

        table.add_column("", justify="center", no_wrap=True)
        table.add_column("", style = "green")

        table.add_row("all data size", str(len(results)))
        table.add_row("uploaded record number", str(sum([i for i in results if isinstance(i, bool) and i])))
        table.add_row("uploaded dispatched task number", str(self.overall_task_num))
        table.add_row("max prompt length(approx)", str(self.max_req_tokens))
        table.add_row("max completions length", str(self.max_resp_tokens))
        table.add_row("max complete tokens", str(self.max_all_tokens))

        self.console.print(table)

    def approx_token(self, messages:List[Dict]):
        messages_str = json.dumps(messages)
        return len(self.encoding.encode(messages_str))


class MathUploader(Uploader):

    def __init__(self,config:UploadConfig):
        super().__init__(config)

    def validate_task(self, dispatched_task):
        return dispatched_task.response["usage"]["completion_tokens"] < 512

    async def req_and_resp(self, data):

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant"
            },
            {
                "role": "user",
                "content": data["problem"]
            }
        ]

        req = {
            "messages": messages,
            "model": self.config.model_name
        },

        resp = {
            "role": "assistant",
            "content": data["solution"]
        }

        return [(req, resp)]


if __name__ == '__main__':

    math_config = {
        "mongo_connection_url": "",
        "task_description": "Test math",
        "model_name": "test_model",
        "chunk_size": 128,
        "data_files": ["assets/math500.jsonl"],
        "mongo_db_name": "test_upload"
    }

    config = UploadConfig(**math_config)

    uploader = MathUploader(config)

    asyncio.run(uploader.upload())