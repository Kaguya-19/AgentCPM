import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional
import openai.types.chat
import openai.types.completion
import datetime
from beanie import UpdateResponse
import sys
sys.path.append("./")
from src.models import InferenceService, DispatchedSamplingTask, Task, Record, NoTaskAvailableError
from src.log import logger


class AsyncSampler(ABC):
    """
    An abstract base class for running a asynchronous sampler.
    """

    def __init__(
        self,
        *args,**kwargs
    ):
        super().__init__()
        self.record = None
    
    
    async def __aenter__(self):
        """Enter the asynchronous context manager."""
        # find a task to sample from
        self.task = await Task.find_one(Task.num_samples < 1,with_children=True).inc({Task.num_samples: 1},response_type=UpdateResponse.NEW_DOCUMENT)
        if self.task is None:
            raise NoTaskAvailableError("No task available for sampling. Please create a task first.")
        self.record = Record(
            task=self.task,
            traj_id=self.task.num_samples - 1,
        )
        # save the record
        await self.record.save()
        
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit the asynchronous context manager."""
        # Clean up resources if necessary
        if self.record is not None:
            if exc_type is None:
                self.record.status = Record.Status.COMPLETED
            else:
                self.record.status = Record.Status.FAILED
                logger.warning(f"Record {self.record.traj_id} for task {self.record.task.id} failed with error: {exc_value}")
                import traceback
                logger.error(traceback.format_exc())
            await self.record.save()
            self.record = None
        if self.task is not None:
            self.task = None
        
        
        if exc_type is not None:
            if isinstance(exc_value, NoTaskAvailableError):
                logger.warning(f"No task available for sampling: {exc_value}")
                raise exc_value
            else:
                logger.error(f"An error occurred during sampling: {exc_value}")
        
    
    async def create_chat_completions(
        self,
        messages: List[dict],
        model: Optional[str],
        tools: Optional[List[dict]] = None,
        priority: int = 0,
        **kwargs
    ) -> openai.types.chat.ChatCompletion:
        """
        Asynchronously create chat completions and submit to scheduler.
        
        Args:
            messages (List[dict]): The messages to send to the model.
            sampling_params (dict): Parameters for sampling.
            model (Optional[str]): The model to use for completion.
        
        Returns:
            ChatCompletion: The response from the OpenAI API.
        """
        
        task = DispatchedSamplingTask(
            task=self.task,
            traj_id=self.record.traj_id,
            req_type="chatcompletions",
            request={
                "messages": messages,
                "model": model,
                "tools": tools,
                **kwargs
            },
            priority=priority,
        )
        # find a service to send
        service = await InferenceService.find(InferenceService.status == "UP").sort(+InferenceService.running_req_count).first_or_none()
        
        if service is None:
            raise RuntimeError("No available inference service found.")
        
        # increse the running request count
        await InferenceService.find_one(InferenceService.id == service.id).inc({InferenceService.running_req_count: 1})
        
        # create a request
        task.sampled_from = service
        task.status = DispatchedSamplingTask.Status.RUNNING

        await task.save()
        self.record.traj.append(Task.link_from_id(task.id))
        await self.record.save()
        
        # send the task to the service
        try:
            match service.connection_type:
                case "openai":
                    import openai
                    client = openai.AsyncOpenAI(
                        api_key=service.configs.api_key,
                        base_url=service.configs.base_url,
                    )
                    response = await client.chat.completions.create(
                        **task.request,
                        logprobs=True,
                        timeout=30
                    )
                    # logger.debug(f"Chat completion response: {response}")
                case _:
                    raise NotImplementedError(f"Connection type {service.connection_type} is not supported.")
        except Exception as e:
            # if there is an error, set the task status to failed and save the error message
            task.status = DispatchedSamplingTask.Status.FAILED
            task.response = str(e)
            await task.save()
            raise e
        finally:
            # decrease the running request count
            await InferenceService.find_one(InferenceService.id == service.id).inc({InferenceService.running_req_count: -1})
        
        task.status = DispatchedSamplingTask.Status.COMPLETED
        task.response = response
        task.finish_time = datetime.datetime.now()
        await task.save()
       
        return response
        
    @abstractmethod
    async def run() -> Record:
        """
        Run the sampler.
        This method should be implemented by subclasses to define the specific behavior of the sampler.
        """

        raise NotImplementedError("Subclasses must implement this method.")

class MCPSampler(AsyncSampler):
    async def run(self):
        """
        Run the MCP sampler.
        This method should be implemented by subclasses to define the specific behavior of the sampler.
        """
        
        ret = await self.create_chat_completions([{"role":"user","content":"hello!"}],model="gpt-4o-mini",temperature=1.0)
        
        return self.record

async def main():
    # 初始化db
    from beanie import init_beanie
    await init_beanie(
        connection_string="mongodb://10.0.1.8:27018,10.0.1.9:27018,10.0.1.11:27018/mcp",
        document_models=[
            InferenceService,
            DispatchedSamplingTask,
            Task,
            Record,
        ]
    )
    
    # 插入推理
    await InferenceService(
        models=["gpt-4o-mini"],
        connection_type="openai",
        configs=InferenceService.OpenAIConfig(
            api_key = "sk-R7x6jGyKSPYNFXMe49B90902Dd264735Bf0286Ee01F5EfC1", # 替换为你的OpenAI API密钥
            base_url = "https://toollearning.cn/v1/", # 替换为你的OpenAI API基础URL
            host="localhost",
            port=8000,
        ),
        status="UP",
    ).insert()
    
    # 插入测试任务
    task = await Task(
        description="Test Task",
    ).insert()
    
    async with MCPSampler() as sampler:
        record = await sampler.run()
        print(f"Record created with ID: {record.id}")
        print(f"Task ID: {record.task.id}")
        print(f"Trajectory ID: {record.traj_id}")
        print(f"Record Status: {record.status}")
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())