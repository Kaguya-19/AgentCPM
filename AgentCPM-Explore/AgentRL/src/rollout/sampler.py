import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Literal, Type, Callable, Coroutine
import openai.types.chat
import datetime
from beanie import UpdateResponse
import beanie

from configs import AgentTrainingConfig
from log import logger
from databases import InferenceService, DispatchedSamplingTask, Task, Record, NoTaskAvailableError, DistributedCounter, DistributedLock
from pymongo import ReturnDocument
from tenacity import retry, stop_after_attempt, wait_fixed
from packaging import version

class AsyncSampler(ABC):
    """
    An abstract base class for running a asynchronous sampler.
    """

    def __init__(
        self,
        config: AgentTrainingConfig,
        *args,
        task_class: Type[Task] = Task,
        record_class: Type[Record] = Record,
        split: Literal["train", "valid", "test", "eval"] = "train",
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.split = split
        self.num_generations = config.num_generations if self.split == "train" else kwargs.get("override_num_generations", 1)
        self.task_class = task_class
        self.record_class = record_class
        assert issubclass(self.task_class, Task), "task_class must be a subclass of Task"
        assert issubclass(self.record_class, Record), "record_class must be a subclass of Record"
        self.record = None
    
    
    async def __aenter__(self):
        """Enter the asynchronous context manager."""
        # find a task to sample from
        self.task = await self.task_class.find_one(
            self.task_class.num_samples < self.num_generations,
            self.task_class.split == self.split,
            self.task_class.status == self.task_class.Status.RUNNING,
            with_children=True
        ).inc(
            {self.task_class.num_samples: 1},
            response_type = UpdateResponse.NEW_DOCUMENT
        )
        if self.task is None:
            self.task = await self.task_class.find_one(
                self.task_class.num_samples < self.num_generations,
                self.task_class.split == self.split,
                self.task_class.status == self.task_class.Status.CREATED,
                with_children=True
            ).update(
                {"$inc": {self.task_class.num_samples: 1}, "$set": {self.task_class.status: self.task_class.Status.RUNNING}},
                response_type = UpdateResponse.NEW_DOCUMENT
            )
        
        if self.task is None:
            raise NoTaskAvailableError("No task available for sampling. Please create a task first.")

        self.record = self.record_class(
            task=self.task,
            traj_id=self.task.num_samples - 1,
            split=self.split,
            status=self.record_class.Status.RUNNING,
        )
        # save the record
        await self.record.save()
        self.running_sampler = await DistributedCounter.create(name=f"running-sampler-{self.split}")
        await self.running_sampler.inc()

        self.infer_count = await DistributedCounter.create(name=f"infer")
        self.split_lock = await DistributedLock.create(name=self.split)
        self.global_step_counter = await DistributedCounter.create(name="global_step")

        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit the asynchronous context manager."""
        # Clean up resources if necessary
        if self.record is not None:
            if self.record.status == self.record_class.Status.RUNNING:
                if exc_type is None:
                    self.record.status = self.record_class.Status.COMPLETED
                else:
                    self.record.status = self.record_class.Status.FAILED
                    logger.warning(f"Record {self.record.traj_id} for task {self.record.task.id} failed with error: {exc_value}")
                    if self.config.resample_error_records:
                        # retrying
                        await self.task_class.find_one(
                            self.task_class.id == self.task.id, with_children=True
                        ).update({
                            "$inc": {"num_samples": -1},
                        })
                await self.record.save()
                
            if self.record.status == self.record_class.Status.COMPLETED:
                # scoring it
                self.record.status = self.record_class.Status.SCORING
                await self.record.save()
                try:
                    score = await self.evaluate_record(self.record)
                    if not isinstance(score, float):
                        score = float(score)
                except Exception as e:
                    import traceback
                    logger.error("Error when calculate score for record "+ str(self.record.id)+"\n"+traceback.format_exc())
                    score = 0.0
                self.record.score = score
                self.record.status = self.record_class.Status.SCORED
                await self.record.save()

                self.task: Task = await self.task_class.find_one(
                    self.task_class.id == self.task.id, with_children=True
                ).update({
                    "$push": {"scores": score}
                }, response_type=UpdateResponse.NEW_DOCUMENT)
                    
            # check whether all records are scored or failed
            status_to_count = [self.record_class.Status.SCORED]
            if not self.config.resample_error_records:
                status_to_count.append(self.record_class.Status.FAILED)
            valid_records_count = await self.record_class.find_many(
                {
                    "task.$id": self.task.id,
                    "status": {
                        "$in": status_to_count
                    }
                }, with_children=True
            ).count()

            if valid_records_count >= self.num_generations:
                # mark all records as ready
                await self.record_class.find_many(
                    {
                        "task.$id":self.task.id, 
                        "status": self.record_class.Status.SCORED
                    }, with_children=True
                ).update({
                    "$set": {
                        "status": self.record_class.Status.READY
                    }
                })

                await self.task.sync()
                self.task.status = self.task_class.Status.COMPLETED
                await self.task.save()

            self.record = None
        if self.task is not None:
            self.task = None
    
        await self.running_sampler.dec()

    @retry(stop=stop_after_attempt(7), wait=wait_fixed(5))
    async def create_chat_completions(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        tools: Optional[List[dict]] = None,
        priority: int = 0,
        timeout: Optional[int] = None,
        repeat_penalty: Optional[float] = 1.0,
        finish_status: Optional[DispatchedSamplingTask.Status] = DispatchedSamplingTask.Status.COMPLETED,
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
        if not self.config.sync_sampling:
            # for async sampling, we need to wait for the split lock release after model upadte
            # for sync sampling, we should ignore the lock and finish the job first
            await self.split_lock.wait()
        # logger.debug(f"Obtaining semaphore for task {self.task.id} with priority {priority}")
        await self.infer_count.wait_for(0, option="gt")
        # logger.debug(f"Semaphore obtained for task {self.task.id} with priority {priority}")
        
        task = DispatchedSamplingTask(
            task=self.task,
            traj_id=self.record.traj_id,
            req_type="chatcompletions",
            request={
                "messages": messages,
                "model": model if model is not None else self.config.model_name,
                "tools": tools,
                **kwargs
            },
            priority=priority,
        )
        # find a service to send
        if version.parse(beanie.__version__) >= version.parse("2.0.0"):
            service = await InferenceService.get_pymongo_collection().find_one_and_update(
                {"status": "UP"},
                {"$inc": {"running_req_count": 1}},
                sort=[("running_req_count", 1)],
                return_document=ReturnDocument.AFTER
            )
        else:
            service = await InferenceService.get_motor_collection().find_one_and_update(
                {"status": "UP"},
                {"$inc": {"running_req_count": 1}},
                sort=[("running_req_count", 1)],
                return_document=ReturnDocument.AFTER
            )
        if service is None:
            raise RuntimeError("No available inference service found.")
        service = InferenceService.model_validate(service)

        # create a request
        task.sampled_from = service
        task.status = DispatchedSamplingTask.Status.RUNNING

        await task.save()
        self.record.traj.append(DispatchedSamplingTask.link_from_id(task.id))
        await self.record.save()
        
        try:
            # check service is still up
            service = await InferenceService.find_one(InferenceService.id == service.id)
            if service.status != "UP":
                raise RuntimeError(f"Inference service {service.id} is not UP.")
            await self.global_step_counter.sync()
            task.created_at_step = self.global_step_counter.n

            # logger.debug(f"Dispatched task {task.id} to service {service.id} for sampling.")
            match service.connection_type:
                case "openai":
                    import openai
                    async with openai.AsyncOpenAI(
                        api_key=service.configs.api_key,
                        base_url=service.configs.base_url,
                    ) as client:
                        presence_penalty = self.config.presence_penalty if self.config.presence_penalty is not None else 0
                        frequency_penalty = self.config.frequency_penalty if self.config.frequency_penalty is not None else 0
                        response = await client.chat.completions.create(
                            **task.request,
                            logprobs=True,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                            max_completion_tokens=self.config.max_new_tokens,
                            timeout=timeout,
                        )
                    
                    assert response.choices[0].finish_reason not in ["abort"], f"Sampling for record {self.record.id} not finished properly with reason: {response.choices[0].finish_reason}"

                    logger.debug(f"Finish Chat Completions in service {service.id} for record {self.record.id}")
                    
                case _:
                    raise NotImplementedError(f"Connection type {service.connection_type} is not supported.")
        except Exception as e:
            # if there is an error, set the task status to failed and save the error message
            task.status = DispatchedSamplingTask.Status.FAILED
            task.response = str(e)
            task.finish_time = datetime.datetime.now()
            await task.save()
            if not isinstance(e, AssertionError):
                import traceback
                logger.error(f"Error occurred while creating chat completions for task {task.id} in service {service.id}:\n{traceback.format_exc()}")
            raise e
        finally:
            # decrease the running request count
            await InferenceService.find_one(InferenceService.id == service.id).inc({InferenceService.running_req_count: -1})
        

        if response.choices[0].finish_reason in ["length", "content_filter"]:
            logger.warning(f"Sampling for task {task.id} stopped due to {response.choices[0].finish_reason}")
            task.status = DispatchedSamplingTask.Status.FAILED
        else:
            task.status = finish_status if finish_status is not None else DispatchedSamplingTask.Status.COMPLETED
        task.response = response
        task.finish_time = datetime.datetime.now()
        await task.save()
       
        return response
        
    @abstractmethod
    async def run(self, task: Optional[Task] = None):
        """
        Run the sampler.
        This method should be implemented by subclasses to define the specific behavior of the sampler.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def evaluate_record(self, record: Record) -> float:
        """
        Calculate the score for the given record using the provided score function.
        Ones could also assign the score for each trajectory step in the record here.

        Args:
            record (Record): The record to calculate the score for.
        Returns:
            float: The calculated score.
        """
    
class StatefulSampler(AsyncSampler):
    async def __aenter__(self):
        """Enter the asynchronous context manager."""
        # find a task to sample from
        self.task = await self.task_class.find_one(
            self.task_class.num_samples < self.config.num_generations if self.split == "train" else self.task_class.num_samples < 1,
            self.task_class.split == self.split,
            self.task_class.status == self.task_class.Status.WAITING,
            with_children=True
        ).update(
            {"$inc": {self.task_class.num_samples: 1}, "$set": {self.task_class.status: self.task_class.Status.RUNNING}},
            response_type = UpdateResponse.NEW_DOCUMENT
        )
        
        if self.task is None:
            self.task = await self.task_class.find_one(
                self.task_class.num_samples < self.config.num_generations if self.split == "train" else self.task_class.num_samples < 1,
                self.task_class.split == self.split,
                self.task_class.status == self.task_class.Status.CREATED,
                with_children=True
            ).update(
                {"$inc": {self.task_class.num_samples: 1}, "$set": {self.task_class.status: self.task_class.Status.RUNNING}},
                response_type = UpdateResponse.NEW_DOCUMENT
            )
        
        if self.task is None:
            raise NoTaskAvailableError("No task available for sampling. Please create a task first.")


        self.record = self.record_class(
            task=self.task,
            traj_id=self.task.num_samples - 1,
            split=self.split,
        )
        # save the record
        await self.record.save()
        
        self.infer_count = await DistributedCounter.create(name=f"infer")
        
        return self
    

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit the asynchronous context manager."""
        await self.task.sync()
        self.task.status = self.task_class.Status.WAITING if self.task.num_samples < (self.config.num_generations if self.split == "train" else 1) else self.task_class.Status.COMPLETED
        await self.task.save()
        return await super().__aexit__(exc_type, exc_value, traceback)
        