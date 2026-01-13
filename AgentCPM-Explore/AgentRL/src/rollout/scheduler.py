import time
import random
import asyncio
from asyncio import CancelledError
from typing import Type, Literal, Callable, Coroutine, Any
import inspect
from torch.utils.data import Dataset,IterableDataset

from log import logger
from configs import AgentTrainingConfig
from databases import InferenceService, DistributedLock, Task, DistributedCounter, init_databases

from .sampler import AsyncSampler, NoTaskAvailableError

class SamplingScheduler:
    """Control the concurrent sampling tasks."""
    def __init__(
        self,
        config: AgentTrainingConfig,
        sampler_class: Type[AsyncSampler],
        dataset: Dataset|IterableDataset,
        split: Literal["train", "valid", "test", "eval"] = "train",
    ):
        super().__init__()
        self.config = config
        self.split = split
        self.sampler_class = sampler_class
        self.stop_signal = False
        self.dataset = dataset
        logger.info(f"Using {sampler_class.__name__} as the sampler class.")
    
    async def start_new_epoch(self, epoch: DistributedCounter):
        """
        Start a new epoch for sampling.
        
        This method is responsible for starting a new epoch for sampling.
        """
        lock = await DistributedLock.create(name = f"epoch-update-{self.split}")
        if (await lock.set()):
            await epoch.inc()
            logger.info(f"Epoch lock acquired for split {self.split}. Pushing new data for epoch {epoch.n}.")
            global_step_counter = await DistributedCounter.create(name="global_step")
            insert_coros = []
            if isinstance(self.dataset, Dataset):
                # import pdb; pdb.set_trace()
                for i in range(len(self.dataset)):
                    task = self.dataset[i]
                    if isinstance(task, Task):
                        # logger.debug(f"Rank {self.rank} pushing task {task.description} into the database.")
                        task.split = self.split
                        task.epoch = epoch.n
                        task.added_step = global_step_counter.n
                        insert_coros.append(asyncio.create_task(task.insert()))
                    else:
                        raise TypeError("Dataset must contain Task instances.")
            elif isinstance(self.dataset, IterableDataset):
                # For IterableDataset, we need to iterate over it
                for task in self.dataset:
                    if isinstance(task, Task):
                        task.split = self.split
                        task.epoch = epoch.n
                        task.added_step = global_step_counter.n
                        insert_coros.append(asyncio.create_task(task.insert()))
                    else:
                        raise TypeError("Dataset must contain Task instances.")
            else:
                raise TypeError("Dataset must be either a Dataset or IterableDataset instance.")
            await asyncio.wait(insert_coros)
            await lock.reset()
        else:
            logger.warning(f"Failed to acquire epoch lock for split {self.split}. Another process may have acquired it.")
            await lock.wait()
            
        return True
        
    def stop(self):
        """
        Stop the sampling scheduler.
        
        This method stops the sampling scheduler by terminating the thread.
        """
        self.stop_signal = True

    async def _exec_sampler(self):
        async with self.sampler_class(
            config=self.config,
            split=self.split,
        ) as sampler:
            if "task" in inspect.getargs(sampler.run.__code__).args:
                await sampler.run(task=sampler.task)
            else:
                await sampler.run()
    
    async def _scheduler_loop(self):
        logger.info(f"Sampling Scheduler on Split {self.split} Started.")
        pending: set[asyncio.Task] = set()
        count = 0
        
        split_lock = await DistributedLock.create(name = self.split, n=1)
        running_scheduler = await DistributedCounter.create(f"{self.split}-running")
        epoch = await DistributedCounter.create(f"{self.split}-epoch")

        while not self.stop_signal:
            await split_lock.wait()
            
            if self.split == "train" and await epoch.check(self.config.num_train_epochs,"gt"):
                logger.info("Maximum training epochs reached.")
                break
            
            if await running_scheduler.check(0, "eq"):
                create_new_task = await self.start_new_epoch(epoch)
                if not create_new_task:
                    break
            else:
                create_new_task = True

            await running_scheduler.inc()
            while not self.stop_signal and \
                (create_new_task or pending) and \
                (await split_lock.wait()):

                if create_new_task and len(pending) < self.config.max_concurrent_samples_per_process:
                    total_sampling = await Task.find(Task.status == "running", Task.split == self.split).count()
                    logger.info(f"Total running tasks for split {self.split}: {total_sampling}")
                    if total_sampling > 12:
                        await asyncio.sleep(1)
                        continue
                    avg_running_req = await InferenceService.find(InferenceService.status == "UP").avg(InferenceService.running_req_count)
                    if avg_running_req is not None and avg_running_req < self.config.target_concurrency:
                        logger.debug(f"Average running requests: {avg_running_req}, starting new sampler task.")
                        task = asyncio.create_task(
                            self._exec_sampler(),
                            name=f"Sampler-{self.split}-{count}"
                        )
                        pending.add(task)
                        count += 1
                
                if not pending:
                    await asyncio.sleep(1)
                    continue

                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=3
                )
                
                for task in done:
                    try:
                        task.result()
                    except NoTaskAvailableError as e:
                        logger.debug(f"{e}")
                        # here we should start new epoch for sampling
                        create_new_task = False
                    except CancelledError as e:
                        logger.warning(f"Sampler task was cancelled: {e}")
                    except Exception as e:
                        logger.error(f"Error occurred while processing task result: {e}")
                        import traceback
                        logger.error(traceback.format_exc())

                if self.stop_signal:
                    for task in pending:
                        if not task.done():
                            # logger.debug(f"Cancelling pending task: {task}")
                            task.cancel()
                    break
    
            await running_scheduler.dec()
            await running_scheduler.wait_for(0,"eq")
            await split_lock.set()

    def run(self):
        """
        Run the sampling scheduler.
        
        This method runs the sampling scheduler in the current thread.
        """
        time.sleep(random.uniform(0, 1))  # wait for a random
        
        loop = asyncio.new_event_loop()
        loop.run_until_complete(init_databases(self.config))
        loop.run_until_complete(self._scheduler_loop())
        loop.close()
        
        if self.stop_signal:
            logger.info("Sampling scheduler stopped by signal.")
            return
        logger.info("Sampling scheduler finished.")