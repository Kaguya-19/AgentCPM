from configs import AgentTrainingConfig
from .sampler import InferenceService,DispatchedSamplingTask,Task,Record,NoTaskAvailableError
from .semaphore import DistributedLock,DistributedCounter
from .dataset import RecordData,DataLabels,DBRecordData
from .env import Environment, OSWorldEnv
import beanie
import asyncio
from packaging import version
from typing import Type, TypeVar
T = TypeVar("T", bound=beanie.Document)

__all__ = [
    "InferenceService",
    "DispatchedSamplingTask",
    "DistributedLock",
    "DistributedCounter",
    "Task",
    "Record",
    "NoTaskAvailableError",
    "RecordData",
    "DataLabels",
    "init_databases",
    "register_new_model",
    "init_data_models",
    "Environment",
]

DATAMODELS_TO_INIT = [
    InferenceService,
    DispatchedSamplingTask,
    DistributedLock,
    DistributedCounter,
    Task,
    Record,
    Environment,
    OSWorldEnv,
    DBRecordData
]

async def init_databases(
    config: AgentTrainingConfig,
):
    """
    Initialize the databases and other services.
    
    Args:
        config (GRPOTrainingConfig): The training configuration.
    """
    if version.parse(beanie.__version__) >= version.parse("2.0.0"):
        await beanie.init_beanie(
            connection_string=config.db_connection_string,
            document_models=DATAMODELS_TO_INIT,
        )
    else:
        await beanie.init_beanie(
            connection_string=config.db_connection_string,
            document_models=DATAMODELS_TO_INIT,
            multiprocessing_mode=True
        )


def register_new_model(cls: Type[T]) -> Type[T]:
    """
    Register a new model to be initialized with Beanie.
    """
    if cls not in DATAMODELS_TO_INIT:
        DATAMODELS_TO_INIT.append(cls)
    else:
        raise ValueError(f"Model {cls.__name__} is already registered.")
    return cls

async def init_data_models(clean_all:bool=False):
    coros = [
        InferenceService.find_all(with_children=True).delete(),
        DistributedCounter.find_all(with_children=True).delete(),
        Environment.find_all(with_children=True).delete(),
    ]
    if clean_all:
        coros.extend([
            DispatchedSamplingTask.find_all(with_children=True).delete(),
            Task.find_all(with_children=True).delete(),
            Record.find_all(with_children=True).delete()
        ])
    
    return await asyncio.gather(*coros)