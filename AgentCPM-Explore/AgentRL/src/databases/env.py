from beanie import Document, Indexed, UpdateResponse
from typing import Any, Annotated, Optional, Type, TypeVar

T = TypeVar("T", bound="Environment")

class Environment(Document):
    """
    Beanie Document to represent an environment.
    """
    name: Annotated[str, Indexed(unique=True)]
    description: str = ""

    in_use: bool = False
    
    class Settings:
        is_root = True
        
    @classmethod
    async def create(cls: Type[T], name: str, in_use: bool = False) -> Optional[T]:
        """
        Create or Get a new semaphore instance with the given name.
        
        :param name: Unique name for the semaphore.
        :return: An instance of DistributedSemaphore.
        """
        doc = await cls.find_one({"name": name}, with_children=True)
        if doc is None:
            doc = cls(name=name, in_use=in_use)
            try:
                await doc.insert()
            except Exception as e:
                doc = await cls.find_one({"name": name}, with_children=True)
                if doc is None:
                    raise e
        return doc
    
    @classmethod
    async def get(cls: Type[T]) -> Optional[T]:
        """
        Get an available environment that is not in use.
        """
        doc = await cls.find_one(
            {"in_use": False}, with_children=True
        ).update(
            {"$set": {"in_use": True}},
            response_type=UpdateResponse.NEW_DOCUMENT
        )
        return doc
    
    async def set(self):
        """
        Set the environment as in use.
        """
        doc = await Environment.find_one(
            {"name": self.name, "in_use": False}, with_children=True
        ).update(
            {"$set": {"in_use": True}},
            response_type=UpdateResponse.NEW_DOCUMENT
        )
        return doc is not None
    
    async def reset(self):
        """
        Reset the environment to not in use.
        """
        doc = await Environment.find_one(
            {"name": self.name, "in_use": True}, with_children=True
        ).update(
            {"$set": {"in_use": False}},
            response_type=UpdateResponse.NEW_DOCUMENT
        )
        return doc is not None
    
    async def is_in_use(self) -> bool:
        """
        Check if the environment is currently in use.
        """
        doc = await Environment.find_one(
            {"name": self.name, "in_use": True}, with_children=True
        )
        return doc is not None
    

class OSWorldEnv(Environment):
    """
    Database model for OSWorld environments.
    """
    description: str = "OSWorld Environment for Virtual Desktop Automation"

    ip: str = ""
    port: int = 0
    os_type: str = "Ubuntu"
    screen_width: int = 1920
    screen_height: int = 1080
    
    @property
    def connect_string(self) -> str:
        return f"http://{self.ip}:{self.port}/mcp"
