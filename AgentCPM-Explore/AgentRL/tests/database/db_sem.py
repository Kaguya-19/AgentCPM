import asyncio
import time
from beanie import init_beanie,UpdateResponse
import sys
import random
sys.path.append("./")  
sys.path.append("./src")
from src.models import models_to_initialize, DistributedLock, DistributedCounter

loop = asyncio.get_event_loop()

loop.run_until_complete(init_beanie(
    connection_string="mongodb://10.0.1.8:27021,10.0.1.9:27021,10.0.1.10:27021,10.0.1.11:27021/test",
    document_models=models_to_initialize,
))

loop.run_until_complete(DistributedCounter.find({}
).delete())


async def main(id):
    counter = await DistributedCounter.create(
        name="test_counter",
    )
    for _ in range(32):
        await counter.inc()
        print("[{}] inc".format(id))
        s_time = time.time()
        await asyncio.sleep(random.random()*3)
        print("[{}] sleep {} seconds".format(id,time.time() - s_time))
        await counter.dec()
        print("[{}] dec".format(id))

        await counter.wait_for(0,"eq")
        print("[{}] released".format(id))
        await asyncio.sleep(5)


coros = [
    main(i)
    for i in range(8)
]

loop.run_until_complete(asyncio.gather(*coros))