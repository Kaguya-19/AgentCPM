import asyncio
import torch
from torch.utils.data import DataLoader, IterableDataset
from beanie import init_beanie,UpdateResponse
import sys
sys.path.append("./")
sys.path.append("./src")
from src.models import models_to_initialize, Record,InferenceService

loop = asyncio.get_event_loop()

loop.run_until_complete(init_beanie(
    connection_string="mongodb://10.0.1.8:27021,10.0.1.9:27021,10.0.1.10:27021,10.0.1.11:27021/math-test",
    document_models=models_to_initialize,
))

# loop.run_until_complete(
#     Record.find_all().set({"trained_count": 0})
# )
# pipeline = [
#     {
#         "$match": {
#             "trained_count": 0,
#             "status": Record.Status.READY
#         }
#     },
#     {
#         "$group": {
#             "_id": 0,
#             "total_traj_length": { 
#                 "$sum": { "$size": "$traj" }
#             }
#         }
#     }
# ]
# result = loop.run_until_complete(Record.aggregate(pipeline).to_list())

# result = loop.run_until_complete(InferenceService.aggregate(
#     [
#         {"$match": {InferenceService.status: "UP"}},
#         {"$sort": {InferenceService.running_req_count: 1}},
#         {"$limit": 1},
#         {"$set": {InferenceService.running_req_count: {"$add": [ "$running_req_count", 1 ]}}},
#         {"$merge": {"into": str(InferenceService), "on": "_id", "whenMatched": "replace", "whenNotMatched": "discard"}}
#     ]
# ).to_list())
from pymongo import ReturnDocument

result = loop.run_until_complete(InferenceService.get_motor_collection().find_one_and_update(
    {"status": "UP"},
    {"$inc": {"running_req_count": 1}},
    sort=[("running_req_count", 1)],
    return_document=ReturnDocument.AFTER
))
result = InferenceService.model_validate(result)

import pdb; pdb.set_trace()
    

