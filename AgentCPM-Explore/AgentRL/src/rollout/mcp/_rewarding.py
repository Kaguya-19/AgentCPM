"""
MCP-specific rewarding logic for task completion and record filtering.
"""
from log import logger
import random
from typing import TYPE_CHECKING, Any, Dict
from beanie.operators import In
from beanie import UpdateResponse

if TYPE_CHECKING:
    from .models import MCPTask
    from databases import Record
    from configs import AgentTrainingConfig




async def apply_mcp_rewarding_logic(
    task: "MCPTask",
    record_class: type,
    task_class: type,
    config: Any
) -> None:
    """
    Apply MCP-specific rewarding logic similar to tool_rewarding.py.
    This includes variance check, balance_sample, filter_wrong, etc.
    
    Args:
        task: The MCPTask instance
        record_class: The Record class type
        task_class: The Task class type
        config: The AgentTrainingConfig instance
    """
    # Check if all scores are identical (variance check)
    # If max - min < 0.1, abandon the task as it provides no useful variance
    if max(task.scores) - min(task.scores) < 0.1:
        task.status = task_class.Status.COMPLETED
        await record_class.find_many(
            {"task.$id": task.id}, with_children=True
        ).update({"$set": {"status": record_class.Status.ABANDONED}})
        await task.save()
        return
    
    # Balance sample logic
    # This method will keep positive samples and negative samples in a balanced way.
    if getattr(config, "balance_sample", False):
        # Get wrong records (score == 0)
        pipeline = [
            {"$match": {"task.$id": task.id, "score": 0}},
            {"$addFields": {"traj_length": {"$size": "$traj"}}},
            {"$sort": {"traj_length": 1}}
        ]
        dict_records = await record_class.aggregate(pipeline).to_list()
        wrong_records = [record_class.model_validate(rec) for rec in dict_records]
        
        # Get right records (score != 0)
        pipeline = [
            {"$match": {"task.$id": task.id, "score": {"$ne": 0}}},
            {"$addFields": {"traj_length": {"$size": "$traj"}}},
            {"$sort": {"traj_length": 1}}
        ]
        dict_records = await record_class.aggregate(pipeline).to_list()
        right_records = [record_class.model_validate(rec) for rec in dict_records]
        
        # Filter out failed_error records
        valid_wrong_records = []
        for rec in wrong_records:
            final_answer = rec.meta_infos.get("final_answer", "")
            if final_answer == "failed_error":
                rec.status = record_class.Status.ABANDONED
                await rec.save()
            else:
                valid_wrong_records.append(rec)
        
        # Balance sampling
        min_len = min(len(valid_wrong_records), len(right_records))
        random.shuffle(valid_wrong_records)
        random.shuffle(right_records)
        
        for i, rec in enumerate(valid_wrong_records):
            if i < min_len + 1:
                rec.status = record_class.Status.READY
            else:
                rec.status = record_class.Status.ABANDONED
            await rec.save()
        
        for i, rec in enumerate(right_records):
            if i < min_len + 2:
                rec.status = record_class.Status.READY
            else:
                rec.status = record_class.Status.ABANDONED
            await rec.save()
        
        task.status = task_class.Status.COMPLETED
        await task.save()
        return
    
    # Default rewarding logic:
    # 1. Set all records to READY
    # 2. Abandon failed_error records to avoid training two many records encountered environment errors.
    
    # Step 1: Set all records to READY
    await record_class.find_many(
        {"task.$id": task.id}, with_children=True
    ).update({"$set": {"status": record_class.Status.READY}})
    
    all_records = await record_class.find_many(
        {"task.$id": task.id}, with_children=True
    ).to_list()
    all_records = [record_class.model_validate(rec) for rec in all_records]
    
    # Step 2: Abandon all failed_error records
    failed_error_records = [rec for rec in all_records if rec.meta_infos.get("final_answer", "") == "failed_error"]
    for rec in failed_error_records:
        rec.status = record_class.Status.ABANDONED
        await rec.save()
    
    #Step 3: Reserve necessary context_error records to avoid losing important information.
    # Randomly reserve a specified number of context error records (length_limit_error and turns_limit_error)
    context_error_types = ["length_limit_error", "turns_limit_error"]
    context_error_records = [rec for rec in all_records if rec.meta_infos.get("final_answer", "") in context_error_types]
    
    if len(context_error_records) > 0:
        context_error_records = [record_class.model_validate(rec) for rec in context_error_records]
        # Randomly shuffle and keep only the specified number
        random.shuffle(context_error_records)
        reserve_count = getattr(config, "reserve_context_error_count", 0)
        reserve_count = max(0, min(reserve_count, len(context_error_records)))  # Clamp to valid range
        
        for i, rec in enumerate(context_error_records):
            if i < reserve_count:
                # Keep these records as READY (already set in Step 1)
                pass
            else:
                # Abandon the rest
                rec.status = record_class.Status.ABANDONED
                await rec.save()
        logger.info(f"Reserved {reserve_count} out of {len(context_error_records)} context error records.")
    
    task.status = task_class.Status.COMPLETED
    await task.save()

