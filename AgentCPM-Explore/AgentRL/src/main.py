import multiprocessing as mp
import torch
import os

from transformers import HfArgumentParser
from configs import AgentTrainingConfig
from training.utils import setup_model_and_processor
from log import logger

# this is a example of custom convert function
# from training.datasets import convert_record_to_data
# from databases import Record
# async def convert_with_experience(
#     record:Record,
#     args: AgentTrainingConfig = None
# ):
#     async for data in convert_record_to_data(record, args):
#         if args.remove_experience:
#             data.messages[0]["content"] = data.messages[0]["content"].split("<EXPERIENCE>")[0]
#         yield data

# Here you could change to other sampler/dataset/score function
# from rollout.scienceworld import ScienceWorldSampler, ScienceWorldDataset
# from rollout.math import MATHSampler, MathDataset
from rollout.mcp import MCPSampler, MCPDataset, convert_record_to_data_mcp
from rollout.scheduler import SamplingScheduler

def sample_proc(args: AgentTrainingConfig):
    # Change sampler_class and dataset_class here
    sampler_class = MCPSampler
    dataset_class = MCPDataset
    logger.info(f"AsyncGRPOTrainer: Using {sampler_class.__name__} as the sampler class.")
    scheduler = SamplingScheduler(
        config=args,
        sampler_class=sampler_class,
        dataset=dataset_class(args.train_file) if args.train_file else None,
    )
    return scheduler.run()


def main(
    training_cfg: AgentTrainingConfig,
):
    torch.cuda.set_device(torch.device(f'''cuda:{os.environ.get("LOCAL_RANK","0")}'''))

    model, processing_class, full_state_dict = setup_model_and_processor(training_cfg)

    if training_cfg.trainer == "MiniCPMVTrainer":
        from training.trainer import MiniCPMVTrainer as TrainerCls
    elif training_cfg.trainer == "QwenVLTrainer":
        from training.trainer import QwenVLTrainer as TrainerCls
    elif training_cfg.trainer == "QwenMoETrainer":
        from training.trainer import QwenMoETrainer as TrainerCls
    else:
        from training.trainer import QwenTrainer as TrainerCls

    import patches.swanlab
    
    trainer = TrainerCls(
        model=model,
        args=training_cfg,
        convert_record_to_data_func=convert_record_to_data_mcp,  # Use MCP-specific convert function
        processing_class=processing_class,
        callbacks=[],
        full_state_dict=full_state_dict if training_cfg.cpu_ram_efficient_loading else None,
    )
    
    if training_cfg.enable_sampling:
        proc = mp.Process(target=sample_proc, args=(training_cfg,), daemon=True)
        proc.start()
        if training_cfg.eval_strategy != "no":
            eval_proc = mp.Process(target=sample_proc, args=(training_cfg, "eval"), daemon=True)
            eval_proc.start()

    try:
        trainer.train(training_cfg.resume_from_checkpoint)
    finally:
        trainer.clean()
        if training_cfg.enable_sampling:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    logger.warning("Sampling process did not terminate in time.")
                    proc.kill()
                    proc.join()
            proc.close()
            if training_cfg.eval_strategy != "no":
                if eval_proc.is_alive():
                    eval_proc.terminate()
                    eval_proc.join(timeout=5)
                    if eval_proc.is_alive():
                        logger.warning("Eval sampling process did not terminate in time.")
                        eval_proc.kill()
                        eval_proc.join()
                eval_proc.close()

if __name__ == "__main__":
    (training_cfg,) = HfArgumentParser(AgentTrainingConfig).parse_args_into_dataclasses()
    main(training_cfg)