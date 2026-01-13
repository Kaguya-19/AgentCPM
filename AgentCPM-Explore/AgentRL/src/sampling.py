from rollout.scheduler import SamplingScheduler
from rollout.mcp import MCPSampler, MCPDataset

from configs import AgentTrainingConfig
import os
from contextlib import nullcontext
from functools import partial
from transformers import HfArgumentParser

# Here you could change to other sampler/dataset/score function
def sample_proc(args: AgentTrainingConfig, split="train"):
    scheduler = SamplingScheduler(
        config=args,
        sampler_class=MCPSampler,
        dataset=MCPDataset(args.train_file) if args.train_file else None,
        split=split,
    )
    scheduler.run()
        

if __name__ == "__main__":
    (training_cfg,) = HfArgumentParser(AgentTrainingConfig).parse_args_into_dataclasses()
    sample_proc(training_cfg)