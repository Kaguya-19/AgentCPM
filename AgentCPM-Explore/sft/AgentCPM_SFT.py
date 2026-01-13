import os
import pickle
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, get_cosine_schedule_with_warmup
from trl import SFTConfig
from torch.optim import AdamW

from accelerate import Accelerator

warnings.filterwarnings("ignore")


def preprocess_data_truncate(data, tokenizer, max_seq_length: int) -> Dataset:
    """Tokenize with truncation to max_seq_length and report length percentiles."""
    input_ids_list = []
    input_lengths = []
    for input_text in tqdm(data, desc="tokenizing(trunc=True)"):
        tokenized = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        )
        ids = tokenized["input_ids"][0].tolist()
        input_ids_list.append(ids)
        input_lengths.append(len(ids))

    quantiles = [20, 40, 60, 80, 90, 95]
    percentiles = np.percentile(input_lengths, quantiles)
    print("\nInput Length Percentiles:")
    for q, p in zip(quantiles, percentiles):
        print(f"{q}th percentile: {int(p)} tokens")

    df = pd.DataFrame({"input_ids": input_ids_list})
    return Dataset.from_pandas(df)


def preprocess_data_drop(data, tokenizer, max_seq_length: int) -> Dataset:
    """Tokenize without truncation, then drop samples >= max_seq_length and report percentiles."""
    input_ids_list = []
    input_lengths = []
    for input_text in tqdm(data, desc="tokenizing(trunc=False)"):
        tokenized = tokenizer(input_text, return_tensors="pt", truncation=False)
        ids = tokenized["input_ids"][0].tolist()
        input_ids_list.append(ids)
        input_lengths.append(len(ids))

    quantiles = [20, 40, 60, 80, 90, 95]
    percentiles = np.percentile(input_lengths, quantiles)
    print("\nInput Length Percentiles:")
    for q, p in zip(quantiles, percentiles):
        print(f"{q}th percentile: {int(p)} tokens")

    input_ids_list = [s for s in input_ids_list if len(s) < max_seq_length]
    print("len(dataset) after drop:", len(input_ids_list))

    df = pd.DataFrame({"input_ids": input_ids_list})
    return Dataset.from_pandas(df)


def find_all_subseq_indices(seq: torch.Tensor, sub: torch.Tensor) -> list[int]:
    """Return all starting indices where sub appears in seq."""
    n, m = len(seq), len(sub)
    if m == 0 or m > n:
        return []
    hits = []
    for i in range(n - m + 1):
        if torch.equal(seq[i : i + m], sub):
            hits.append(i)
    return hits


class CollatorWithIgnoreIndex:
    """
    Build padded batch tensors and labels with ignore_index (-100).

    supervise_mode:
      - "all": supervise all assistant spans between start/end markers
      - "last": supervise only the last assistant span
    """

    def __init__(
        self,
        tokenizer,
        loss_start_token: str,
        loss_end_token: str,
        include_end_token: bool = True,
        report_supervised_ratio: bool = True,
        supervise_mode: str = "all",
    ):
        self.tok = tokenizer
        self.start_ids = torch.tensor(
            self.tok.encode(loss_start_token, add_special_tokens=False),
            dtype=torch.long,
        )
        self.end_ids = torch.tensor(
            self.tok.encode(loss_end_token, add_special_tokens=False),
            dtype=torch.long,
        )
        self.include_end_token = include_end_token
        self.report = report_supervised_ratio
        self.supervise_mode = supervise_mode

        print("pad token id:", self.tok.pad_token_id)
        print("start_ids:", self.start_ids)
        print("end_ids:", self.end_ids)
        print("supervise_mode:", supervise_mode)

    def __call__(self, features):
        pad_id = self.tok.pad_token_id
        if pad_id is None:
            self.tok.pad_token = self.tok.eos_token
            pad_id = self.tok.pad_token_id

        item_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        padded = pad_sequence(item_ids, batch_first=True, padding_value=pad_id)
        attention_mask = (padded != pad_id).long()

        labels = torch.full_like(padded, fill_value=-100)

        total_supervised = 0
        total_tokens = 0

        for b in range(padded.size(0)):
            ids = padded[b]
            valid_len = int(attention_mask[b].sum().item())
            seq = ids[:valid_len]

            start_hits = find_all_subseq_indices(seq, self.start_ids)
            end_hits_sorted = sorted(find_all_subseq_indices(seq, self.end_ids))

            def nearest_end_after(start_idx: int) -> int | None:
                for eh in end_hits_sorted:
                    if eh > start_idx:
                        return eh
                return None

            def supervise_span(start_idx: int):
                nonlocal total_supervised
                content_start = start_idx + len(self.start_ids)
                e = nearest_end_after(start_idx)

                if e is None:
                    content_end_exclusive = valid_len
                else:
                    content_end_exclusive = (
                        min(e + len(self.end_ids), valid_len) if self.include_end_token else e
                    )

                if content_end_exclusive > content_start:
                    labels[b, content_start:content_end_exclusive] = ids[content_start:content_end_exclusive]
                    total_supervised += (content_end_exclusive - content_start)

            if self.supervise_mode == "last":
                if start_hits:
                    supervise_span(start_hits[-1])
            else:
                for s in start_hits:
                    supervise_span(s)

            total_tokens += valid_len

        if self.report:
            ratio = (total_supervised / max(total_tokens, 1)) if total_tokens > 0 else 0.0
            print(f"[Collator] Supervised token ratio: {ratio:.3f} ({total_supervised}/{total_tokens})")
            if total_supervised == 0:
                print("[Collator][WARN] No supervised tokens found. Check your marker tokens in samples.")

        return {
            "input_ids": padded,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class IgnoreIndexTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
        **kwargs,
    ):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        logits = outputs.logits.float()

        if labels is None:
            loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else None
            if loss is None:
                loss = torch.tensor(0.0, device=logits.device)
            return (loss, outputs) if return_outputs else loss

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        num_items = num_items_in_batch
        if num_items is None:
            num_items = torch.tensor(int((shift_labels != -100).sum().item()), device=shift_logits.device)

        if hasattr(num_items, "device") and num_items.device != shift_logits.device:
            num_items = num_items.to(shift_logits.device)

        num_items = torch.clamp(num_items, min=1)
        loss = loss / num_items
        loss = loss * self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


def maybe_init_swanlab(experiment_name: str):
    """
    Optional SwanLab login.
    Set SWANLAB_API_KEY in your environment to enable.
    """
    api_key = os.getenv("SWANLAB_API_KEY", "").strip()
    if not api_key:
        return
    try:
        import swanlab
        swanlab.login(api_key=api_key, save=False)
    except Exception as e:
        print(f"[WARN] SwanLab init failed: {e}")


def main():
    max_seq_length = 21000
    supervise_mode = "all"  # "all" or "last"

    model_path = "PATH_OR_HF_MODEL_ID"
    experiment_name = "YOUR_EXPERIMENT_NAME"
    output_path = f"./ckpts/{experiment_name}"

    sft_data_path_list = [
        "./input/your_sft_data.pkl",
    ]

    learning_rate = 2e-5
    num_train_epochs = 3
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 4
    warmup_ratio = 0.03

    loss_start_token = "<|im_start|>assistant"
    loss_end_token = "<|im_end|>"

    print(f"Experiment: {experiment_name}")
    maybe_init_swanlab(experiment_name)

    raw_list = []
    for sft_data_path in sft_data_path_list:
        with open(sft_data_path, "rb") as fp:
            temp = pickle.load(fp)
        raw_list += temp

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = preprocess_data_drop(raw_list, tokenizer, max_seq_length)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="flash_attention_2",
    )

    if os.getenv("PYCHARM_HOSTED") != "1":
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=6))

    _ = Accelerator(mixed_precision="bf16")

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    steps_per_epoch = max(1, len(dataset) // (per_device_train_batch_size * world_size))
    total_training_steps = max(1, (steps_per_epoch * num_train_epochs) // gradient_accumulation_steps)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(total_training_steps * warmup_ratio),
        num_training_steps=total_training_steps,
    )

    training_args = SFTConfig(
        output_dir=output_path,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        bf16=True,
        logging_steps=1,
        gradient_checkpointing=True,
        save_strategy="epoch",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=1.0,
        report_to="swanlab",
        run_name=experiment_name,
    )

    data_collator = CollatorWithIgnoreIndex(
        tokenizer=tokenizer,
        loss_start_token=loss_start_token,
        loss_end_token=loss_end_token,
        include_end_token=True,
        report_supervised_ratio=True,
        supervise_mode=supervise_mode,
    )

    trainer = IgnoreIndexTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()


if __name__ == "__main__":
    main()
