import asyncio
import os
import time
import tqdm
import torch
import torch.distributed as dist
import requests
import multiprocessing
import multiprocessing.connection
from typing import Optional, Callable
from torch.distributed.tensor import DTensor
import json

from configs import AgentTrainingConfig
from log import logger
from databases import InferenceService
from .utils import init_process_group, obtain_local_port, _compute_tensor_nbytes

from sglang.srt.utils import MultiprocessingSerializer
# from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.entrypoints.http_server import launch_server, ServerArgs

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
except (ImportError, ModuleNotFoundError):
    from sglang.srt.patch_torch import monkey_patch_torch_reductions


def _compute_tensor_nbytes(t: torch.Tensor) -> int:
    try:
        return t.numel() * t.element_size()
    except Exception:
        # Fallback in unexpected cases
        return int(t.nelement()) * int(getattr(t, "element_size", lambda: 1)())


def fixed_launch_server_process(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from typing import List
    # We should replace the detokenize_logprob_tokens function with the one in the tokenizer manager
    def detokenize_logprob_tokens(
        self,
        token_logprobs_val: List[float],
        token_logprobs_idx: List[int],
        decode_to_text: bool,
    ):
        if not decode_to_text:
            return [
                (logprob, token_id, None)
                for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx)
            ]
        else:
            assert self.tokenizer is not None
            # token_texts = self.tokenizer.batch_decode(token_logprobs_idx)
            token_texts = self.tokenizer.convert_ids_to_tokens(token_logprobs_idx)
            return list(zip(token_logprobs_val, token_logprobs_idx, token_texts))

    TokenizerManager.detokenize_logprob_tokens = detokenize_logprob_tokens
    
    launch_server(
        server_args=server_args,
        pipe_finish_writer=pipe_finish_writer,
        launch_callback=launch_callback,
    )


class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class InferenceManager(metaclass=SingletonMeta):
    
    def __init__(
        self,
        config: AgentTrainingConfig,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """
        Initialize the Scheduler with a configuration and an optional event loop.

        Args:
            config (SamplerConfig): Configuration for the scheduler.
            loop (Optional[asyncio.AbstractEventLoop]): Event loop to use for asynchronous operations.
        """
        self.config = config
        self.loop = loop if loop is not None else asyncio.get_event_loop()
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # create a device mesh
        self.mesh = dist.device_mesh.init_device_mesh(
            device_type="cuda",
            mesh_shape=(self.world_size // self.config.inf_tp_size, self.config.inf_tp_size),
            mesh_dim_names=["dp", "tp"]
        )
        self.tp_rank = self.mesh.get_local_rank("tp")
        self.dp_rank = self.mesh.get_local_rank("dp")

        self.local_rank = int(os.environ.get("LOCAL_RANK",))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE",))
        
        # node_ranks = list(range(self.rank - self.local_rank, global_rank - self.local_rank + self.local_world_size))
        self.service_id = None
        self.weights_update_group = None

        dist.barrier(self.mesh.get_group("tp"))
        if self.tp_rank == 0:
            # Only the first process in the tensor parallel group will set up the service
            self.loop.run_until_complete(self._run_sglang_process(model_name_or_path=self.config.model_name_or_path))
        dist.barrier(self.mesh.get_group("tp"))
        self.loop.run_until_complete(self.release_memory_occupation(timeout=600.0))
    
    def is_running(self):
        """Check if the inference service is running."""
        return hasattr(self, "inference_proc") and self.inference_proc is not None and self.inference_proc.is_alive()

    async def _run_sglang_process(self, model_name_or_path):
        # prepare for the launch
        os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
        monkey_patch_torch_reductions()
        import torch
        torch.multiprocessing.set_start_method("spawn",force=True)
        # obtain local ip
        local_ip, port = obtain_local_port()
        
        if self.mesh["tp"].size() <= 1:
            # Parse CUDA_VISIBLE_DEVICES robustly without using eval
            raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            # Ensure TP group sees the same set of visible devices.
            # Compute a base local rank that aligns all TP ranks to the same window.
            tp_size = self.mesh["tp"].size()
            base_local_rank = max(0, self.local_rank - self.tp_rank)
            if raw_visible:
                # Accept formats like "0,1,2" or "0, 1, 2" and convert to a list of ints
                try:
                    devices = [int(x.strip()) for x in raw_visible.split(',') if x.strip() != ""]
                except ValueError:
                    # Fallback: if parsing fails, default to sequential local ranks
                    devices = list(range(self.local_world_size))

                # Select the window of devices for the current TP group starting from local_rank
                selected = []
                for i in range(tp_size):
                    idx = base_local_rank + i
                    if idx < len(devices):
                        selected.append(str(devices[idx]))
                    else:
                        # If provided list is shorter, continue with sequential ids as a safe fallback
                        selected.append(str(idx))
                os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(selected)
            else:
                # No restriction set; expose a contiguous range starting at local_rank for TP size
                os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
                    str(base_local_rank + i) for i in range(tp_size)
                )
        
        server_args=ServerArgs(
            model_path=model_name_or_path,
            host=local_ip,
            port=port,
            tp_size=self.config.inf_tp_size,
            ep_size=self.config.inf_ep_size,
            dp_size=1,
            enable_dp_attention=True,
            mem_fraction_static=self.config.inf_mem_ratio,
            trust_remote_code=True,
            dtype=self.config.torch_dtype,
            served_model_name=self.config.model_name,
            enable_memory_saver=True,
            enable_fp32_lm_head=True,
            load_format="dummy",
            log_level="warning",
            base_gpu_id=0 if self.mesh["tp"].size() <= 1 else self.dp_rank * self.config.inf_tp_size % self.local_world_size,
            # base_gpu_id=self.dp_rank * self.config.inf_tp_size % self.local_world_size,
            tool_call_parser=self.config.tool_call_parser,
            mm_attention_backend="fa3",
            attention_backend="fa3",
            gpu_id_step=1,
            random_seed=self.rank,
            preferred_sampling_params=self.config.preferred_sampling_params
        )
    
        p = multiprocessing.Process(target=fixed_launch_server_process, args=(server_args,))
        p.start()

        base_url = server_args.url()
        timeout = 300.0  # Increased timeout to 5 minutes for downloading large models
        start_time = time.perf_counter()

        with requests.Session() as session:
            while time.perf_counter() - start_time < timeout:
                try:
                    headers = {
                        "Content-Type": "application/json; charset=utf-8",
                        "Authorization": f"Bearer {server_args.api_key}",
                    }
                    response = session.get(f"{base_url}/health_generate", headers=headers)
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass

                if not p.is_alive():
                    raise Exception("Server process terminated unexpectedly.")

                time.sleep(2)
        
        self.inference_proc = p
        await asyncio.sleep(3)  # wait for the service to start
        logger.debug(f"Inference service started at {base_url}")

        if self.mesh["tp"].size() <= 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = raw_visible  # restore
        
        # send registration message to the scheduler
        service = InferenceService(
            models=[model_name_or_path],
            connection_type="openai",
            configs=InferenceService.OpenAIConfig(
                base_url=f"http://{local_ip}:{port}/v1",
                host=local_ip,
                port=port,
            ),
            status="UP"
        )
        
        await service.save()
        self.service_id = service.id
            
    def get_load(self,service:InferenceService):
        response = requests.get(
            f"http://{service.configs.host}:{service.configs.port}/get_load",
            json={}
        )
        response.raise_for_status()
        load_info = response.json()
        if isinstance(load_info, dict):
            if "load" in load_info:
                return load_info["load"]
            elif "num_reqs" in load_info:
                return load_info["num_reqs"] + load_info.get("num_waiting_reqs",0) + load_info.get("num_tokens",0)
            else:
                raise ValueError("Invalid load info format received from inference service.")
        elif isinstance(load_info, list):
            return sum(map(lambda x:x["num_reqs"]+x["num_waiting_reqs"]+x["num_tokens"],load_info))
        else:
            raise ValueError("Invalid load info format received from inference service.")
    
    def _pause(self, service: InferenceService):
        """Pause the inference service."""
        if service is None:
            return
        response = requests.post(
            f"http://{service.configs.host}:{service.configs.port}/pause_generation",
            json={}
        )
        response.raise_for_status()
        logger.debug("Inference service paused successfully.")

    def _continue(self, service: InferenceService):
        """Continue the inference service."""
        if service is None:
            return
        response = requests.post(
            f"http://{service.configs.host}:{service.configs.port}/continue_generation",
            json={}
        )
        response.raise_for_status()
        logger.debug("Inference service continued successfully.")

    async def release_memory_occupation(self, timeout: float = 90.0):
        """Release memory occupation of the inference service."""
        if self.service_id is not None:
            service = await InferenceService.get(self.service_id)
            service = await service.set(
                {InferenceService.status: "DOWN"}
            )

            start_time = time.perf_counter()
            MAX_PROBE_TIMES = int(os.environ.get("SG_SRT_MAX_PROBE_TIMES", 5))
            prob_times = MAX_PROBE_TIMES
            while prob_times > 0:
                prob_times -= 1
                while (load := self.get_load(service)) > 0 or service.running_req_count > 0:
                    if time.perf_counter() - start_time < timeout:
                        await asyncio.sleep(1)
                    else:
                        logger.warning("Releasing memory after waiting for {:.2f} seconds (load {}).".format(time.perf_counter() - start_time, load))
                        self._pause(service)
                        start_time = time.perf_counter()
                    await service.sync()
                    prob_times = MAX_PROBE_TIMES # reset prob_times if still busy
                await asyncio.sleep(2)

            try:
                response = requests.post(
                    f"http://{service.configs.host}:{service.configs.port}/release_memory_occupation",
                    json={}
                )
                response.raise_for_status()
                logger.debug("Memory occupation released successfully.")
            except requests.RequestException as e:
                logger.error(f"Failed to release memory occupation: {e}")
        dist.barrier(self.mesh.get_group("tp"))

        if self.service_id is not None and not self.is_running():
            # check whether the process is alive, sometimes will exit and should be fixed
            logger.warning(f"Inference service {self.service_id} is not running.")
            self.terminate_inference_service()
            # delete service
            await service.delete()

            # Restart the service or take appropriate action
            await self._run_sglang_process(self.config.model_name_or_path)
            try:
                response = requests.post(
                    f"http://{service.configs.host}:{service.configs.port}/release_memory_occupation",
                    json={}
                )
                response.raise_for_status()
                logger.debug("Memory occupation released successfully.")
            except requests.RequestException as e:
                logger.error(f"Failed to release memory occupation while restarting inference: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError("Inference Service failed (and failed to restart) while releasing memory occupation") from e

        dist.barrier(self.mesh.get_group("tp"))

    async def resume_memory_occupation(self):
        """Resume memory occupation of the inference service."""
        if self.service_id is not None:
            logger.debug(f"Resuming memory occupation for service {self.service_id}...")
            try:
                service = await InferenceService.get(self.service_id)
                
                response = requests.post(
                    f"http://{service.configs.host}:{service.configs.port}/resume_memory_occupation",
                    json={}
                )
                response.raise_for_status()
                logger.debug("Memory occupation resumed successfully.")
                service.status = "UP"
                await service.save()
            except requests.RequestException as e:
                logger.error(f"Failed to resume memory occupation: {e}")
        
        dist.barrier(self.mesh.get_group("tp"))

    # async def update_weights_from_dist(
    #     self,
    #     named_tensors: list[tuple[str, torch.Tensor]],
    #     device: Optional[torch.device],
    #     flush_cache: bool = True,
    #     abort_all_requests: bool = False,
    # ):
    #     if self.rank == 0:
    #         if self.weights_update_group is None:
    #             await self.init_weights_update_group(group_name="weights_update")
    #         weight_sync_tasks = []
    #         services = await InferenceService.all().to_list()
    #         for service in services:
    #             weight_sync_tasks.append(
    #                 asyncio.to_thread(
    #                     requests.post,
    #                     f"http://{service.configs.host}:{service.configs.port}/update_weights_from_dist",
    #                     json={
    #                         "names": [name for name, _ in named_tensors],
    #                         "dtypes": [str(tensor.dtype).replace("torch.", "") for _, tensor in named_tensors],
    #                         "shapes": [tensor.shape for _, tensor in named_tensors],
    #                         "group_name": self.group_name,
    #                         "flush_cache": flush_cache,
    #                         "abort_all_requests": abort_all_requests,
    #                     }
    #                 )
    #             )
    #         await asyncio.gather(*weight_sync_tasks)
            
    #         buffer_size = 512 * 1024 * 1024  # 512MB
            
    #         current_buffer_size = 0
    #         chunk = []
    #         pbar = tqdm.tqdm(total=len(named_tensors), desc="Broadcasting weights", unit="tensors")
    #         for name, tensor in named_tensors:
    #             tensor_size = tensor.numel() * tensor.element_size()
    #             if current_buffer_size + tensor_size > buffer_size and chunk:
                    
    #                 gpu_tensors = [(n, t.to(device=device)) for n, t in chunk]
                    
    #                 handles = []
    #                 for _, gpu_tensor in gpu_tensors:
    #                     handles.append(
    #                         dist.broadcast(gpu_tensor, src=0, group=self.weights_update_group, async_op=True)
    #                     )
    #                 for handle in handles:
    #                     handle.wait()
    #                     pbar.update()
                    
    #                 del gpu_tensors
    #                 torch.cuda.empty_cache()
                    
    #                 chunk = []
    #                 current_buffer_size = 0
                
    #             chunk.append((name, tensor))
    #             current_buffer_size += tensor_size
            
    #         if chunk:
    #             gpu_tensors = [(n, t.to(device=device)) for n, t in chunk]
                
    #             handles = []
    #             for _, gpu_tensor in gpu_tensors:
    #                 handles.append(
    #                     dist.broadcast(gpu_tensor, src=0, group=self.weights_update_group, async_op=True)
    #                 )
    #             for handle in handles:
    #                 handle.wait()
    #                 pbar.update()
                
    #             del gpu_tensors
    #             torch.cuda.empty_cache()
    #         pbar.close()
    #     dist.barrier()
        
    # async def init_weights_update_group(self, group_name: str):
    #     """Initialize a weight update group in the inference service."""
    #     if self.rank == 0:
    #         master_address = os.environ.get("MASTER_ADDR", "localhost")
    #         _, master_port = obtain_local_port()
            
    #         create_group_tasks = []
    #         services = await InferenceService.all().to_list()
    #         for idx, service in enumerate(services):
    #             create_group_tasks.append(
    #                 asyncio.to_thread(
    #                     requests.post,
    #                     f"http://{service.configs.host}:{service.configs.port}/init_weights_update_group",
    #                     json={
    #                         "master_address": master_address,
    #                         "master_port": master_port,
    #                         "rank_offset": idx*self.mesh["tp"].size() + 1, #TODO: should + 1?
    #                         "world_size": self.world_size,
    #                         "group_name": group_name,
    #                         "backend": "nccl",
    #                     }
    #                 )
    #             )
    #         self.weights_update_group = init_process_group(
    #             backend="nccl",
    #             init_method=f"tcp://{master_address}:{master_port}",
    #             world_size=self.world_size,
    #             rank=self.rank,
    #             group_name=group_name,
    #         )
    #         logger.info(f"Weight update group {group_name} created.")
    #         await asyncio.gather(*create_group_tasks)
    #         self.group_name = group_name
            
    #     dist.barrier()
    
    async def update_model_weights(self, state_dict: dict[str, torch.Tensor], device: Optional[torch.device], *, budget_bytes: Optional[int] = 512 * 1024 * 1024):
        """Update the model weights in the inference service."""
        # only rank 0 get full state dict
        if self.service_id is not None:
            service = await InferenceService.get(self.service_id)
        else:
            service = None

        pbar = tqdm.tqdm(total=len(state_dict), desc="Updating weights", unit="tensors", disable=self.rank != 0)
            
        # Accumulate tensors into batches based on budget_bytes. We still broadcast/gather per tensor,
        # but only send HTTP when a batch is formed.
        # Keep strong references to tensors to avoid being overwritten/freed before flush
        named_tensors_batch = []  # list[tuple[str, torch.Tensor]]
        current_batch_bytes = 0

        def flush_batch_if_needed(force: bool = False):
            nonlocal named_tensors_batch, current_batch_bytes
            # Serialize per tensor and gather across TP for rank 0 in TP group
            serialized_tensors = MultiprocessingSerializer.serialize(named_tensors_batch, output_str=True)
            if self.tp_rank == 0:
                gathered_serialized_tensors = [None for _ in range(self.mesh.get_group("tp").size())]
            else:
                gathered_serialized_tensors = None
            dist.gather_object(
                obj=serialized_tensors,
                object_gather_list=gathered_serialized_tensors,
                group=self.mesh.get_group("tp"),
                group_dst=0,
            )
            
            if not service or self.tp_rank != 0:
                named_tensors_batch = []
                current_batch_bytes = 0
                return
            
            if named_tensors_batch and (force or current_batch_bytes > (budget_bytes or 0)):
                pbar.set_description(f"Updating {len(named_tensors_batch)} weights with {current_batch_bytes / (1024*1024):.2f} MB")
                if self.tp_rank == 0:
                    response = requests.post(
                        f"http://{service.configs.host}:{service.configs.port}/update_weights_from_tensor",
                        json={
                            "serialized_named_tensors": gathered_serialized_tensors,
                            "load_format": None,
                            "flush_cache": False,
                        },
                    )
                    response.raise_for_status()
                pbar.update(len(named_tensors_batch))
                named_tensors_batch = []
                current_batch_bytes = 0

        for name, tensor in state_dict.items():
            # Prepare tensor across ranks
            if self.rank == 0:
                tensor = tensor.to(device).detach()
            else:
                tensor = torch.empty_like(tensor, dtype=tensor.dtype, device=device)
            dist.broadcast(tensor, src=0)
            named_tensors_batch.append((name, tensor))
            # Compute bytes of this tensor for batching
            tensor_nbytes = _compute_tensor_nbytes(tensor)
            current_batch_bytes += tensor_nbytes
            
            if current_batch_bytes > budget_bytes:
                # logger.debug(f"Flushing batch of size {current_batch_bytes} bytes.")
                # Accumulate for sending only on tp_rank 0
                flush_batch_if_needed(force=True)

        flush_batch_if_needed(force=True)
        pbar.close()
        self._continue(service)
        dist.barrier()

    def terminate_inference_service(self,*args, **kwargs):
        """Terminate the inference service."""
        if hasattr(self, "inference_proc") and self.inference_proc:
            if self.inference_proc.is_alive():
                self.inference_proc.terminate()
                if self.inference_proc.join(timeout=10):
                    self.inference_proc.kill()
            
            self.inference_proc.close()
            self.inference_proc = None
            logger.debug("Inference service process terminated successfully.")

    def clean(self):
        """Shutdown the inference service."""
        self.terminate_inference_service()