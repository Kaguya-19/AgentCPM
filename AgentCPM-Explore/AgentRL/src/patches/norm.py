import torch
from typing import Optional
import functools
import types
import typing
import warnings
from torch.nn.utils.clip_grad import _tensor_or_tensors, _no_grad, Tensor,_group_tensors_by_device_and_dtype, _has_foreach_support, _device_has_foreach_support, _get_total_norm, _clip_grads_with_norm_
from torch.distributed.tensor import DTensor
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Replicate
from torch.distributed import ProcessGroup

@_no_grad
def _get_total_norm(
    tensors: _tensor_or_tensors,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    r"""Compute the norm of an iterable of tensors.

    The norm is computed over the norms of the individual tensors, as if the norms of
    the individual tensors were concatenated into a single vector.

    Args:
        tensors (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will be normalized
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of :attr:`tensors` is ``nan``, ``inf``, or ``-inf``.
            Default: ``False``
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the tensors (viewed as a single vector).
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    else:
        tensors = list(tensors)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)
    first_device = tensors[0].device
    grouped_tensors: dict[
        tuple[torch.device, torch.dtype], tuple[list[list[Tensor]], list[int]]
    ] = _group_tensors_by_device_and_dtype(
        [tensors]  # type: ignore[list-item]
    )  # type: ignore[assignment]

    norms: list[Tensor] = []
    for (device, _), ([device_tensors], _) in grouped_tensors.items():
        if (foreach is None and _has_foreach_support(device_tensors, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend(
                [torch.linalg.vector_norm(g, norm_type) for g in device_tensors]
            )

    # OLD IMPLEMENTATION can not handle DTensor correctly
    # total_norm = torch.linalg.vector_norm(
    #     torch.stack([norm.to(first_device) for norm in norms]), norm_type
    # )

    # group DTensors by their mesh to avoid device mismatch error during torch.stack
    dtensor_norms: dict[typing.Optional[torch.distributed.device_mesh.DeviceMesh], list[Tensor]] = {}
    normal_norms: list[Tensor] = []
    for norm in norms:
        if isinstance(norm, DTensor):
            mesh = norm.device_mesh
            if mesh not in dtensor_norms:
                dtensor_norms[mesh] = []
            dtensor_norms[mesh].append(norm)
        else:
            normal_norms.append(norm)
    

    total_norms = []
    if normal_norms:
        total_norms.append(
            torch.linalg.vector_norm(
                torch.stack([norm.to(first_device) for norm in normal_norms]), norm_type
            )
        )
    for mesh, mesh_norms in dtensor_norms.items():
        norm = torch.linalg.vector_norm(
            torch.stack([norm.to(first_device) for norm in mesh_norms]),
            norm_type,
        ).full_tensor()
        total_norms.append(norm)
    
    total_norm = torch.linalg.vector_norm(
        torch.stack(total_norms), norm_type
    )

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return total_norm


@_no_grad
def _clip_grads_with_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    total_norm: torch.Tensor,
    foreach: Optional[bool] = None,
) -> None:
    r"""Scale the gradients of an iterable of parameters given a pre-calculated total norm and desired max norm.

    The gradients will be scaled by the following calculation

    .. math::
        grad = grad * \frac{max\_norm}{total\_norm + 1e-6}

    Gradients are modified in-place.

    This function is equivalent to :func:`torch.nn.utils.clip_grad_norm_` with a pre-calculated
    total norm.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        total_norm (Tensor): total norm of the gradients to use for clipping
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        None
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad._local_tensor if isinstance(p.grad, DTensor) else p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    if len(grads) == 0:
        return
    grouped_grads: dict[
        tuple[torch.device, torch.dtype], tuple[list[list[Tensor]], list[int]]
    ] = _group_tensors_by_device_and_dtype([grads])  # type: ignore[assignment]

    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for (device, _), ([device_grads], _) in grouped_grads.items():
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device)

@_no_grad
def clip_grad_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
    pp_pg: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    r"""Clip the gradient norm of an iterable of parameters.

    The norm is computed over the norms of the individual gradients of all parameters,
    as if the norms of the individual gradients were concatenated into a single vector.
    Gradients are modified in-place.

    This function is equivalent to :func:`torch.nn.utils.get_total_norm` followed by
    :func:`torch.nn.utils.clip_grads_with_norm_` with the ``total_norm`` returned by ``get_total_norm``.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float, optional): type of the used p-norm. Can be ``'inf'`` for
            infinity norm. Default: 2.0
        error_if_nonfinite (bool, optional): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False
        foreach (bool, optional): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        is_generator = isinstance(parameters, types.GeneratorType)
        # prevent generators from being exhausted
        parameters = list(parameters)
        if is_generator and len(parameters) == 0:
            warnings.warn(
                "`parameters` is an empty generator, no gradient clipping will occur.",
                stacklevel=3,
            )
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    
    if pp_pg is not None:
        if norm_type == float("inf"):
            torch.distributed.all_reduce(
                total_norm,
                op=torch.distributed.ReduceOp.MAX,
                group=pp_pg,
            )
        else:
            norm = total_norm**norm_type
            torch.distributed.all_reduce(
                norm,
                op=torch.distributed.ReduceOp.SUM,
                group=pp_pg,
            )
            total_norm = norm**(1.0 / norm_type)
    
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm