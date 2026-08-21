"""Independent SepPrune reconstruction for A-FRCNN-12 and SuDoRM-RF.

The released SepPrune scripts import private ``*_pruned.py`` model files that
are not present in the public repository.  This module reconstructs only the
channel dependencies that are explicitly visible in ``mask_learning_*.py``
and ``finetune_*.py``:

* A-FRCNN-12: the four fusion outputs and the corresponding input groups of
  ``last_layer``;
* SuDoRM-RF: the 512-channel hidden path inside each of its 16 UConv blocks.

All non-mask parameters are frozen during mask search.  Physical pruning
copies every dependent convolution, bias, normalization and activation
parameter; the released fine-tuning scripts copy convolution weights only.
That omission would silently randomize surviving normalization and output
projection parameters, so it is deliberately not reproduced here.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .tdanet_seprune import DifferentiableChannelMask


class MaskedChannelOutput(nn.Module):
    """Apply a learnable channel mask after an existing feature-producing layer."""

    def __init__(
        self,
        base: nn.Module,
        channels: int,
        *,
        epsilon: float,
        temperature: float,
        seed: int,
        initial_probability_low: float = 0.55,
        initial_probability_high: float = 0.85,
    ) -> None:
        super().__init__()
        self.base = base
        self.mask = DifferentiableChannelMask(
            channels,
            epsilon=epsilon,
            temperature=temperature,
            seed=seed,
            initial_probability_low=initial_probability_low,
            initial_probability_high=initial_probability_high,
        )
        self.stochastic = True

    def forward(self, x: Tensor) -> Tensor:
        return self.mask(self.base(x), stochastic=self.stochastic)


class BudgetedMaskedChannelOutput(nn.Module):
    """Apply a mask supplied by a model-level parameter-budget controller."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.current_mask: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        if self.current_mask is None:
            raise RuntimeError("Budgeted mask was not prepared before model forward")
        return self.base(x) * self.current_mask.view(1, -1, 1)

    def __deepcopy__(self, memo: dict[int, object]) -> "BudgetedMaskedChannelOutput":
        # ``current_mask`` can be a non-leaf STE tensor carrying a live graph;
        # it is transient and must never enter the physical model copy.
        copied = type(self)(copy.deepcopy(self.base, memo))
        memo[id(self)] = copied
        return copied


class ParameterBudgetController(nn.Module):
    """Gumbel binary masks projected to a physical parameter budget.

    The paper does not release the mechanism that prevents task-only masks
    from opening every channel.  This controller makes that missing constraint
    explicit: the hard forward masks are greedily projected to the reported
    parameter budget, while the backward pass follows independent Binary
    Concrete probabilities (straight-through estimator).
    """

    def __init__(
        self,
        costs: Sequence[Tensor],
        *,
        original_parameters: int,
        target_parameters: int,
        temperature: float,
        seed: int,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < target_parameters < original_parameters:
            raise ValueError("target parameters must be between zero and original size")
        cpu_costs = [cost.detach().to(device="cpu", dtype=torch.int64) for cost in costs]
        if not cpu_costs or any(cost.ndim != 1 or torch.any(cost <= 0) for cost in cpu_costs):
            raise ValueError("Each layer needs a positive one-dimensional channel-cost vector")
        total_prunable = sum(int(cost.sum()) for cost in cpu_costs)
        self.fixed_parameters = int(original_parameters - total_prunable)
        self.variable_budget = int(target_parameters - self.fixed_parameters)
        minimum = sum(int(cost.min()) for cost in cpu_costs)
        if self.variable_budget < minimum or self.variable_budget > total_prunable:
            raise ValueError(
                f"Infeasible variable budget {self.variable_budget}; range=[{minimum}, {total_prunable}]"
            )
        self.original_parameters = int(original_parameters)
        self.target_parameters = int(target_parameters)
        self.temperature = float(temperature)
        self.logits = nn.ParameterList()
        generator = torch.Generator().manual_seed(seed)
        for index, cost in enumerate(cpu_costs):
            self.register_buffer(f"cost_{index}", cost, persistent=True)
            # Symmetric, low-amplitude initialization avoids a built-in keep bias.
            initial = torch.empty(cost.numel()).uniform_(-0.01, 0.01, generator=generator)
            parameter = nn.Parameter(initial)
            parameter.register_hook(lambda gradient: gradient.clamp_(-1.0, 1.0))
            self.logits.append(parameter)
        self.last_hard_parameters: int | None = None

    @property
    def costs(self) -> list[Tensor]:
        return [getattr(self, f"cost_{index}") for index in range(len(self.logits))]

    @staticmethod
    def _gumbel(shape: torch.Size, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        uniform = torch.rand(shape, device=device, dtype=dtype).clamp_(1e-6, 1 - 1e-6)
        return -torch.log(-torch.log(uniform))

    def _hard_project(self, values: Sequence[Tensor]) -> list[Tensor]:
        """Select at least one channel per layer without exceeding the budget."""

        flat_candidates: list[tuple[float, int, int, int]] = []
        selected: list[set[int]] = []
        used = 0
        for layer_index, (layer_values, layer_costs) in enumerate(zip(values, self.costs, strict=True)):
            detached = layer_values.detach().cpu()
            costs = layer_costs.detach().cpu()
            first = int(torch.argmax(detached))
            selected.append({first})
            used += int(costs[first])
            for channel_index in range(detached.numel()):
                if channel_index != first:
                    flat_candidates.append(
                        (float(detached[channel_index]), layer_index, channel_index, int(costs[channel_index]))
                    )
        flat_candidates.sort(key=lambda item: item[0], reverse=True)
        for _, layer_index, channel_index, cost in flat_candidates:
            if used + cost <= self.variable_budget:
                selected[layer_index].add(channel_index)
                used += cost

        masks: list[Tensor] = []
        for layer_values, indices in zip(values, selected, strict=True):
            hard = torch.zeros_like(layer_values)
            hard[list(indices)] = 1.0
            masks.append(hard)
        self.last_hard_parameters = self.fixed_parameters + used
        return masks

    def sample(self, *, stochastic: bool) -> list[Tensor]:
        probabilities: list[Tensor] = []
        for logits in self.logits:
            if stochastic:
                # Difference of two Gumbels is the Binary Concrete noise.
                noise = self._gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
                noise = noise - self._gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
            else:
                noise = torch.zeros_like(logits)
            probabilities.append(torch.sigmoid((logits + noise) / self.temperature))
        hard_masks = self._hard_project(probabilities)
        return [
            hard + probability - probability.detach()
            for hard, probability in zip(hard_masks, probabilities, strict=True)
        ]

    def deterministic_masks(self) -> list[Tensor]:
        return [mask.detach().cpu() for mask in self.sample(stochastic=False)]


def _convnormact_output_channel_cost(layer: nn.Module) -> int:
    conv = layer.conv
    if conv.groups != 1:
        raise ValueError("Expected a dense ConvNormAct")
    return (
        conv.in_channels * conv.kernel_size[0]
        + (1 if conv.bias is not None else 0)
        + 2  # GlobLN gamma and beta
    )


def parameter_channel_costs(model: nn.Module, model_name: str) -> list[Tensor]:
    """Return the exact number of physical parameters controlled per channel."""

    costs: list[Tensor] = []
    if model_name == "afrcnn12":
        block = model.sm.blocks
        downstream = block.last_layer[0].conv.out_channels * block.last_layer[0].conv.kernel_size[0]
        for layer in block.concat_layer:
            per_channel = _convnormact_output_channel_cost(layer) + downstream
            costs.append(torch.full((layer.conv.out_channels,), per_channel, dtype=torch.int64))
    elif model_name == "sudormrf":
        for block in model.sm:
            per_channel = _convnormact_output_channel_cost(block.proj_1x1)
            for stage in block.spp_dw:
                conv = stage.conv
                per_channel += conv.weight.numel() // conv.out_channels
                per_channel += 1 if conv.bias is not None else 0
                per_channel += 2
            per_channel += 2  # final GlobLN gamma and beta
            per_channel += block.res_conv.out_channels * block.res_conv.kernel_size[0]
            costs.append(torch.full((block.proj_1x1.conv.out_channels,), per_channel, dtype=torch.int64))
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return costs


def attach_budgeted_original_masks(
    model: nn.Module,
    model_name: str,
    *,
    target_parameters: int,
    temperature: float,
    seed: int,
) -> tuple[list[BudgetedMaskedChannelOutput], ParameterBudgetController]:
    costs = parameter_channel_costs(model, model_name)
    original_parameters = sum(parameter.numel() for parameter in model.parameters())
    wrappers: list[BudgetedMaskedChannelOutput] = []
    device = next(model.parameters()).device
    if model_name == "afrcnn12":
        for index, layer in enumerate(model.sm.blocks.concat_layer):
            wrapper = BudgetedMaskedChannelOutput(layer)
            model.sm.blocks.concat_layer[index] = wrapper
            wrappers.append(wrapper)
    elif model_name == "sudormrf":
        for index, block in enumerate(model.sm):
            wrapper = BudgetedMaskedChannelOutput(block.proj_1x1)
            block.proj_1x1 = wrapper
            wrappers.append(wrapper)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    controller = ParameterBudgetController(
        costs,
        original_parameters=original_parameters,
        target_parameters=target_parameters,
        temperature=temperature,
        seed=seed,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return wrappers, controller


def prepare_budgeted_masks(
    wrappers: Sequence[BudgetedMaskedChannelOutput],
    controller: ParameterBudgetController,
    *,
    stochastic: bool,
) -> list[Tensor]:
    masks = controller.sample(stochastic=stochastic)
    if len(masks) != len(wrappers):
        raise AssertionError("Controller/wrapper count mismatch")
    for wrapper, mask in zip(wrappers, masks, strict=True):
        wrapper.current_mask = mask
    return masks


def _freeze_except_masks(model: nn.Module, wrappers: Sequence[MaskedChannelOutput]) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for wrapper in wrappers:
        wrapper.mask.alpha.requires_grad_(True)


def attach_original_masks(
    model: nn.Module,
    model_name: str,
    *,
    epsilon: float,
    temperature: float,
    seed: int,
    initial_probability_low: float = 0.55,
    initial_probability_high: float = 0.85,
) -> list[MaskedChannelOutput]:
    """Attach exactly the mask locations exposed by the released scripts."""

    wrappers: list[MaskedChannelOutput] = []

    def wrap(layer: nn.Module, channels: int, mask_seed: int) -> MaskedChannelOutput:
        device = next(layer.parameters()).device
        wrapper = MaskedChannelOutput(
            layer,
            channels,
            epsilon=epsilon,
            temperature=temperature,
            seed=mask_seed,
            initial_probability_low=initial_probability_low,
            initial_probability_high=initial_probability_high,
        ).to(device)
        if wrapper.mask.alpha.device != device:
            raise RuntimeError(
                f"Mask device {wrapper.mask.alpha.device} does not match layer device {device}"
            )
        return wrapper

    if model_name == "afrcnn12":
        block = model.sm.blocks
        if len(block.concat_layer) != 4:
            raise ValueError(f"Expected four A-FRCNN fusion layers, got {len(block.concat_layer)}")
        for index, layer in enumerate(block.concat_layer):
            channels = int(layer.conv.out_channels)
            wrapper = wrap(layer, channels, seed + index)
            block.concat_layer[index] = wrapper
            wrappers.append(wrapper)
    elif model_name == "sudormrf":
        if len(model.sm) != 16:
            raise ValueError(f"Expected 16 SuDoRM-RF blocks, got {len(model.sm)}")
        for index, block in enumerate(model.sm):
            layer = block.proj_1x1
            channels = int(layer.conv.out_channels)
            wrapper = wrap(layer, channels, seed + index)
            block.proj_1x1 = wrapper
            wrappers.append(wrapper)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    _freeze_except_masks(model, wrappers)
    return wrappers


def mask_parameters(wrappers: Sequence[MaskedChannelOutput]) -> list[nn.Parameter]:
    return [wrapper.mask.alpha for wrapper in wrappers]


def deterministic_masks(wrappers: Sequence[MaskedChannelOutput]) -> list[Tensor]:
    return [wrapper.mask.deterministic_mask().detach().cpu() for wrapper in wrappers]


def _checked_keep(mask: Tensor, expected: int, label: str) -> Tensor:
    keep = mask.detach().to(device="cpu", dtype=torch.bool)
    if keep.ndim != 1 or keep.numel() != expected:
        raise ValueError(f"{label}: expected [{expected}] mask, got {tuple(keep.shape)}")
    if not torch.any(keep):
        raise ValueError(f"{label}: cannot prune every channel")
    return keep


def _copy_convnormact_output(old: nn.Module, keep: Tensor) -> nn.Module:
    """Rebuild ConvNormAct with selected output channels and copy all state."""

    conv = old.conv
    if conv.groups != 1:
        raise ValueError("Only dense ConvNormAct output pruning is supported")
    new = type(old)(
        conv.in_channels,
        int(keep.sum()),
        conv.kernel_size[0],
        conv.stride[0],
        conv.groups,
    )
    with torch.no_grad():
        new.conv.weight.copy_(conv.weight[keep])
        if conv.bias is not None:
            new.conv.bias.copy_(conv.bias[keep])
        new.norm.gamma.copy_(old.norm.gamma[keep])
        new.norm.beta.copy_(old.norm.beta[keep])
        new.act.load_state_dict(old.act.state_dict())
    return new


def _copy_convnormact_input(old: nn.Module, keep: Tensor) -> nn.Module:
    """Rebuild ConvNormAct with selected input channels and unchanged output."""

    conv = old.conv
    if conv.groups != 1:
        raise ValueError("Only dense ConvNormAct input pruning is supported")
    new = type(old)(
        int(keep.sum()),
        conv.out_channels,
        conv.kernel_size[0],
        conv.stride[0],
        conv.groups,
    )
    with torch.no_grad():
        new.conv.weight.copy_(conv.weight[:, keep, :])
        if conv.bias is not None:
            new.conv.bias.copy_(conv.bias)
        new.norm.load_state_dict(old.norm.state_dict())
        new.act.load_state_dict(old.act.state_dict())
    return new


def physically_prune_afrcnn(model: nn.Module, masks: Sequence[Tensor]) -> nn.Module:
    if len(masks) != 4:
        raise ValueError(f"A-FRCNN requires four masks, got {len(masks)}")
    pruned = copy.deepcopy(model).cpu()
    block = pruned.sm.blocks
    keeps: list[Tensor] = []
    for index in range(4):
        wrapped = block.concat_layer[index]
        old = wrapped.base if isinstance(
            wrapped, (MaskedChannelOutput, BudgetedMaskedChannelOutput)
        ) else wrapped
        keep = _checked_keep(masks[index], old.conv.out_channels, f"fusion[{index}]")
        block.concat_layer[index] = _copy_convnormact_output(old, keep)
        keeps.append(keep)

    old_last = block.last_layer[0]
    input_keep = torch.cat(keeps)
    if input_keep.numel() != old_last.conv.in_channels:
        raise AssertionError("A-FRCNN concatenated mask does not match last_layer input")
    block.last_layer[0] = _copy_convnormact_input(old_last, input_keep)
    return pruned


def _copy_sudormrf_block(old: nn.Module, keep: Tensor) -> nn.Module:
    wrapped = old.proj_1x1
    old_proj = wrapped.base if isinstance(
        wrapped, (MaskedChannelOutput, BudgetedMaskedChannelOutput)
    ) else wrapped
    keep = _checked_keep(keep, old_proj.conv.out_channels, "SuDoRM-RF hidden path")
    new = type(old)(
        out_channels=old_proj.conv.in_channels,
        in_channels=int(keep.sum()),
        upsampling_depth=old.depth,
    )
    new.proj_1x1 = _copy_convnormact_output(old_proj, keep)

    with torch.no_grad():
        for old_stage, new_stage in zip(old.spp_dw, new.spp_dw, strict=True):
            old_conv = old_stage.conv
            new_stage.conv.weight.copy_(old_conv.weight[keep])
            if old_conv.bias is not None:
                new_stage.conv.bias.copy_(old_conv.bias[keep])
            new_stage.norm.gamma.copy_(old_stage.norm.gamma[keep])
            new_stage.norm.beta.copy_(old_stage.norm.beta[keep])
        new.final_norm.norm.gamma.copy_(old.final_norm.norm.gamma[keep])
        new.final_norm.norm.beta.copy_(old.final_norm.norm.beta[keep])
        new.final_norm.act.load_state_dict(old.final_norm.act.state_dict())
        new.res_conv.weight.copy_(old.res_conv.weight[:, keep, :])
        if old.res_conv.bias is not None:
            new.res_conv.bias.copy_(old.res_conv.bias)
    return new


def physically_prune_sudormrf(model: nn.Module, masks: Sequence[Tensor]) -> nn.Module:
    if len(masks) != len(model.sm):
        raise ValueError(f"SuDoRM-RF requires {len(model.sm)} masks, got {len(masks)}")
    pruned = copy.deepcopy(model).cpu()
    for index, keep in enumerate(masks):
        pruned.sm[index] = _copy_sudormrf_block(pruned.sm[index], keep)
    return pruned


def physically_prune_original(
    model: nn.Module, model_name: str, masks: Sequence[Tensor]
) -> nn.Module:
    if model_name == "afrcnn12":
        return physically_prune_afrcnn(model, masks)
    if model_name == "sudormrf":
        return physically_prune_sudormrf(model, masks)
    if model_name == "tdanet":
        # TDANet prunes one FFN hidden dimension, so its mask list holds exactly
        # one tensor. The slicing itself lives in `tdanet_seprune` because it
        # predates this module and is already covered by the TDANet pipeline.
        from .tdanet_seprune import physically_prune_ffn

        if len(masks) != 1:
            raise ValueError(f"TDANet expects a single mask, got {len(masks)}")
        return physically_prune_ffn(model, masks[0])
    raise ValueError(f"Unsupported model: {model_name}")
