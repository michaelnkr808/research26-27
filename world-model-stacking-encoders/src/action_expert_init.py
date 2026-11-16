"""Action expert initialised from the Wan 2.2 video expert by Fast-WAM's recipe (stage 1 baseline PLAN, Open item 2,
decided by the user 2026-11-15: "derive from wan 2.2 weights").

A port of Fast-WAM 7faa711 `scripts/preprocess_action_dit_backbone.py` (MIT), applied in process to a built model whose
video expert already holds Wan blocks 0-7 (pretrained_sizing.load_wan), in fp32 (the script's default dtype, our recipe;
Fast-WAM's README runs it in bf16, which this port does not do). For every action-expert tensor except the ones that
start with `action_encoder.` or `head.` (ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES):
  - same shape as the video tensor of the same name: copied;
  - otherwise resized to the action shape one dimension at a time in index order (dim 0 first, then dim 1, ...), each
    by 1-D linear interpolation with align_corners=True in fp32 (`_interpolate_last_dim`, `_resize_tensor_to_shape`,
    script lines 53-96, copied verbatim below);
  - then, if the tensor has 2 or more dimensions and its last dimension changed, multiplied by
    alpha = sqrt(source last dim / target last dim) (script lines 206-208). As the script does, this also scales the
    3-D block `modulation` (added, not multiplied, in the block) and `time_projection.1` is interpolated over its stacked
    six-chunk axis, so values near chunk boundaries mix neighbouring chunks (fastwam-reading.md Q5).
`action_encoder.*` and `head.*` keep their fresh init from the model build, as in Fast-WAM; so does the proprio encoder
(not part of the script). At the Wan 2.2 TI2V-5B sizes with 8 layers: 226 derived tensors (80 copied, 146 resized) and
4 fresh in the action expert, plus 2 fresh in the proprio encoder.
"""

import torch
import torch.nn.functional as F

SKIP_PREFIXES = ("action_encoder.", "head.")   # ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES (action_dit.py:32)


# --- verbatim from Fast-WAM 7faa711 scripts/preprocess_action_dit_backbone.py lines 53-96 (type hints kept) ----------

def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        # Permute the target dimension to the end for interpolation
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        # Construct inverse permutation to restore original order
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        # Permute, interpolate, and restore original order
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape for tensor. src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)

# ---------------------------------------------------------------------------------------------------------------------


def classify(video_shapes, action_shapes):
    """{action key: 'copied' | 'resized' | 'fresh'} from shapes only (works for meta tensors). Raises KeyError if a
    backbone key has no video tensor of the same name (the script raises there too, line 196-198)."""
    out = {}
    for k, shp in action_shapes.items():
        if k.startswith(SKIP_PREFIXES):
            out[k] = "fresh"
            continue
        if k not in video_shapes:
            raise KeyError(f"action-expert key {k} has no video-expert tensor of the same name")
        out[k] = "copied" if tuple(video_shapes[k]) == tuple(shp) else "resized"
    return out


def alpha_of(src_shape, target_shape):
    """The script's alpha (lines 206-208): sqrt(src last / target last) if ndim >= 2 and the last dim changed, else None."""
    if len(src_shape) >= 2 and src_shape[-1] != target_shape[-1]:
        return (float(src_shape[-1]) / float(target_shape[-1])) ** 0.5
    return None


def derive_backbone(video_state, action_state, apply_alpha_scaling=True):
    """The script's loop (lines 192-210) on two state dicts: {key: derived tensor} for every backbone key of the action
    expert, in the action tensor's dtype, on CPU (as the script saves it), and {key: kind}."""
    kinds = classify({k: v.shape for k, v in video_state.items()}, {k: v.shape for k, v in action_state.items()})
    out = {}
    for key in sorted(k for k, kind in kinds.items() if kind != "fresh"):
        src, target = video_state[key], action_state[key]
        if tuple(src.shape) == tuple(target.shape):
            value = src
        else:
            value = _resize_tensor_to_shape(src, tuple(target.shape))
            if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target.shape[-1]:
                alpha = (float(src.shape[-1]) / float(target.shape[-1])) ** 0.5
                value = value.to(torch.float32) * alpha
        out[key] = value.detach().to(dtype=target.dtype, device="cpu").contiguous()
    return out, kinds


@torch.no_grad()
def init_action_expert(model, log=print):
    """Overwrite model.action_expert's backbone with tensors derived from model.video_expert (call after load_wan).
    The fresh tensors (action_encoder, head) and the video expert are left untouched. Returns a report:
    counts, and per tensor {kind, shape, source_shape, alpha}."""
    video_state = model.video_expert.state_dict()
    action = model.action_expert
    if dict(action.named_buffers()):
        raise RuntimeError(f"action expert has buffers the derivation does not cover: {list(dict(action.named_buffers()))}")
    action_state = action.state_dict()
    derived, kinds = derive_backbone(video_state, action_state)
    params = dict(action.named_parameters())
    for k, v in derived.items():
        params[k].copy_(v.to(params[k].device))
    per = {}
    for k, kind in kinds.items():
        e = {"kind": kind, "shape": list(action_state[k].shape)}
        if kind != "fresh":
            e["source"] = "video_expert." + k
            e["source_shape"] = list(video_state[k].shape)
            a = alpha_of(tuple(video_state[k].shape), tuple(action_state[k].shape)) if kind == "resized" else None
            e["alpha"] = a
        per["action_expert." + k] = e
    counts = {c: sum(1 for x in kinds.values() if x == c) for c in ("copied", "resized", "fresh")}
    counts["derived"] = counts["copied"] + counts["resized"]
    log(f"[action expert] derived from the video expert by Fast-WAM's recipe: {counts}")
    return {"recipe": "Fast-WAM 7faa711 scripts/preprocess_action_dit_backbone.py, ported; fp32; alpha scaling on; "
                      "sequential 1-D linear interpolation, align_corners=True, dim 0 first",
            "counts": counts, "parameters": per}


def alpha_values(kinds, video_shapes, action_shapes):
    """{key: alpha or None} for the resized tensors (for reports and tests)."""
    return {k: alpha_of(tuple(video_shapes[k]), tuple(action_shapes[k])) for k, kind in kinds.items()
            if kind == "resized"}

