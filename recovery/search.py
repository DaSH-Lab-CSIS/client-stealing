"""Candidate generation: source-direction extraction, head sketches, cached beam search, and selection."""
from __future__ import annotations

import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from recovery.model import (
    _batch,
    embeddings_are_tied,
    is_neox_parallel_model,
    parallel_base_model,
    parallel_layers,
    parallel_source_kind,
    parallel_token_activation,
)


@torch.no_grad()
def extract_source_directions(model, observation, generic_bag_topk=4096,
                              source_ridge_relative=1e-8,
                              tied_direction_norm_factor=10.0):
    """Exact token-conditioned first-block residual gradients, when identifiable.

    On parallel-residual GPT-NeoX, the first MLP's activation depends only on the
    input token. G_mlp0 = D.T @ A, where A has width 3072 (not 768).
    """
    gin = observation["input"]
    norm = gin.norm(dim=1)
    # After offline subtraction, retain the original activation dictionary in
    # the solve so numerical residuals on removed tokens do not contaminate the
    # remaining token directions. Only live input tokens are searched.
    source_support = observation.get("source_support")
    initial_tied_superset = False
    if source_support is None:
        if embeddings_are_tied(model) and parallel_source_kind(model) is not None:
            # A tied head makes every embedding row dense through the softmax
            # gradient.  Use a conservative row-norm superset and let the
            # overdetermined MLP equation assign near-zero cotangents to rows
            # that are not input tokens.  This is blind; no reference tokens or
            # counts enter the selection.
            bag = norm.topk(min(generic_bag_topk, len(norm))).indices
            source_support = bag
            initial_tied_superset = True
        else:
            bag = (norm > norm.max() * 1e-7).nonzero().flatten()
            source_support = bag
    else:
        live = norm[source_support] > norm[source_support].max() * 1e-7
        bag = source_support[live]
    kind = parallel_source_kind(model)
    if kind is None or "mlp0" not in observation:
        trimmed = False
        original_support = int(len(bag))
        # Tied input/output embeddings make the observed embedding gradient dense
        # because it includes the softmax head gradient.  Keep the strongest rows
        # as likely source/target tokens; otherwise the generic search starts from
        # nearly every vocabulary item in token-id order.
        if len(bag) > 4096:
            topk = min(generic_bag_topk, len(norm))
            bag = norm.topk(topk).indices
            trimmed = True
        return bag, None, {
            "input_support": len(bag),
            "source_fingerprint": False,
            "source_fingerprint_reason": "generic input/output fallback",
            "generic_support_trimmed": trimmed,
            "generic_original_support": original_support,
        }
    embedding = model.get_input_embeddings().weight[source_support]
    activation = parallel_token_activation(model, embedding)
    if activation.numel() > 20_000_000:
        # Equivalent full-column-rank least squares via normal equations.  This
        # avoids an enormous QR workspace for Falcon's 18,176-wide MLP while
        # retaining the blind overdetermined solve.  The ridge is scale-relative
        # and its value is recorded through the resulting fit residual.
        a = activation.float()
        g = observation["mlp0"].float()
        gram = a @ a.T
        ridge = gram.diag().median() * source_ridge_relative
        directions = torch.linalg.solve(
            gram + ridge * torch.eye(len(a), device=a.device, dtype=a.dtype),
            a @ g.T,
        )
    else:
        directions = torch.linalg.lstsq(
            activation.T.double(), observation["mlp0"].T.double(), rcond=1e-8,
        ).solution.float()
    fit = directions.T.float() @ activation.float()
    observed_mlp = observation["mlp0"].float()
    error = float((fit - observed_mlp).norm() / observed_mlp.norm())
    lookup = {int(t): i for i, t in enumerate(source_support)}
    directions = directions[torch.tensor(
        [lookup[int(t)] for t in bag], device=directions.device
    )]
    unpruned_support = len(bag)
    direction_threshold = None
    if initial_tied_superset and tied_direction_norm_factor > 0:
        direction_norm = directions.float().norm(dim=1)
        direction_threshold = float(direction_norm.median() * tied_direction_norm_factor)
        keep = direction_norm >= direction_threshold
        bag = bag[keep]
        directions = directions[keep]
    return bag, directions, {
        "mlp_fit_relative_error": error,
        "input_support": len(bag),
        "source_fingerprint": True,
        "parallel_source_kind": kind,
        "dictionary_shape": list(activation.shape),
        "source_ridge_relative": source_ridge_relative if activation.numel() > 20_000_000 else None,
        "tied_unpruned_support": unpruned_support if initial_tied_superset else None,
        "tied_direction_norm_factor": tied_direction_norm_factor if initial_tied_superset else None,
        "tied_direction_norm_threshold": direction_threshold,
        "tied_pruned_support": len(bag) if initial_tied_superset else None,
    }


@torch.no_grad()
def last_states(model, rows, pad, direction=None, epsilon=0.01):
    device = next(model.parameters()).device
    ids, mask = _batch(rows, pad, device)
    end = mask.sum(1) - 1
    bi = torch.arange(len(rows), device=device)
    handle = None
    if direction is not None:
        def shift(module, inputs, output):
            hidden = output[0].clone()
            hidden[bi, end] += epsilon * direction
            return (hidden,) + output[1:]
        handle = model.gpt_neox.layers[0].register_forward_hook(shift)
    try:
        hidden = model.gpt_neox(ids, attention_mask=mask, use_cache=False).last_hidden_state
        return hidden[bi, end]
    finally:
        if handle is not None:
            handle.remove()


@torch.no_grad()
def cloud_transform(gradient, pc=16):
    # CUDA SVD/eigh do not implement bfloat16.  The score is a numerical
    # diagnostic/recovery transform, so evaluate it in float32 even when the
    # public checkpoint and captured gradients use reduced precision.
    gradient = gradient.float()
    weight = 1 / (gradient.std(0) + gradient.std(0).median() * 1e-3)
    cloud = -gradient * weight
    mean = cloud.mean(0)
    basis = torch.linalg.svd(cloud - mean, full_matrices=False).Vh[:pc]
    def transform(x):
        x = x.float() * weight - mean
        return F.normalize(x - (x @ basis.T) @ basis, dim=-1)
    return transform, transform(-gradient)


@torch.no_grad()
def covariance_cloud_transforms(gradient, pc=16):
    """Diagnostic: regularized full covariance after the existing PC removal."""
    gradient = gradient.float()
    weight = 1 / (gradient.std(0) + gradient.std(0).median() * 1e-3)
    cloud = -gradient * weight
    mean = cloud.mean(0)
    basis = torch.linalg.svd(cloud - mean, full_matrices=False).Vh[:pc]
    def linear(x):
        value = x.float() * weight - mean
        return value - (value @ basis.T) @ basis
    raw = linear(-gradient)
    result = {}
    for normalized in (False, True):
        sample = F.normalize(raw, dim=1) if normalized else raw
        ev, vectors = torch.linalg.eigh(sample.double().T @ sample.double() / len(sample))
        for floor in (.01, .001):
            white = ((vectors / ev.clamp_min(ev.max() * floor).sqrt()) @ vectors.T).float()
            def transform(x, white=white):
                return F.normalize(linear(x) @ white, dim=1)
            result[f"full_cov_norm{normalized}_{floor}"] = (transform, transform(-gradient))
    return result


class _StopFeatures(Exception):
    pass


@torch.no_grad()
def features(model, rows, pad, chunk=64):
    result = []
    for start in range(0, len(rows), chunk):
        part = rows[start:start + chunk]
        ids, mask = _batch(part, pad, next(model.parameters()).device)
        def capture(module, inputs):
            result.append(inputs[0][
                torch.arange(len(part), device=inputs[0].device), mask.sum(1) - 1
            ].clone())
            raise _StopFeatures
        handle = model.gpt_neox.layers[1].mlp.dense_4h_to_h.register_forward_pre_hook(capture)
        try:
            model.gpt_neox(ids, attention_mask=mask, use_cache=False)
        except _StopFeatures:
            pass
        finally:
            handle.remove()
    return torch.cat(result)


class HeadSketch:
    """Fixed, reproducible linear map of a V-by-d head gradient."""
    def __init__(self, gradient, bag, buckets=128, width=16, seed=9341):
        generator = torch.Generator(device=gradient.device).manual_seed(seed)
        self.buckets = buckets
        self.bucket = torch.randint(buckets, (len(gradient),), generator=generator,
                                    device=gradient.device)
        sign = torch.randint(2, (len(gradient),), generator=generator,
                             device=gradient.device).float() * 2 - 1
        norm = gradient.norm(dim=1)
        self.scale = sign / norm.clamp_min(norm[bag].median() * .25)
        dimension_scale = 1 / gradient[bag].std(0).clamp_min(gradient[bag].std(0).median() * .01)
        self.project = torch.randn(gradient.shape[1], width, generator=generator,
                                   device=gradient.device) / width ** .5
        self.project *= dimension_scale[:, None]

    def matrix(self, gradient):
        gradient = gradient.float()
        result = gradient.new_zeros(self.buckets, self.project.shape[1])
        result.index_add_(0, self.bucket, (gradient @ self.project) * self.scale[:, None])
        return result.flatten()

    def prediction(self, hidden, probability):
        hidden = hidden.float()
        probability = probability.float()
        moment = hidden.new_zeros(len(hidden), self.buckets)
        moment.scatter_add_(1, self.bucket[None].expand(len(hidden), -1), probability * self.scale)
        projected = hidden @ self.project
        return moment[:, :, None] * projected[:, None, :], projected

    def append(self, accumulated, predicted, projected, parent, label):
        value = accumulated[parent] + predicted[parent]
        value[
            torch.arange(len(label), device=value.device), self.bucket[label]
        ] -= self.scale[label, None] * projected[parent]
        return value


class ContextMembership:
    """Soft contextual-MLP membership above the exact span method's rank limit."""
    @torch.no_grad()
    def __init__(self, model, tok, observation, bag, background_count=4096):
        # features() is embedded in this standalone file.
        generator = torch.Generator().manual_seed(811)
        background = [bag[torch.randint(len(bag), (length,), generator=generator)].tolist()
                      for length in torch.randint(1, 12, (background_count,), generator=generator).tolist()]
        activation = features(model, background, tok.pad_token_id).double()
        covariance = activation.T @ activation / len(activation)
        ev, vectors = torch.linalg.eigh(covariance)
        white = (vectors / ev.clamp_min(ev.max() * 1e-5).sqrt()) @ vectors.T
        basis = torch.linalg.svd(observation["mlp1"].double() @ white, full_matrices=False).Vh.T
        self.white = white.float()
        self.signal = (white @ basis).float()

    @torch.no_grad()
    def next(self, model, updated, parent, label):
        from transformers.cache_utils import DynamicCache
        class Stop(Exception):
            pass
        captured = []
        block = model.gpt_neox.layers[1]
        def capture(module, inputs):
            hidden = inputs[0]
            captured.append(block.mlp.act(block.mlp.dense_h_to_4h(block.post_attention_layernorm(hidden)))[:, -1])
            raise Stop
        handle = block.register_forward_pre_hook(capture)
        # Only the first block executes, so no other cached layer needs copying.
        cache = DynamicCache.from_legacy_cache(
            (tuple(t.index_select(0, parent) for t in updated[0]),))
        try:
            model.gpt_neox(label[:, None], past_key_values=cache, use_cache=True)
        except Stop:
            pass
        finally:
            handle.remove()
        activation = captured[0]
        return (activation @ self.signal).square().sum(1) / (activation @ self.white).square().sum(1).clamp_min(1e-20)


@torch.no_grad()
def cached_step(model, tokens, past=None, direction=None, epsilon=.01):
    """Fork the cache so finite-difference probes cannot mutate the true prefix."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache.from_legacy_cache(past) if past is not None else None
    handle = None
    if direction is not None:
        def shift(module, inputs, output):
            return (output[0] + epsilon * direction[:, None],) + output[1:]
        handle = model.gpt_neox.layers[0].register_forward_hook(shift)
    try:
        output = model.gpt_neox(tokens[:, None], past_key_values=cache, use_cache=True)
        return output.last_hidden_state[:, -1], output.past_key_values.to_legacy_cache()
    finally:
        if handle is not None:
            handle.remove()


@torch.no_grad()
def full_prefix_step(model, rows, pad):
    """Architecture-generic next-token state by recomputing the whole prefix."""
    device = next(model.parameters()).device
    rows = rows.to(device)
    mask = torch.ones_like(rows, dtype=torch.long, device=device)
    base_name = getattr(model, "base_model_prefix", None)
    base = getattr(model, base_name, None) if base_name else None
    if base is not None and base is not model:
        output = base(rows, attention_mask=mask, use_cache=False)
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        return hidden[:, -1], None
    output = model(rows, attention_mask=mask, use_cache=False, output_hidden_states=True)
    return output.hidden_states[-1][:, -1], None


@torch.no_grad()
def generic_cached_step(model, tokens, past=None):
    """Architecture-generic causal-LM next-token state with KV caching."""
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    base_name = getattr(model, "base_model_prefix", None)
    base = getattr(model, base_name, None) if base_name else None
    if base is not None and base is not model:
        output = base(tokens[:, None], past_key_values=past, use_cache=True)
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        next_past = output.past_key_values if hasattr(output, "past_key_values") else output[1]
        return hidden[:, -1], next_past
    output = model(tokens[:, None], past_key_values=past, use_cache=True, output_hidden_states=True)
    return output.hidden_states[-1][:, -1], output.past_key_values


def fork_cached_state(past):
    """Make an independent cache for finite-difference source probes."""
    if past is None:
        return None
    if hasattr(past, "to_legacy_cache"):
        from transformers.cache_utils import DynamicCache
        return DynamicCache.from_legacy_cache(past.to_legacy_cache())
    # Legacy tuple caches are immutable containers; model implementations
    # concatenate rather than mutate their tensors.
    return tuple(tuple(t for t in layer) for layer in past)


@torch.no_grad()
def parallel_cached_step(model, tokens, past=None, direction=None, epsilon=.01):
    """Architecture-generic cached step with a first-block residual intervention."""
    base = parallel_base_model(model)
    handle = None
    if direction is not None:
        def shift(module, inputs, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                hidden[:, -1] += epsilon * direction.to(hidden.dtype)
                return (hidden,) + output[1:]
            hidden = output.clone()
            hidden[:, -1] += epsilon * direction.to(hidden.dtype)
            return hidden
        handle = parallel_layers(model)[0].register_forward_hook(shift)
    try:
        output = base(
            tokens[:, None], past_key_values=fork_cached_state(past), use_cache=True,
        )
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        updated = output.past_key_values if hasattr(output, "past_key_values") else output[1]
        return hidden[:, -1], updated
    finally:
        if handle is not None:
            handle.remove()


def select_cached_state(past, parent):
    """Select beam parents from tuple or Transformers Cache past-key-values."""
    if past is None:
        return None
    if hasattr(past, "batch_select_indices"):
        past.batch_select_indices(parent.to(past.layers[0].keys.device))
        return past
    return tuple(tuple(t.index_select(0, parent.to(t.device)) for t in layer) for layer in past)


def sentence_start_like_tokens(tok, vocab: int, norm: torch.Tensor) -> list[int]:
    """Reference-independent start seeds for tokenizers with leading-space words.

    GPT-2/RoBPE-style tokenizers encode sentence-initial words without a leading
    space, while most mid-sentence words begin with a space.  Dense tied
    embedding gradients rank punctuation and leading-space words first, so a
    generic model needs a separate start-seed ordering.
    """
    start_like = []
    for token in range(vocab):
        text = tok.decode([token])
        if text and not text[0].isspace() and not text.startswith(" "):
            start_like.append(token)
    if not start_like:
        return []
    norm_cpu = norm.detach().cpu()
    return sorted(start_like, key=lambda token: float(norm_cpu[token]), reverse=True)


@torch.no_grad()
def search(model, tok, observation, *, beam=8, seed_batch=16, max_length=24,
           source_weight=.05, lm_weight=.02, seed_limit=0, mlp_weight=0., lookahead=32,
           keep_atoms=True, full_covariance=False, checkpoint=None, checkpoint_identity=None,
           generic_bag_topk=4096, target_extra_topk=256,
           prefix_fanout_depth=0, prefix_fanout_branch=0):
    device = next(model.parameters()).device
    bag, directions, info = extract_source_directions(model, observation, generic_bag_topk=generic_bag_topk)
    supports_source = directions is not None and parallel_source_kind(model) is not None
    # The contextual layer-1 subspace code is still NeoX-specific.  It was near
    # neutral in the Pythia ablation, so modern adapters retain the strong source
    # score while explicitly disabling this optional term.
    supports_mlp_membership = supports_source and is_neox_parallel_model(model) and "mlp1" in observation
    if source_weight and not supports_source:
        print("Source-fingerprint scoring unavailable for this architecture; using output-gradient + LM search.", flush=True)
        source_weight = 0.
    if mlp_weight and not supports_mlp_membership:
        print("MLP membership scoring unavailable for this architecture; disabling --mlp-weight.", flush=True)
        mlp_weight = 0.
    vocab = model.get_output_embeddings().weight.shape[0]
    # Some tokens occur only as a final target and have no input gradient.
    norm = observation["output"].norm(dim=1)
    outside = norm.clone()
    outside[bag] = -1
    extra = outside.topk(min(target_extra_topk, vocab - len(bag))).indices
    targets = torch.cat((bag, extra))
    target_is_input = torch.arange(len(targets), device=targets.device) < len(bag)
    transform, cloud = cloud_transform(observation["output"][targets])
    if full_covariance:
        transform, _ = covariance_cloud_transforms(observation["output"][bag])[
            "full_cov_normTrue_0.001"]
        cloud = transform(-observation["output"][targets])
    source_index = None
    if supports_source:
        source_index = torch.full((vocab,), -1, dtype=torch.long, device=bag.device)
        source_index[bag] = torch.arange(len(bag), device=bag.device)
        covariance = directions.double().T @ directions.double() / len(directions)
        ev, vectors = torch.linalg.eigh(covariance)
        inverse = (vectors / ev.clamp_min(ev.max() * 1e-3)) @ vectors.T
        directions = F.normalize((directions.double() @ inverse).float(), dim=1)
    head = model.get_output_embeddings().weight
    sketch = HeadSketch(observation["output"], bag)
    observed = sketch.matrix(observation["output"])
    normalizer = observed.norm()
    observed /= normalizer
    normalizer_cpu = normalizer.detach().cpu()
    membership = ContextMembership(model, tok, observation, bag) if mlp_weight else None
    if supports_source:
        seeds = bag.tolist()
        seed_mode = "input_support"
    else:
        generic_seeds = sentence_start_like_tokens(tok, vocab, norm)
        seeds = generic_seeds or bag.tolist()
        seed_mode = "generic_start_like_norm" if generic_seeds else "generic_support_norm"
    if seed_limit:
        # Truncate the selected reference-independent seed order for bounded runs.
        seeds = seeds[:seed_limit]
    checkpoint_path = Path(checkpoint) if checkpoint is not None else None
    checkpoint_every = max(seed_batch, int(os.environ.get("RECOVERY_SEARCH_CHECKPOINT_STARTS", "512")))
    search_info = {**info, "beam": beam, "starts": len(seeds),
        "max_length": max_length, "source_weight": source_weight, "lm_weight": lm_weight,
        "mlp_weight": mlp_weight, "lookahead": lookahead if mlp_weight else 0,
        "full_covariance": full_covariance,
        "generic_bag_topk": generic_bag_topk, "target_extra_topk": target_extra_topk,
        "prefix_fanout_depth": prefix_fanout_depth, "prefix_fanout_branch": prefix_fanout_branch,
        "seed_mode": seed_mode,
        "aggregate_observations": 1, "teacher_forced": False, "inserted_bos": False}
    checkpoint_config = {
        "beam": beam, "seed_batch": seed_batch, "max_length": max_length,
        "source_weight": source_weight, "lm_weight": lm_weight,
        "seed_limit": seed_limit, "mlp_weight": mlp_weight,
        "lookahead": lookahead, "keep_atoms": keep_atoms,
        "generic_bag_topk": generic_bag_topk, "target_extra_topk": target_extra_topk,
        "prefix_fanout_depth": prefix_fanout_depth, "prefix_fanout_branch": prefix_fanout_branch,
        "full_covariance": full_covariance, "seeds": seeds,
        "checkpoint_identity": checkpoint_identity,
        "search_info": search_info,
    }
    terminal = torch.tensor(
        [tok.decode([t]).rstrip().endswith((".", "?", "!")) for t in targets.tolist()],
        device=targets.device,
    )
    candidates, atoms, qualities = [], [], []
    completed_starts = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, weights_only=True)
        if saved.get("config") != checkpoint_config:
            raise ValueError(f"Search checkpoint {checkpoint_path} used a different configuration")
        candidates = saved["rows"]
        qualities = saved["qualities"]
        completed_starts = saved["completed_starts"]
        if keep_atoms:
            atoms = saved["atoms"]
        print(
            f"Resuming search checkpoint {checkpoint_path}: "
            f"{completed_starts}/{len(seeds)} starts; candidates={len(candidates)}",
            flush=True,
        )
    last_checkpoint = completed_starts
    started = time.monotonic()
    for start in range(completed_starts, len(seeds), seed_batch):
        part = seeds[start:start + seed_batch]
        seed_groups = len(part)
        groups = seed_groups
        rows = torch.tensor(part, device=device)[:, None]
        quality = torch.zeros(groups, device=device)
        accumulated = torch.zeros(groups, sketch.buckets, sketch.project.shape[1], device=device)
        past = None
        current_width = 1
        for length in range(1, max_length):
            if parallel_source_kind(model) is not None:
                if is_neox_parallel_model(model):
                    hidden, updated = cached_step(model, rows[:, -1], past)
                else:
                    hidden, updated = parallel_cached_step(model, rows[:, -1], past)
            else:
                hidden, updated = generic_cached_step(model, rows[:, -1], past)
            logits = hidden @ head.T
            logprob = logits.log_softmax(-1)
            scores = transform(hidden) @ cloud.T + lm_weight * logprob[:, targets]
            if source_weight and supports_source:
                direction = directions[source_index[rows[:, -1]]]
                if is_neox_parallel_model(model):
                    plus, _ = cached_step(model, rows[:, -1], past, direction, .01)
                    minus, _ = cached_step(model, rows[:, -1], past, direction, -.01)
                else:
                    plus, _ = parallel_cached_step(model, rows[:, -1], past, direction, .01)
                    minus, _ = parallel_cached_step(model, rows[:, -1], past, direction, -.01)
                source = -((plus - minus) / .02) @ head[targets].T
                source = (source - source.mean(-1, keepdim=True)) / source.std(-1, keepdim=True).clamp_min(1e-8)
                scores += source_weight * source
            # No blanket repetition ban: real training text can repeat tokens.
            # Penalize only a generated three-token loop, while allowing an isolated repeat.
            if length >= 6:
                loops = (rows[:, -3:] == rows[:, -6:-3]).all(1)
                scores[loops, :] -= 1
            if keep_atoms:
                predicted, projected = sketch.prediction(hidden, logprob.exp())
            combined = quality[:, None] + scores
            # Keep completed hypotheses independently, so punctuation need not consume an active beam.
            if length >= 3:
                end_scores = combined.clone()
                if length < max_length - 1:
                    end_scores[:, ~(terminal | ~target_is_input)] = -torch.inf
                end_scores = end_scores.reshape(groups, -1)
                values, indices = end_scores.topk(min(beam, end_scores.shape[1]), dim=1)
                parent = (
                    indices // len(targets)
                    + torch.arange(groups, device=indices.device)[:, None] * current_width
                ).flatten()
                label = targets[(indices % len(targets)).flatten()]
                finite = values.flatten().isfinite()
                parent, label = parent[finite], label[finite]
                endings = torch.cat((rows[parent], label[:, None]), dim=1)
                candidates.extend(endings.tolist())
                if keep_atoms:
                    sketches = sketch.append(accumulated, predicted, projected, parent, label)
                    atoms.append(sketches.flatten(1).cpu() / normalizer_cpu)
                qualities.extend((values.flatten()[finite] / length).tolist())
            if length == max_length - 1:
                break
            combined[:, ~target_is_input] = -torch.inf
            in_fanout = prefix_fanout_depth and length <= prefix_fanout_depth
            if in_fanout:
                proposal_width = max(1, prefix_fanout_branch or beam)
            else:
                proposal_width = max(beam, lookahead) if membership else beam
            values, indices = combined.reshape(groups, -1).topk(proposal_width, dim=1)
            parent = (
                indices // len(targets)
                + torch.arange(groups, device=indices.device)[:, None] * current_width
            ).flatten()
            label = targets[(indices % len(targets)).flatten()]
            if membership and not in_fanout:
                evidence = membership.next(model, updated, parent, label).reshape(groups, proposal_width)
                values, choice = (values + mlp_weight * evidence).topk(beam, dim=1)
                choice = (choice + torch.arange(groups, device=choice.device)[:, None] * proposal_width).flatten()
                parent, label = parent[choice], label[choice]
            if keep_atoms:
                accumulated = sketch.append(accumulated, predicted, projected, parent, label)
            rows = torch.cat((rows[parent], label[:, None]), dim=1)
            quality = values.flatten()
            if is_neox_parallel_model(model):
                past = tuple(tuple(t.index_select(0, parent) for t in layer) for layer in updated)
            else:
                past = select_cached_state(updated, parent)
            if in_fanout:
                # Treat every selected prefix as its own independent sub-search.
                groups = len(parent)
                current_width = 1
            else:
                current_width = beam
        print(f"Searched {start + seed_groups}/{len(seeds)} starts; candidates={len(candidates)}; "
              f"seconds={time.monotonic() - started:.1f}", flush=True)
        if checkpoint_path is not None and (
            start + seed_groups == len(seeds) or start + seed_groups - last_checkpoint >= checkpoint_every
        ):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(".tmp")
            payload = {
                "config": checkpoint_config,
                "completed_starts": start + seed_groups,
                "rows": candidates,
                "qualities": qualities,
            }
            if keep_atoms:
                payload["atoms"] = atoms
            torch.save(payload, temporary)
            temporary.replace(checkpoint_path)
            last_checkpoint = start + seed_groups
    return {"rows": candidates, "atoms": torch.cat(atoms) if keep_atoms else torch.empty(0, 0), "target": observed.cpu(),
            "qualities": qualities, "info": search_info}


def select(pool, count=128):
    """Nonnegative OMP on sequence gradients; no reference text or true lengths."""
    from scipy.optimize import nnls
    atoms, target = pool["atoms"], pool["target"]
    norms = atoms.norm(dim=1).clamp_min(1e-12)
    dictionary = atoms / norms[:, None]
    residual = target.clone()
    selected, coefficients = [], []
    available = torch.ones(len(atoms), dtype=torch.bool)
    for step in range(count):
        correlation = dictionary @ residual
        correlation[~available] = -torch.inf
        index = int(correlation.argmax())
        if correlation[index] <= 0:
            break
        selected.append(index)
        available[index] = False
        design = dictionary[selected].T.double().numpy()
        coefficient, _ = nnls(design, target.double().numpy(), maxiter=10 * len(selected))
        coefficients = torch.from_numpy(coefficient).float()
        residual = target - dictionary[selected].T @ coefficients
        if step % 16 == 15:
            print(f"Joint selection {step + 1}/{count}; sketch residual={float(residual.norm()):.4f}", flush=True)
    keep = [(i, float(c / norms[i])) for i, c in zip(selected, coefficients) if c > 1e-6]
    return {"rows": [pool["rows"][i] for i, _ in keep], "indices": [i for i, _ in keep],
            "coefficients": [c for _, c in keep], "sketch_relative_error": float(residual.norm()),
            "candidate_count": len(pool["rows"]), "search": pool["info"]}


def ranked_unique_candidates(pools, budget, exclude=()):
    """Dedupe candidates while preserving each pool's internal quality ranking."""
    orders = [torch.tensor(pool["qualities"]).argsort(descending=True).tolist() for pool in pools]
    seen, unique = set(map(tuple, exclude)), []
    depth = 0
    while len(unique) < budget and any(depth < len(order) for order in orders):
        for pool, order in zip(pools, orders):
            if depth >= len(order):
                continue
            row = tuple(pool["rows"][order[depth]])
            if row in seen:
                continue
            seen.add(row)
            unique.append(list(row))
            if len(unique) >= budget:
                break
        depth += 1
    return unique
