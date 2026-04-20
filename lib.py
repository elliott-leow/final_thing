"""
Activation extraction, contrastive directions, and analysis helpers
for probing clinical sycophancy in language models.
"""

import gc
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score

SEED = 42


def set_seeds(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(model):
    return next(model.parameters()).device


def check_model_arch(model):
    """Verify model has the expected OLMo/Llama layer structure."""
    assert hasattr(model, 'model') and hasattr(model.model, 'layers'), \
        f"Expected model.model.layers (OLMo/Llama arch), got {type(model)}"


def format_prompt(tokenizer, user_text, system_prompt=None):
    """Wrap user text in the model's chat template if available.

    This is important: instruct models expect chat-formatted input.
    Feeding raw text to a chat model means the activations reflect
    out-of-distribution processing, not the model's trained behavior.
    """
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_text})
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
    return user_text


def cohens_d(group_a, group_b):
    """Effect size between two groups."""
    na, nb = len(group_a), len(group_b)
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    return (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram():
    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0


def clear_hf_cache(model_id):
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache/huggingface/hub",
        "models--" + model_id.replace("/", "--"),
    )
    if os.path.exists(cache_dir):
        try:
            sz = sum(os.path.getsize(os.path.join(dp, f))
                     for dp, dn, fn in os.walk(cache_dir) for f in fn)
        except (FileNotFoundError, OSError):
            sz = 0
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"  Cleared {sz / 1e9:.1f} GB cache for {model_id}")


def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=SEED):
    rng = np.random.RandomState(seed)
    values = np.array(values)
    boot_means = [rng.choice(values, len(values), replace=True).mean()
                  for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return {
        "mean": float(values.mean()),
        "ci_lo": float(np.percentile(boot_means, alpha * 100)),
        "ci_hi": float(np.percentile(boot_means, (1 - alpha) * 100)),
    }


# --- error bar helpers ---

def bootstrap_cosine_ci_by_layer(pos_a, neg_a, pos_b, neg_b, layers,
                                  n_boot=200, ci=0.95, seed=SEED):
    """Bootstrap CI on cosine similarity between two contrastive directions.

    Resamples stimuli pairs with replacement, recomputes both directions,
    and measures cosine at each layer. This captures uncertainty in the
    direction estimates from finite stimuli — the most relevant source of
    error for direction comparison plots.
    """
    rng = np.random.RandomState(seed)
    n_a, n_b = len(pos_a), len(pos_b)
    boot = {l: [] for l in layers}
    for _ in range(n_boot):
        ia = rng.choice(n_a, n_a, replace=True)
        ib = rng.choice(n_b, n_b, replace=True)
        da = compute_contrastive_direction([pos_a[i] for i in ia],
                                           [neg_a[i] for i in ia])
        db = compute_contrastive_direction([pos_b[i] for i in ib],
                                           [neg_b[i] for i in ib])
        for l in layers:
            boot[l].append(F.cosine_similarity(
                da[l].unsqueeze(0), db[l].unsqueeze(0)).item())
    alpha = (1 - ci) / 2
    return {l: {"mean": float(np.mean(boot[l])),
                "lo": float(np.percentile(boot[l], alpha * 100)),
                "hi": float(np.percentile(boot[l], (1 - alpha) * 100))}
            for l in layers}


def bootstrap_probe_ci(src_pos, src_neg, tgt_pos, tgt_neg, layers,
                        n_boot=200, ci=0.95, seed=SEED):
    """Bootstrap CI on cross-domain probe accuracy.

    Resamples training set with replacement, retrains probe, evaluates on
    full test set. This captures uncertainty from finite training data —
    the dominant error source for probe transfer.
    """
    rng = np.random.RandomState(seed)
    n_src = len(src_pos)
    boot = {l: [] for l in layers}
    for _ in range(n_boot):
        idx = rng.choice(n_src, n_src, replace=True)
        sp = [src_pos[i] for i in idx]
        sn = [src_neg[i] for i in idx]
        for l in layers:
            Xtr = np.concatenate([np.stack([a[l].numpy() for a in sp]),
                                   np.stack([a[l].numpy() for a in sn])])
            ytr = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))])
            Xte = np.concatenate([np.stack([a[l].numpy() for a in tgt_pos]),
                                   np.stack([a[l].numpy() for a in tgt_neg])])
            yte = np.concatenate([np.ones(len(tgt_pos)),
                                  np.zeros(len(tgt_neg))])
            clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            try:
                clf.fit(Xtr, ytr)
                boot[l].append(accuracy_score(yte, clf.predict(Xte)))
            except Exception:
                pass
    alpha = (1 - ci) / 2
    return {l: {"mean": float(np.mean(boot[l])) if boot[l] else float("nan"),
                "lo": float(np.percentile(boot[l], alpha * 100)) if boot[l] else float("nan"),
                "hi": float(np.percentile(boot[l], (1 - alpha) * 100)) if boot[l] else float("nan")}
            for l in layers}


def bootstrap_decomp_ci_by_layer(target_pos, target_neg,
                                  comp_pos_neg_dict, layers,
                                  n_boot=200, ci=0.95, seed=SEED):
    """Bootstrap CI on variance-explained from direction decomposition.

    Resamples stimuli, recomputes target + all component directions,
    runs decomposition. Returns per-layer, per-component CIs on unique
    variance explained and residual.
    """
    rng = np.random.RandomState(seed)
    n = len(target_pos)
    comp_names = list(comp_pos_neg_dict.keys())
    boot = {l: {c: [] for c in comp_names + ["residual"]} for l in layers}
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        tp = [target_pos[i] for i in idx]
        tn = [target_neg[i] for i in idx]
        td = compute_contrastive_direction(tp, tn)
        cds = {}
        for c in comp_names:
            cp, cn = comp_pos_neg_dict[c]
            ci_idx = rng.choice(len(cp), len(cp), replace=True)
            cds[c] = compute_contrastive_direction(
                [cp[i] for i in ci_idx], [cn[i] for i in ci_idx])
        dec = decompose_by_layer(td, cds)
        for l in layers:
            if l not in dec:
                continue
            for c in comp_names:
                boot[l][c].append(
                    dec[l]["unique_variance_explained"].get(c, 0))
            boot[l]["residual"].append(
                dec[l]["residual_variance_fraction"])
    al = (1 - ci) / 2
    result = {}
    for l in layers:
        result[l] = {}
        for c in comp_names + ["residual"]:
            vals = boot[l][c]
            if vals:
                result[l][c] = {
                    "mean": float(np.mean(vals)),
                    "lo": float(np.percentile(vals, al * 100)),
                    "hi": float(np.percentile(vals, (1 - al) * 100))}
            else:
                result[l][c] = {"mean": 0, "lo": 0, "hi": 0}
    return result


def plot_with_ci(ax, x, ci_dict, color, label, ls='-', lw=1.5):
    """Plot a line with shaded 95% CI band from bootstrap_*_ci_by_layer output."""
    means = [ci_dict[l]["mean"] for l in x]
    lo = [ci_dict[l]["lo"] for l in x]
    hi = [ci_dict[l]["hi"] for l in x]
    ax.plot(x, means, ls, color=color, label=label, lw=lw)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15)


# --- activation extraction ---

@torch.no_grad()
def extract_activations(model, input_ids, layers=None):
    """Hidden states at each layer via forward hooks. Returns CPU float32."""
    hidden = {}
    hooks = []
    n_layers = model.config.num_hidden_layers
    targets = set(layers) if layers is not None else set(range(n_layers))

    def make_hook(idx):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hidden[idx] = h.detach().cpu().float().squeeze(0)
        return fn

    for i in targets:
        hooks.append(model.model.layers[i].register_forward_hook(make_hook(i)))
    try:
        model(input_ids)
    finally:
        for h in hooks:
            h.remove()
    return hidden


def extract_completion_acts(model, tokenizer, prompt, completion,
                            layers=None, n_completion_tokens=None,
                            use_chat_template=True):
    """Activations over completion tokens, pooled to one vector per layer.

    To avoid length confounds between sycophantic and therapeutic completions,
    pass n_completion_tokens to truncate both to the same number of tokens
    before pooling.
    """
    if use_chat_template:
        formatted = format_prompt(tokenizer, prompt)
        # Chat template ends with generation prompt (e.g. <|assistant|>\n)
        # Completion follows directly without extra space
        prompt_ids = tokenizer.encode(formatted, return_tensors="pt")
        full_ids = tokenizer.encode(formatted + completion, return_tensors="pt")
    else:
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
        full_ids = tokenizer.encode(prompt + " " + completion, return_tensors="pt")
    full_ids = full_ids.to(get_device(model))
    prompt_len = prompt_ids.shape[1]

    # Truncate completion to fixed length if specified (avoids length confound)
    if n_completion_tokens is not None:
        max_len = prompt_len + n_completion_tokens
        full_ids = full_ids[:, :max_len]

    acts = extract_activations(model, full_ids, layers)
    pooled = {}
    for idx, h in acts.items():
        comp = h[prompt_len:]
        if len(comp) == 0:
            comp = h[-1:]
        pooled[idx] = comp.mean(0)
    return pooled


def batch_extract_contrastive(model, tokenizer, stimuli, pos_key, neg_key,
                              layers=None, n_completion_tokens=None, desc="Extracting",
                              use_chat_template=True):
    # Default changed from 15 → None (use full completion). 15 tokens was
    # too short: v2 completions are 100-250 tokens and the critical
    # sycophantic-validation vs therapeutic-correction content happens
    # around tokens 20-80. Pooling over only the first 15 opening tokens
    # makes the "contrast" a style direction (opening phrase style) rather
    # than a sycophancy direction. This was the root cause of the
    # steering-sign-flip observed in the OLMo-3 7B Instruct DPO run.
    """Extract paired activations for a list of contrastive stimuli.

    n_completion_tokens: truncate both completions to this many tokens
    before extracting, avoiding length confounds in the mean pooling.
    use_chat_template: if False, skip chat formatting for controlled
    cross-checkpoint comparisons where some models lack templates.
    """
    pos_list, neg_list = [], []
    for i, s in enumerate(tqdm(stimuli, desc=desc)):
        pos_list.append(extract_completion_acts(
            model, tokenizer, s["user_prompt"], s[pos_key], layers,
            n_completion_tokens=n_completion_tokens,
            use_chat_template=use_chat_template))
        neg_list.append(extract_completion_acts(
            model, tokenizer, s["user_prompt"], s[neg_key], layers,
            n_completion_tokens=n_completion_tokens,
            use_chat_template=use_chat_template))
        if (i + 1) % 10 == 0:
            cleanup()
    return pos_list, neg_list


# --- contrastive directions ---

def compute_contrastive_direction(pos_acts, neg_acts):
    """Mean-difference direction per layer, normalized to unit length."""
    layers = sorted(pos_acts[0].keys())
    dirs = {}
    for l in layers:
        pos = torch.stack([a[l] for a in pos_acts])
        neg = torch.stack([a[l] for a in neg_acts])
        dirs[l] = F.normalize(pos.mean(0) - neg.mean(0), dim=0)
    return dirs


def cosine_sim_by_layer(dir_a, dir_b):
    layers = sorted(set(dir_a) & set(dir_b))
    return {l: F.cosine_similarity(dir_a[l].unsqueeze(0),
                                    dir_b[l].unsqueeze(0)).item()
            for l in layers}


def permutation_test_cosine(pos_a, neg_a, pos_b, neg_b, layer, n_perms=1000,
                            seed=SEED):
    """Test whether cosine similarity between two contrastive directions at a
    given layer is significantly different from chance.

    Shuffles labels within each domain and recomputes directions to get
    a null distribution. The layer should be pre-registered (e.g., the
    median layer) rather than chosen post-hoc to avoid multiple-testing bias.
    """
    rng = np.random.RandomState(seed)

    # Observed cosine
    dir_a = compute_contrastive_direction(pos_a, neg_a)
    dir_b = compute_contrastive_direction(pos_b, neg_b)
    observed = F.cosine_similarity(dir_a[layer].unsqueeze(0),
                                   dir_b[layer].unsqueeze(0)).item()

    # Null distribution: shuffle pos/neg labels within each domain
    all_a = pos_a + neg_a
    all_b = pos_b + neg_b
    n_a, n_b = len(pos_a), len(pos_b)
    null_cos = []
    for _ in range(n_perms):
        perm_a = rng.permutation(len(all_a))
        perm_b = rng.permutation(len(all_b))
        pa, na = [all_a[j] for j in perm_a[:n_a]], [all_a[j] for j in perm_a[n_a:]]
        pb, nb = [all_b[j] for j in perm_b[:n_b]], [all_b[j] for j in perm_b[n_b:]]
        da = compute_contrastive_direction(pa, na)
        db = compute_contrastive_direction(pb, nb)
        null_cos.append(F.cosine_similarity(
            da[layer].unsqueeze(0), db[layer].unsqueeze(0)).item())

    p_value = float(np.mean([abs(nc) >= abs(observed) for nc in null_cos]))
    null_std = float(np.std(null_cos))
    # NOTE: This is a z-score against the permutation null, NOT Cohen's d
    # (Cohen's d requires pooled SD between two groups). Field `cohens_d` is
    # kept for backward compat but is an alias of `null_z`.
    null_z = (observed - float(np.mean(null_cos))) / null_std if null_std > 0 else 0
    return {"observed": observed, "p_value": p_value,
            "null_mean": float(np.mean(null_cos)),
            "null_std": null_std,
            "null_z": null_z,
            "cohens_d": null_z}  # deprecated alias


# --- probing ---

def within_domain_probing(pos_acts, neg_acts, layers, cv=5):
    """Cross-validated within-domain probe accuracy. Establishes that the
    probe works before testing cross-domain transfer."""
    results = {}
    for l in layers:
        X = np.concatenate([np.stack([a[l].numpy() for a in pos_acts]),
                             np.stack([a[l].numpy() for a in neg_acts])])
        y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        n_cv = min(cv, min(int(y.sum()), int((1-y).sum())))
        if n_cv < 2:
            results[l] = {"mean_accuracy": float("nan"), "std_accuracy": float("nan")}
            continue
        scores = cross_val_score(clf, X, y, cv=n_cv, scoring="accuracy")
        results[l] = {"mean_accuracy": float(scores.mean()),
                      "std_accuracy": float(scores.std())}
    return results


def cross_domain_probing(src_pos, src_neg, tgt_pos, tgt_neg, layers):
    """Train probe on source domain, test on target. Returns per-layer acc/auc."""
    results = {}
    for l in layers:
        Xtr = np.concatenate([np.stack([a[l].numpy() for a in src_pos]),
                               np.stack([a[l].numpy() for a in src_neg])])
        ytr = np.concatenate([np.ones(len(src_pos)), np.zeros(len(src_neg))])
        Xte = np.concatenate([np.stack([a[l].numpy() for a in tgt_pos]),
                               np.stack([a[l].numpy() for a in tgt_neg])])
        yte = np.concatenate([np.ones(len(tgt_pos)), np.zeros(len(tgt_neg))])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs").fit(Xtr, ytr)
        pred = clf.predict(Xte)
        prob = clf.predict_proba(Xte)[:, 1]
        try:
            auc = roc_auc_score(yte, prob)
        except ValueError:
            auc = float("nan")
        results[l] = {"accuracy": accuracy_score(yte, pred), "auc": auc}
    return results


# --- variance decomposition ---

def decompose_direction(target, components):
    """Gram-Schmidt decomposition of target into named components + residual."""
    total_var = target.norm().item() ** 2
    residual = target.clone()
    result = {"projections": {}, "variance_explained": {},
              "unique_variance_explained": {}}

    for name, comp in components.items():
        cn = F.normalize(comp, dim=0)
        proj = (target @ cn).item()
        result["projections"][name] = proj
        result["variance_explained"][name] = proj ** 2 / total_var if total_var > 0 else 0

    used_dirs = []
    for name, comp in components.items():
        cn = F.normalize(comp, dim=0)
        # Orthogonalize against all previously used directions
        for prev in used_dirs:
            cn = cn - (cn @ prev) * prev
        cn_norm = cn.norm()
        if cn_norm < 1e-8:
            result["unique_variance_explained"][name] = 0.0
            continue
        cn = cn / cn_norm
        before = residual.norm().item() ** 2
        residual = residual - (residual @ cn) * cn
        after = residual.norm().item() ** 2
        result["unique_variance_explained"][name] = (
            (before - after) / total_var if total_var > 0 else 0)
        used_dirs.append(cn)

    result["residual_variance_fraction"] = (
        residual.norm().item() ** 2 / total_var if total_var > 0 else 0)
    return result


def decompose_by_layer(target_dirs, component_dirs_dict):
    results = {}
    for l in sorted(target_dirs.keys()):
        comps = {n: d[l] for n, d in component_dirs_dict.items() if l in d}
        results[l] = decompose_direction(target_dirs[l], comps)
    return results


# --- logit lens ---

@torch.no_grad()
def logit_lens(model, input_ids, position=-1):
    """Project each layer's hidden state through layernorm + unembedding."""
    device = get_device(model)
    model_dtype = next(model.parameters()).dtype
    hidden = {}
    hooks = []

    def make_hook(idx):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hidden[idx] = h.detach().cpu()
        return fn

    for i in range(model.config.num_hidden_layers):
        hooks.append(model.model.layers[i].register_forward_hook(make_hook(i)))
    try:
        model(input_ids.to(device))
    finally:
        for h in hooks:
            h.remove()

    norm = model.model.norm
    lm_head = model.lm_head
    logits = {}
    for i in sorted(hidden.keys()):
        h = hidden[i][0, position, :].unsqueeze(0).unsqueeze(0)
        h = h.to(device=device, dtype=model_dtype)
        logits[i] = lm_head(norm(h)).squeeze().float().cpu()
    return logits


def compute_correct_signal(model, tokenizer, prompt, ther_comp, syc_comp,
                           n_tokens=3, use_chat_template=True):
    """log P(therapeutic) - log P(sycophantic) at each layer.

    Uses the average log-prob over the first n_tokens of each completion
    rather than just the first token, to reduce sensitivity to shared
    opening words.
    """
    ther_toks = tokenizer.encode(ther_comp, add_special_tokens=False)[:n_tokens]
    syc_toks = tokenizer.encode(syc_comp, add_special_tokens=False)[:n_tokens]
    formatted = format_prompt(tokenizer, prompt) if use_chat_template else prompt
    ids = tokenizer.encode(formatted, return_tensors="pt")
    layer_logits = logit_lens(model, ids)

    signal = {}
    for l, lg in layer_logits.items():
        lp = F.log_softmax(lg, dim=-1)
        ther_lp = np.mean([lp[t].item() for t in ther_toks])
        syc_lp = np.mean([lp[t].item() for t in syc_toks])
        signal[l] = ther_lp - syc_lp
    return signal


# --- steering ---

def steer_hook(direction, alpha):
    """Returns a hook that subtracts alpha * direction from ALL positions.

    Steering all positions (not just -1) ensures the intervention persists
    across autoregressive generation steps.
    """
    def fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h -= alpha * direction
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h
    return fn


def measure_steering_shift(model, tokenizer, stimuli, layer, direction, alpha,
                           n_random=100, seed=SEED, use_chat_template=True):
    """Measure logit shift from steering, with random-vector baseline for z-score.

    Returns mean_shift, z_score, per-stimulus shifts, and random shifts.
    """
    rng = np.random.RandomState(seed)
    device = get_device(model)
    dtype = next(model.parameters()).dtype
    vec = direction.to(device=device, dtype=dtype)

    # Baseline and steered logit diffs
    baseline_diffs, steered_diffs = [], []
    for s in stimuli:
        prompt = format_prompt(tokenizer, s["user_prompt"]) if use_chat_template else s["user_prompt"]
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        ther_tok = tokenizer.encode(s["therapeutic_completion"], add_special_tokens=False)[0]
        syc_tok = tokenizer.encode(s["sycophantic_completion"], add_special_tokens=False)[0]

        with torch.no_grad():
            logits_b = model(ids).logits[0, -1, :]
        lp_b = F.log_softmax(logits_b.float(), dim=-1)
        baseline_diffs.append((lp_b[ther_tok] - lp_b[syc_tok]).item())

        hook = model.model.layers[layer].register_forward_hook(steer_hook(vec, alpha))
        try:
            with torch.no_grad():
                logits_s = model(ids).logits[0, -1, :]
        finally:
            hook.remove()
        lp_s = F.log_softmax(logits_s.float(), dim=-1)
        steered_diffs.append((lp_s[ther_tok] - lp_s[syc_tok]).item())

    shifts = [s - b for s, b in zip(steered_diffs, baseline_diffs)]
    mean_shift = float(np.mean(shifts))

    # Random baseline
    random_shifts = []
    for _ in range(n_random):
        rv = torch.randn_like(vec)
        rv = F.normalize(rv, dim=0) * vec.norm()
        rand_diffs = []
        for s in stimuli:
            prompt = format_prompt(tokenizer, s["user_prompt"]) if use_chat_template else s["user_prompt"]
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            ther_tok = tokenizer.encode(s["therapeutic_completion"], add_special_tokens=False)[0]
            syc_tok = tokenizer.encode(s["sycophantic_completion"], add_special_tokens=False)[0]
            hook = model.model.layers[layer].register_forward_hook(steer_hook(rv, alpha))
            try:
                with torch.no_grad():
                    logits_r = model(ids).logits[0, -1, :]
            finally:
                hook.remove()
            lp_r = F.log_softmax(logits_r.float(), dim=-1)
            rand_diffs.append((lp_r[ther_tok] - lp_r[syc_tok]).item())
        random_shifts.append(float(np.mean(rand_diffs)) - float(np.mean(baseline_diffs)))

    rand_std = float(np.std(random_shifts)) if len(random_shifts) > 1 else 1.0
    z = (mean_shift - float(np.mean(random_shifts))) / max(rand_std, 1e-8)

    return {"mean_shift": mean_shift, "z_score": z, "shifts": shifts,
            "random_shifts": random_shifts, "baseline_mean": float(np.mean(baseline_diffs))}
