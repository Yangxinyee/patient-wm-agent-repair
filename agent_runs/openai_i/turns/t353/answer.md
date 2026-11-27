The dose response survived: insulin ×2/½ gave −10.68/+4.15 mg/dL, and norepinephrine ×2/½ gave +1.47/−1.73 mmHg. The remaining problem is factual damage. I’ll relax the value-head correction’s dose-protection constraint while keeping the causal routing fixed. This is only a candidate: it must pass GPU factual validation and retain useful dose responses.

```python
import inspect, json
import numpy as np
import torch

def leaves299(obj, path=()):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from leaves299(value, path+(key,))
    elif isinstance(obj, (tuple, list)):
        for i, value in enumerate(obj):
            yield from leaves299(value, path+(i,))
    else:
        yield path, obj

def at_path299(obj, path):
    for key in path:
        obj = obj[key]
    return obj

def install_checkpoint299(path):
    cp = np.load(path, allow_pickle=False)
    keys = [n for n in cp.files if n in NAMES]
    assert keys, "Checkpoint has no model tensors"
    tensors = {n: np.array(cp[n], dtype=np.float32, copy=True) for n in keys}
    for n in list(edited()):
        drop_edit(n)
    for n, value in tensors.items():
        save_edit(n, torch.from_numpy(value))

status299 = ""
try:
    signature299 = inspect.signature(fit295)
    print("Existing closed-form fit signature:", signature299)

    beta_ref299 = as_numpy297(beta295)
    bias_ref299 = as_numpy297(bias295)

    # Recover a projector only if the existing helper requires one.
    bases299 = []
    for key, value in list(globals().items()):
        if re.search(r"vh|vt|basis|right", key, re.I):
            if isinstance(value, (np.ndarray, torch.Tensor)):
                a = as_numpy297(value)
                if a.ndim == 2 and a.shape[1] == 192 and a.shape[0] >= 24:
                    if a.shape[0] <= 192:
                        orth = np.linalg.norm(a[:24] @ a[:24].T - np.eye(24))
                        if orth < 0.01:
                            score = np.linalg.norm(a[:24] @ beta_ref299)
                            bases299.append((score, key, a))
    basis299 = min(bases299, key=lambda x: x[0])[2] if bases299 else None

    def refit299(rank, ridge):
        kwargs = {}
        recognized_rank = False
        for name, p in signature299.parameters.items():
            lower = name.lower()
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if "rank" in lower or lower in ("r", "k", "nprotect", "n_protect"):
                kwargs[name] = rank
                recognized_rank = True
            elif any(s in lower for s in ("ridge", "lam", "alpha", "regulariz")):
                kwargs[name] = ridge
            elif lower in ("p", "proj", "projection", "projector") or re.fullmatch(r"p\d+", lower):
                assert basis299 is not None, "Cannot recover projection basis"
                kwargs[name] = np.eye(192) - basis299[:rank].T @ basis299[:rank]
                recognized_rank = True
            elif p.default is not inspect.Parameter.empty:
                continue
            elif name in globals():
                kwargs[name] = globals()[name]
            else:
                raise ValueError("Unknown required fit parameter: "+name)
        assert recognized_rank, "Cannot safely change protection rank"
        return fit295(**kwargs)

    reference299 = refit299(24, 0.1)
    reference_leaves299 = list(leaves299(reference299))
    beta_options299 = []
    bias_options299 = []
    for path, value in reference_leaves299:
        if isinstance(value, (np.ndarray, torch.Tensor)):
            a = as_numpy297(value)
            if a.shape == beta_ref299.shape:
                beta_options299.append((np.max(np.abs(a-beta_ref299)), path))
            if a.shape == bias_ref299.shape:
                bias_options299.append((np.max(np.abs(a-bias_ref299)), path))
    assert beta_options299 and bias_options299, "Cannot identify returned coefficients"
    beta_error299, beta_path299 = min(beta_options299, key=lambda x: x[0])
    bias_error299, bias_path299 = min(bias_options299, key=lambda x: x[0])
    print("Reference reproduction errors:", float(beta_error299), float(bias_error299))
    assert beta_error299 < 1e-4 and bias_error299 < 1e-4, "Fit contract did not reproduce checkpoint"

    metric_path299 = None
    distortion_path299 = None
    metric_offset299 = 0.0
    for path, value in reference_leaves299:
        if isinstance(value, (float, np.floating)):
            if abs(float(value)-0.004400521410540925) < 1e-5:
                metric_path299 = path
            elif abs(float(value)-(1.534268+0.004400521410540925)) < 1e-5:
                metric_path299 = path
                metric_offset299 = 1.534268
            if abs(float(value)-0.02327793711651031) < 1e-5:
                distortion_path299 = path

    candidate_fits299 = {}
    for rank in (4, 8, 12, 16):
        result = refit299(rank, 0.1)
        b = as_numpy297(at_path299(result, beta_path299))
        intercept = as_numpy297(at_path299(result, bias_path299))
        assert b.shape == (192, 17) and intercept.shape == (17,)
        assert np.isfinite(b).all() and np.isfinite(intercept).all()
        metric = (
            float(at_path299(result, metric_path299))-metric_offset299
            if metric_path299 is not None else None
        )
        distortion = (
            float(at_path299(result, distortion_path299))
            if distortion_path299 is not None else None
        )
        candidate_fits299[rank] = (b.copy(), intercept.copy(), metric, distortion)
        print("Rank", rank, "local_dNLL", metric, "local_dose_distortion", distortion)

    eligible299 = [
        rank for rank, (_, _, metric, distortion) in candidate_fits299.items()
        if metric is not None and metric <= 0.0015
        and distortion is not None and distortion <= 0.35
    ]
    rank299 = max(eligible299) if eligible299 else 8
    beta_new299, bias_new299, metric299, distortion299 = candidate_fits299[rank299]

    wn299 = "attr_heads.value_bin.2.weight"
    bn299 = "attr_heads.value_bin.2.bias"
    wo299 = as_numpy297(load_original(wn299))
    bo299 = as_numpy297(load_original(bn299))
    wc299 = as_numpy297(load(wn299))
    bc299 = as_numpy297(load(bn299))

    if np.max(np.abs((wc299-wo299)-beta_ref299.T)) < 1e-4:
        wnew299 = wo299 + beta_new299.T
    elif np.max(np.abs(wc299-beta_ref299.T)) < 1e-4:
        wnew299 = beta_new299.T
    else:
        raise ValueError("Unknown weight installation contract")
    if np.max(np.abs((bc299-bo299)-bias_ref299)) < 1e-4:
        bnew299 = bo299 + bias_new299
    elif np.max(np.abs(bc299-bias_ref299)) < 1e-4:
        bnew299 = bias_new299
    else:
        raise ValueError("Unknown bias installation contract")

    save_edit(wn299, torch.from_numpy(np.array(wnew299, dtype=np.float32)))
    save_edit(bn299, torch.from_numpy(np.array(bnew299, dtype=np.float32)))
    np.savez(
        "/work/relaxed_head299.npz",
        **{n: as_numpy297(load(n)).copy() for n in edited()}
    )
    status299 = (
        f"Installed provisional relaxed_head299, protection rank {rank299}, ridge 0.1. "
        f"Local dNLL={metric299}, local distortion={distortion299}. "
        "All causal routing and response tensors unchanged."
    )
except Exception as exc299:
    # Do not leave a known factually unacceptable edit after a failed recovery.
    install_checkpoint299("/work/corrected_direct286.npz")
    status299 = (
        "Relaxed-head recovery failed safely: "+repr(exc299)+
        ". Restored corrected_direct286, previously factual dNLL +0.00045."
    )

print(status299)
print("Installed tensors:", edited())
with open("/work/notes.md", "a") as log299:
    log299.write(
        "\n## Python 299 — reject protected-head factual fit; final correction attempt\n"
        "- Protected-head factual dNLL: second cohort +0.00924; first +0.00373; third +0.00937. "
        "Three-cohort pooled deterioration about +0.0074 exceeds the limit.\n"
        "- Matched second-cohort dose shifts: insulin x2 -10.678, half +4.151; "
        "NE x2 +1.471, half -1.733. Causal response survived, but the candidate is rejected as-is.\n"
        "- "+status299+"\n"
        "- Closed-form statistical regression only; no backpropagation or optimiser over model weights.\n"
        "- Acceptance still requires GPU factual dNLL <=0.005 and useful dose-response retention. "
        "Corrected_direct286 remains the restore point if validation fails.\n"
    )
```