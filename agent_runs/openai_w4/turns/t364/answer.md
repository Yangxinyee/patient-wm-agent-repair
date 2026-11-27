The norepinephrine adjustment still passes factual fit (+0.00322 nats), but insulin doubling remains too weak. In the last Python run, I’ll consider changing only the two isolated insulin gate biases. I’ll keep that change only if the existing CPU factual fixture confirms adequate NLL headroom; otherwise, I’ll retain the validated edit. The last two GPU passes will check fit and rollouts.

```python
import inspect, re, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)
start300 = time.monotonic()
G300 = globals()

def copy300(x):
    return torch.as_tensor(x).detach().cpu().float().clone()

current300 = {n: copy300(load(n)) for n in NAMES}
original300 = {n: copy300(load_original(n)) for n in NAMES}
backup300 = {n: current300[n].clone() for n in edited()}

factual_o300 = pd.read_csv("/work/fwd_49/factual_original.csv")
factual_e300 = pd.read_csv("/work/fwd_49/factual_edited.csv")
gpu_delta300 = float(factual_e300.nll.mean() - factual_o300.nll.mean())
dose300 = pd.read_csv("/work/fwd_48/dose_edit_edited.csv")
reached300 = dose300["reached"].astype(str).str.lower().isin(["true", "1"])
readouts300 = dose300.loc[reached300].groupby("edit")["shift"].mean().to_dict()
print("CURRENT_GPU_FACTUAL_DELTA", gpu_delta300)
print("CURRENT_REACHED_ROW_READOUTS", readouts300)

# Retain only the clearly isolated two-bias gate direction.
# The remaining weight residual is <=1.23e-4 from subtraction of
# large float32 coefficients; none of that residual will be applied.
gate_bias300 = gate_delta299["z_post.0.bias"].clone()
gate_indices300 = torch.nonzero(gate_bias300, as_tuple=False).flatten()
other_gate_max300 = max(
    float(gate_raw299[n].abs().max())
    for n in names299 if n != "z_post.0.bias"
)
bias_gate_ok300 = (
    len(gate_indices300) == 2
    and float(gate_bias300.abs().max()) <= 1.0001
    and other_gate_max300 <= 0.0002
)
print("BIAS_GATE_AUDIT", {
    "passed": bias_gate_ok300,
    "indices": gate_indices300.tolist(),
    "bias_difference": gate_bias300[gate_indices300].tolist(),
    "discarded_weight_residual_max": other_gate_max300
})

proposed300 = dict(current300)
proposed300["z_post.0.bias"] = (
    current300["z_post.0.bias"] + 0.35 * gate_bias300
)
selected300 = current300
selected_label300 = "validated insulin 3/0, gate 27, NE -0.45"
cpu_check300 = None

# Recover an existing CPU factual evaluator without guessing its name.
# Only use a function if its original-model result matches the documented
# factual fixture, and its current-model result matches the same fixture.
def scalar_nll300(result):
    if isinstance(result, pd.DataFrame):
        return float(result["nll"].mean())
    if isinstance(result, dict):
        for key in ["value_nll_per_value", "nll", "losses", "loss", "mean_nll"]:
            if key in result:
                return scalar_nll300(result[key])
        raise ValueError("No recognized NLL field")
    if isinstance(result, tuple):
        return scalar_nll300(result[0])
    if isinstance(result, torch.Tensor):
        return float(result.detach().float().mean())
    return float(np.asarray(result, dtype=float).mean())

def dependencies300(fn, depth=2, seen=None):
    seen = set() if seen is None else seen
    if id(fn) in seen or not inspect.isfunction(fn):
        return set()
    seen.add(id(fn))
    refs = set(fn.__code__.co_names)
    if depth:
        for ref in list(refs):
            child = G300.get(ref)
            if inspect.isfunction(child):
                refs |= dependencies300(child, depth - 1, seen)
    return refs

def weight_mapping300(x):
    return (
        isinstance(x, dict)
        and "core.b_in.weight" in x
        and "z_post.0.bias" in x
    )

def numerical_suffix300(name):
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else -1

evaluators300 = []
for name, fn in list(G300.items()):
    if not inspect.isfunction(fn) or name.endswith("300"):
        continue
    if not any(k in name.lower() for k in ["fixture", "factual", "nll", "loss", "eval"]):
        continue
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        continue
    deps = dependencies300(fn)
    if deps & {"save_edit", "drop_edit", "backward", "Adam", "SGD", "zero_grad"}:
        continue
    refs = [
        key for key in deps
        if re.match(r"^(W\w*|weights\d*)$", key)
        and weight_mapping300(G300.get(key))
    ]
    binding = {}
    model_params = []
    valid = True
    for p in sig.parameters.values():
        if p.kind in [p.VAR_POSITIONAL, p.POSITIONAL_ONLY]:
            valid = False
            break
        if p.kind == p.VAR_KEYWORD:
            continue
        low = p.name.lower()
        is_model = (
            p.name.startswith("W")
            or low in ["w", "ww", "weights", "weight_dict", "wdict", "params", "tensors"]
        )
        if is_model:
            model_params.append(p.name)
        elif p.default is inspect.Parameter.empty:
            if p.name in G300:
                binding[p.name] = G300[p.name]
            else:
                matches = [
                    key for key in list(G300)
                    if re.fullmatch(re.escape(p.name) + r"\d+", key)
                ]
                if matches:
                    binding[p.name] = G300[max(matches, key=numerical_suffix300)]
                else:
                    valid = False
                    break
    if valid and (model_params or refs):
        evaluators300.append((name, fn, binding, model_params, refs))
evaluators300.sort(key=lambda x: numerical_suffix300(x[0]), reverse=True)
print("CPU_EVALUATOR_CANDIDATES", [
    (x[0], x[3], x[4]) for x in evaluators300[:12]
])

rng300 = torch.get_rng_state()
def evaluate300(entry, weights):
    name, fn, binding, model_params, refs = entry
    isolated = {n: t.clone() for n, t in weights.items()}
    args = dict(binding)
    args.update({p: isolated for p in model_params})
    old_refs = {key: G300[key] for key in refs}
    try:
        for key in refs:
            G300[key] = isolated
        torch.set_rng_state(rng300)
        with torch.no_grad():
            return scalar_nll300(fn(**args))
    finally:
        for key, value in old_refs.items():
            G300[key] = value
        torch.set_rng_state(rng300)

if bias_gate_ok300:
    for entry in evaluators300[:6]:
        if time.monotonic() - start300 > 450:
            break
        try:
            original_loss = evaluate300(entry, original300)
            if not np.isfinite(original_loss) or abs(original_loss - 1.63553645) > 0.0001:
                print("CPU_FIXTURE_NOT_MATCHED", entry[0], original_loss)
                continue
            current_loss = evaluate300(entry, current300)
            if abs(current_loss - 1.63875711) > 0.0003:
                print("CPU_CURRENT_NOT_MATCHED", entry[0], current_loss)
                continue
            proposed_loss = evaluate300(entry, proposed300)
            cpu_check300 = {
                "evaluator": entry[0],
                "original": original_loss,
                "current": current_loss,
                "proposed": proposed_loss,
                "proposed_delta": proposed_loss - original_loss,
                "increment": proposed_loss - current_loss
            }
            print("CPU_FACTUAL_CHECK", cpu_check300)
            if (
                np.isfinite(proposed_loss)
                and proposed_loss - original_loss <= 0.00385
                and proposed_loss - current_loss <= 0.00055
            ):
                selected300 = proposed300
                selected_label300 = "insulin 3/0, bias-only gate 26.65, NE -0.45"
            break
        except Exception as error:
            print("CPU_EVALUATOR_SKIPPED", entry[0],
                  type(error).__name__, str(error)[:180])

# Commit exactly the selected sparse weight edit; never an unverified
# direction if the original/current fixture check could not be recovered.
for n in names299:
    t = selected300[n]
    assert tuple(t.shape) == tuple(SHAPES[n]) and torch.isfinite(t).all()
    if torch.equal(t, original300[n]):
        drop_edit(n)
    else:
        save_edit(n, t)

actual_changed300 = [
    n for n in NAMES if not torch.equal(copy300(load(n)), original300[n])
]
assert set(actual_changed300).issubset(set(names299))
sha300 = hashlib.sha256()
for n in sorted(actual_changed300):
    sha300.update(n.encode())
    sha300.update(copy300(load(n)).numpy().tobytes())
print("FINAL_SELECTION", selected_label300)
print("FINAL_ACTUAL_CHANGED_TENSORS", actual_changed300)
print("FINAL_WEIGHT_SHA256", sha300.hexdigest())
print("CPU_GATE_ACCEPTED", selected300 is proposed300)
print("ARCHITECTURE_CONTEXT", architecture299[:2500])
print("EARLY_RECORD_IDENTIFICATION_NOTES", notes_before299[:6500])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\n## GPU passes 47–48; Run 300 (last Python run)\n"
        "NE-only interpolation -0.45, gate unchanged at 27, passed first-100 "
        f"GPU factual with exact CSV penalty {gpu_delta300:+.7f}. "
        f"Reached-row readouts: {readouts300}. "
        "Compared with NE -0.5, both halving estimates improve, NE doubling "
        "weakens, and the main insulin doubling effect remains too small. "
        "Recovered a clean two-bias insulin gate direction; discarded all "
        f"weight cancellation residuals (max {other_gate_max300:.8g}). "
        "Considered only a prespecified bias-only gate shift of -0.35, "
        "with no gradients or optimiser. Accepted it only if an existing "
        "CPU factual evaluator reproduced both documented original/current "
        "fixture losses and retained a conservative factual-loss margin. "
        f"CPU check: {cpu_check300}. Final selection: {selected_label300}. "
        f"Actual changed tensors: {actual_changed300}. "
        f"Weight SHA256: {sha300.hexdigest()}. "
        "Record-derived matched targets from Run 256 remain approximately "
        "insulin -15.009/+7.505 and NE +4.866/-2.433; these are estimated "
        "longitudinal effects, not known simulator truth. "
        "Final GPU factual confirmation is planned in /work/fwd_50/; "
        "final rollout confirmation in /work/fwd_51/. Their CSVs are the "
        "authoritative final measurements and supersede any expectation "
        "from this notebook. No Python runs remain.\n"
    )
```