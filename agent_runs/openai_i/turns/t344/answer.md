We now have matched states from 100 training stays. I’ll fit a closed-form value-head correction to the routing distortion, while protecting the directions produced by dose changes so the correction does not simply erase the repair. Held-out training stays provide an initial check; GPU factual and rollout tests will decide whether it is usable.

```python
import sys
from scipy.special import logsumexp

def array294(x):
    return x.detach().cpu().numpy().copy() if torch.is_tensor(x) else np.asarray(x).copy()

original294 = np.load("/work/fwd_38/states_original.npz")
routed294 = np.load("/work/fwd_38/states_edited.npz")
assert np.array_equal(original294["stay_id"], routed294["stay_id"])
assert np.array_equal(original294["hour"], routed294["hour"])
xorig294, xroute294 = original294["x"], routed294["x"]
sid294, hour294 = original294["stay_id"], original294["hour"]

savedW294 = W
threads294 = torch.get_num_threads()
W = {n: torch.as_tensor(load_original(n), dtype=torch.float32) for n in NAMES}

def capture294(fn, *args):
    captured = {}
    old_profile = sys.getprofile()
    mlp_args = mlp136.__code__.co_varnames[:mlp136.__code__.co_argcount]
    def profile(frame, event, arg):
        if frame.f_code is mlp136.__code__:
            if frame.f_locals.get(mlp_args[1]) == "attr_heads.value_bin":
                if event == "call":
                    captured["input"] = array294(frame.f_locals[mlp_args[0]])
                elif event == "return":
                    captured["head_output"] = array294(arg)
        if frame.f_code is fn.__code__ and event == "return":
            for key in ("logits", "last_bin", "last_hour", "last_value"):
                if key in frame.f_locals:
                    value = frame.f_locals[key]
                    captured[key] = array294(value) if torch.is_tensor(value) else value
    sys.setprofile(profile)
    try:
        result = fn(*args)
    finally:
        sys.setprofile(old_profile)
    return captured, result

inputs_o294, inputs_e294, logits_o294, logits_e294 = [], [], [], []
case_stays294, labels294 = [], []
protected_cases294 = []
try:
    torch.set_num_threads(1)
    records294 = pd.read_csv("/data/train.csv", nrows=50000)
    records294 = records294[
        records294.stay_id.isin(np.unique(sid294))
        & records294.event.isin(events182)
        & records294.value.notna()
    ]
    by_stay294 = {
        int(s): np.flatnonzero(sid294 == s) for s in np.unique(sid294)
    }
    example294 = None
    for row in records294.itertuples(index=False):
        s, t, event = int(row.stay_id), float(row.hour), row.event
        positions = by_stay294[s]
        j = np.searchsorted(hour294[positions], t - 1e-6, side="left") - 1
        if j < 0:
            continue
        i = positions[j]
        if abs(np.floor(hour294[i]) + 1.0 - t) > 1e-3:
            continue
        args = (s, float(hour294[i]), event)
        co, _ = capture294(
            readout153,
            torch.as_tensor(xorig294[i, 192:960].reshape(1, 4, 192)),
            torch.as_tensor(xorig294[i, :192].reshape(1, 192)), *args
        )
        ce, _ = capture294(
            readout153,
            torch.as_tensor(xroute294[i, 192:960].reshape(1, 4, 192)),
            torch.as_tensor(xroute294[i, :192].reshape(1, 192)), *args
        )
        ct, _ = capture294(
            readout153,
            torch.as_tensor(xorig294[i, 192:960].reshape(1, 4, 192)),
            torch.as_tensor(xorig294[i, :192].reshape(1, 192)),
            s, t + 1e-4, event
        )
        assert "input" in co and "input" in ce, (co.keys(), ce.keys())
        label = -1
        if (
            "last_bin" in ct and "last_value" in ct
            and abs(float(ct["last_value"]) - float(row.value)) < 1e-3
        ):
            label = int(ct["last_bin"])
        inputs_o294.append(co["input"].reshape(192))
        inputs_e294.append(ce["input"].reshape(192))
        logits_o294.append(np.asarray(co["logits"]).reshape(17))
        logits_e294.append(np.asarray(ce["logits"]).reshape(17))
        case_stays294.append(s)
        labels294.append(label)
        if example294 is None:
            example294 = ce

    inputs_o294 = np.asarray(inputs_o294, dtype=np.float32)
    inputs_e294 = np.asarray(inputs_e294, dtype=np.float32)
    logits_o294 = np.asarray(logits_o294, dtype=np.float64)
    logits_e294 = np.asarray(logits_e294, dtype=np.float64)
    case_stays294 = np.asarray(case_stays294)
    labels294 = np.asarray(labels294)
    assert len(inputs_e294) > 1000

    w1_294 = W["attr_heads.value_bin.0.weight"]
    b1_294 = W["attr_heads.value_bin.0.bias"]
    w2_294 = W["attr_heads.value_bin.2.weight"]
    b2_294 = W["attr_heads.value_bin.2.bias"]
    activations294 = {
        "gelu": lambda a: F.gelu(a),
        "gelu_tanh": lambda a: F.gelu(a, approximate="tanh"),
        "silu": F.silu,
        "relu": F.relu
    }
    first294 = F.linear(torch.as_tensor(example294["input"]), w1_294, b1_294)
    activation_errors294 = {
        name: float(np.max(np.abs(
            array294(F.linear(act(first294), w2_294, b2_294))
            - example294["head_output"]
        ))) for name, act in activations294.items()
    }
    activation_name294 = min(activation_errors294, key=activation_errors294.get)
    assert activation_errors294[activation_name294] < 1e-4, activation_errors294
    def hidden294(x):
        shape = np.asarray(x).shape
        h = activations294[activation_name294](
            F.linear(torch.as_tensor(np.asarray(x).reshape(-1, 192)), w1_294, b1_294)
        )
        return array294(h).reshape(shape)

    ho294, he294 = hidden294(inputs_o294), hidden294(inputs_e294)
    torch.set_num_threads(min(8, threads294))

    def route_head294(action_inputs):
        hh, oo = replay_route228(corrected269, action_inputs=action_inputs)
        cap, _ = capture294(readout214, hh, oo)
        return hidden294(cap["input"]), np.asarray(cap["logits"], dtype=np.float64)

    hbase294, lbase294 = route_head294(Ubase215)
    directions294 = []
    for drug in ("insulin", "norepinephrine"):
        sel = np.asarray(drugs182) == drug
        for scale in (0.0, 0.5, 2.0, 3.0):
            hc, lc = route_head294(actions229[(drug, scale)])
            directions294.append((hc[sel] - hbase294[sel]).reshape(-1, 192))
            protected_cases294.append((drug, scale, sel, hc, lc))
    directions294 = np.concatenate(directions294).astype(np.float64)
    _, singular294, vh294 = np.linalg.svd(directions294, full_matrices=False)
    energy294 = np.cumsum(singular294**2) / np.sum(singular294**2)

    training294 = case_stays294 < 1010000070
    held294 = ~training294
    valid294 = labels294 >= 0
    def nll294(logits, mask):
        m = mask & valid294
        return float(np.mean(
            logsumexp(logits[m], axis=1)
            - logits[m, labels294[m]]
        )) if m.any() else np.nan

    target294 = logits_o294 - logits_e294
    target294 -= target294.mean(axis=1, keepdims=True)
    def probabilities294(logits):
        return np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
    base_means294 = np.sum(probabilities294(lbase294) * centres182[None], axis=-1)

    def fit294(rank, ridge, mask):
        p = np.eye(192) - vh294[:rank].T @ vh294[:rank]
        xp = he294.astype(np.float64) @ p
        mu = xp[mask].mean(axis=0)
        sd = np.maximum(xp[mask].std(axis=0), 0.05)
        z = (xp - mu) / sd
        ym = target294[mask].mean(axis=0)
        coef = np.linalg.solve(
            z[mask].T @ z[mask] + ridge * mask.sum() * np.eye(192),
            z[mask].T @ (target294[mask] - ym)
        )
        raw = coef / sd[:, None]
        return p @ raw, ym - mu @ raw

    results294 = []
    for rank in (4, 8, 12, 16, 24, 32):
        for ridge in (0.001, 0.01, 0.1, 1.0):
            beta, bias = fit294(rank, ridge, training294)
            new_logits = logits_e294 + he294 @ beta + bias
            rms = float(np.sqrt(np.mean(
                (new_logits[held294] - logits_o294[held294]
                 - (new_logits[held294] - logits_o294[held294]).mean(axis=1, keepdims=True))**2
            )))
            protection = []
            for drug, scale, sel, hc, lc in protected_cases294:
                if scale not in (0.5, 2.0):
                    continue
                own = events182.index("glucose_finger" if drug == "insulin" else "map_arterial")
                old_shift = float(np.mean(
                    np.sum(probabilities294(lc[sel]) * centres182[None], axis=-1)[:, own]
                    - base_means294[sel, own]
                ))
                new_base = np.sum(
                    probabilities294(lbase294[sel] + hbase294[sel] @ beta + bias)
                    * centres182[None], axis=-1
                )
                new_changed = np.sum(
                    probabilities294(lc[sel] + hc[sel] @ beta + bias)
                    * centres182[None], axis=-1
                )
                new_shift = float(np.mean(new_changed[:, own] - new_base[:, own]))
                protection.append(abs(new_shift - old_shift) / max(abs(old_shift), 0.25))
            dnll = nll294(new_logits, held294) - nll294(logits_o294, held294)
            results294.append((dnll, rms, max(protection), rank, ridge))

    eligible294 = [r for r in results294 if r[2] <= 0.15]
    assert eligible294, sorted(results294)[:5]
    best294 = min(
        eligible294,
        key=lambda r: r[0] if np.isfinite(r[0]) else r[1]
    )
    beta294, bias294 = fit294(best294[3], best294[4], np.ones(len(he294), dtype=bool))
    candidate294 = {k: array294(v) for k, v in corrected269.items()}
    candidate294["attr_heads.value_bin.2.weight"] = (
        array294(w2_294) + beta294.T
    ).astype(np.float32)
    candidate294["attr_heads.value_bin.2.bias"] = (
        array294(b2_294) + bias294
    ).astype(np.float32)

    print("Cases / labelled / held:", len(he294), valid294.sum(), held294.sum())
    print("Activation:", activation_name294, activation_errors294[activation_name294])
    print("Held original / routed NLL:",
          nll294(logits_o294, held294), nll294(logits_e294, held294))
    print("Protected direction energy:",
          {r: round(float(energy294[r-1]), 5) for r in (4, 8, 16, 32)})
    print("Best (held dNLL, logit RMS, dose distortion, rank, ridge):", best294)
    print("Top eligible:", sorted(eligible294)[:5])

finally:
    W = savedW294
    torch.set_num_threads(threads294)

if "candidate294" in globals():
    for name294, tensor294 in candidate294.items():
        save_edit(name294, tensor294)
    np.savez("/work/protected_head294.npz", **candidate294)
    print("Installed provisional protected-head compensation:", edited())
    with open("/work/notes.md", "a") as f:
        f.write(
            "\n## Python 294 — protected closed-form readout compensation\n"
            f"- Paired teacher-forced training-state cases: {len(he294)}; labels recovered: {valid294.sum()}.\n"
            f"- Held local original/routed NLL: {nll294(logits_o294, held294):.6f} / {nll294(logits_e294, held294):.6f}.\n"
            f"- Selected (held dNLL, RMS, local dose distortion, protected rank, ridge): {best294}.\n"
            "- Regressed original-minus-routed logits on routed value-head hidden features, constrained away from dose-response directions.\n"
            "- Installed provisionally; GPU factual and dose validation are mandatory. No model gradients or weight optimiser used.\n"
        )
```