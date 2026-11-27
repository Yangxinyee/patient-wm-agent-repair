The calibrated sensors are `R72`: they recover the two dose directions independently, and the correction has zero response to the mean state. Its nilpotent structure gives an exact inverse. I’ll install a half-strength, register-compensated pilot. Better rollout effects with factual NLL within the limit would support this pathway; poor fit or unchanged effects would reject it.

```python
import numpy as np
import json
from pathlib import Path
from collections import Counter

def snapshot_ops109():
    saved = []
    for operation in ops():
        item = dict(operation)
        if item["op"] == "rank1":
            v = item["vector"]
            if isinstance(v, str):
                p = Path(v)
                if not p.is_absolute():
                    p = Path("/work") / p
                v = np.load(p)
            item["vector"] = np.asarray(v, dtype=float).copy()
        saved.append(item)
    return saved

def replay_ops109(saved):
    clear_ops()
    for item in saved:
        if item["op"] == "rank1":
            rank1(item["name"], item["vector"].copy(),
                  float(item["alpha"]), side=item.get("side", "out"))
        elif item["op"] == "scale":
            c = item.get("c", item.get("factor", item.get("scale")))
            scale(item["name"], float(c))
        else:
            raise ValueError(item)

baseline_ops109 = snapshot_ops109()
assert len(baseline_ops109) == 1
assert baseline_ops109[0]["name"] == "embed.numeric.weight"

# Independent factorization: each returned factor is applied on the left.
def orthogonal_factors109(Q):
    work = np.asarray(Q, dtype=float).copy()
    n = len(work)
    elimination = []
    for j in range(n):
        v = np.zeros(n)
        v[j:] = work[j:, j]
        v[j] -= 1.0
        length = np.linalg.norm(v)
        if length > 1e-10:
            v /= length
            work -= 2.0 * np.outer(v, v @ work)
            elimination.append((v.copy(), -2.0))
    assert np.max(np.abs(work - np.eye(n))) < 1e-8
    return elimination[::-1]

def matrix_factors109(M):
    U, singular, Vt = np.linalg.svd(M)
    # M = Q P, with P symmetric positive definite.
    Q = U @ Vt
    factors = []
    for j, s in enumerate(singular):
        if abs(s - 1.0) > 1e-12:
            factors.append((Vt[j].copy(), float(s - 1.0)))
    factors.extend(orthogonal_factors109(Q))
    check = np.eye(len(M))
    for v, alpha in factors:
        check = (np.eye(len(M)) + alpha * np.outer(v, v)) @ check
    assert np.max(np.abs(check - M)) < 1e-8
    return factors

def apply_basis_transform109(name, basis, reduced, side):
    factors = matrix_factors109(reduced)
    if side == "in":
        factors = factors[::-1]
    for v, alpha in factors:
        full_v = basis @ v
        rank1(name, full_v.copy(), alpha, side=side)

strengths109 = np.array([0.5, 0.5])
basis109 = np.linalg.qr(np.column_stack((T72, R72)))[0]
shear109 = (T72 * strengths109) @ R72.T
G109 = np.eye(192) + shear109
H109 = np.eye(192) - shear109
assert np.linalg.norm(H109 @ G109 - np.eye(192)) < 1e-10
assert np.linalg.norm(
    basis109 @ (basis109.T @ shear109 @ basis109) @ basis109.T - shear109
) < 1e-10

Gsmall109 = basis109.T @ G109 @ basis109
Hsmall109 = basis109.T @ H109 @ basis109

embedding_names109 = [name for name in NAMES if name.startswith("embed.")]
register_names109 = ["last_numeric.weight", "last_numeric.bias", "last_status.weight"]
encoder_names109 = ["encode.mlp1.0.weight", "encode_action.mlp1.0.weight"]

# Atom/register columns become H times their old values.
for name in embedding_names109 + register_names109:
    shape = SHAPES[name]
    if len(shape) == 1:
        assert shape[0] == 192
        apply_basis_transform109(name, basis109, Hsmall109, "out")
    elif shape[1] == 192:
        apply_basis_transform109(name, basis109, Hsmall109.T, "in")
    elif shape[0] == 192:
        apply_basis_transform109(name, basis109, Hsmall109, "out")
    else:
        raise ValueError((name, shape))

# Encoder inputs are unchanged; value-head registers are unchanged.
for name in encoder_names109 + ["attr_heads.value_bin.0.weight"]:
    apply_basis_transform109(name, basis109, Gsmall109, "in")

candidate_ops109 = snapshot_ops109()

def effective109(saved, name, side):
    result = np.eye(192)
    for item in saved:
        if item["name"] != name or item["op"] != "rank1":
            continue
        if item.get("side", "out") != side:
            continue
        v = item["vector"] / np.linalg.norm(item["vector"])
        F = np.eye(192) + item["alpha"] * np.outer(v, v)
        result = F @ result if side == "out" else result @ F
    return result

errors109 = {}
for name in embedding_names109 + register_names109:
    shape = SHAPES[name]
    side = "out" if len(shape) == 1 or shape[1] != 192 else "in"
    expected = H109 if side == "out" else H109.T
    errors109[name] = np.max(np.abs(effective109(candidate_ops109, name, side) - expected))
for name in encoder_names109 + ["attr_heads.value_bin.0.weight"]:
    errors109[name] = np.max(np.abs(effective109(candidate_ops109, name, "in") - G109))
assert max(errors109.values()) < 1e-8

rng109 = np.random.default_rng(109)
q109, r109 = rng109.normal(size=(2, 192))
print("Maximum actual operation-transform error:", max(errors109.values()))
print("Encoder compensation error:", np.linalg.norm(G109 @ H109 - np.eye(192)))
print("Value-head query/register identity error:",
      np.linalg.norm(G109 @ (q109 + H109 @ r109) - (G109 @ q109 + r109)))
print("Strengths [insulin, NE]:", strengths109)
print("Operation count:", len(candidate_ops109))
print("Operations by tensor:", dict(Counter(x["name"] for x in candidate_ops109)))
print("Probe IDs:", probe_ids)

def json_ops109(saved):
    return [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in item.items()} for item in saved]

Path("/work/query_only_baseline_109.json").write_text(
    json.dumps(json_ops109(baseline_ops109)))
Path("/work/query_only_candidate_109.json").write_text(
    json.dumps(json_ops109(candidate_ops109)))
np.savez("/work/query_only_design_109.npz",
         basis=basis109, G=G109, H=H109, strengths=strengths109,
         targets=T72, sensors=R72)

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 109: compensated query-only pilot installed\n"
            "- Numeric gain 4 retained; query shear strengths insulin=0.5, NE=0.5.\n"
            "- Uses R72 calibrated sensors, not W72. R72.T @ D72.T is identity.\n"
            "- Atom embeddings and reading registers transformed by H=I-S;\n"
            "  first encoder layers and value-head first layer compensated by G=I+S.\n"
            "- Algebra: encoders unchanged; value head sees G*q + original registers.\n"
            "- Independent rank-one factorization verified against saved operation vectors.\n"
            "- Tied code logits and unrelated attribute heads are not preserved;\n"
            "  factual value NLL and recorded-schedule rollouts are the required tests.\n"
            "- Candidate is unvalidated. Forward passes used: 40; 10 remain.\n")
```