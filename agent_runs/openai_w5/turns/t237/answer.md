The private normalization parameters reveal that the intended channel pairs are cross-paired; my first diagnostic compared the wrong coordinates. No weights changed. I’ll test the correct pairs for single and simultaneous infusions. Linear, drug-specific signals would confirm the encoder design; cross-talk or saturation would refute it.

```python
import torch.nn.functional as F

private_pairs = [(21, 61), (15, 66)]  # equal-normalization pairs
ordinary_idx = torch.tensor([i for i in range(192) if i not in priv],
                            dtype=torch.long)

print("=== CORRECTLY PAIRED ENCODER SIGNALS ===")
encoder_audit = {}
for _drug, _rates in [
    ("insulin", np.array([0, .5, 1, 2, 4, 8, 12, 16, 24, 32, 48], np.float32)),
    ("norepinephrine", np.array([0, .01, .05, .1, .2, .4, .8, 1, 1.5, 2, 3], np.float32))
]:
    _atoms = drug_atom205(_drug, _rates, W)
    _u = set205(_atoms[:, None, :], "encode_action", W)
    _signals = torch.stack([_u[:, a] - _u[:, b] for a, b in private_pairs], 1)
    encoder_audit[_drug] = (_rates, _u, _signals)
    print(_drug, np.column_stack([_rates, _signals.numpy()]))

print("\n=== SIMULTANEOUS INFUSIONS ===")
_ir = np.array([0, 1, 4, 8, 16], np.float32)
_nr = np.array([0, .05, .2, .4, 1], np.float32)
_both = torch.stack([drug_atom205("insulin", _ir, W),
                     drug_atom205("norepinephrine", _nr, W)], dim=1)
_ub = set205(_both, "encode_action", W)
print(np.column_stack([_ir, _nr,
    (_ub[:, 21] - _ub[:, 61]).numpy(),
    (_ub[:, 15] - _ub[:, 66]).numpy()]))

print("\n=== RESERVED HEAD INPUTS ===")
for _j in reserved:
    print(_j,
          "private", W["attr_heads.value_bin.0.weight"][_j, priv].numpy(),
          "bias", round(float(W["attr_heads.value_bin.0.bias"][_j]), 6),
          "ordinary_norm", round(float(torch.linalg.vector_norm(
              W["attr_heads.value_bin.0.weight"][_j, ordinary_idx])), 6))
print("\n=== RESERVED HEAD OUTPUTS ===")
print(W["attr_heads.value_bin.2.weight"][:, reserved].numpy())

print("\n=== PRELUDE TENSOR SHAPES ===")
print([(n, tuple(W[n].shape)) for n in NAMES if n.startswith("core.prelude.0")])
print("\n=== SAVED STATE SCHEMA ===")
state_file = np.load("/work/fwd_15/states_edited.npz")
print([(k, state_file[k].shape, str(state_file[k].dtype)) for k in state_file.files])
print("\n=== FORWARD FILES ===")
for _folder in ["fwd_17", "fwd_18", "fwd_19"]:
    print(_folder, [p.name for p in Path("/work", _folder).iterdir()])
```