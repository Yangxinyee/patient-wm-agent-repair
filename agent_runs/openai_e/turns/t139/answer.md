The kernel has a local recurrent-state calculation and a reading decoder that I can use without gradients. I’ll measure how different state directions affect MAP and glucose in each slot. Distinct sensitivities would let a closed-form fit protect factual outputs more precisely; nearly identical sensitivities would make this approach unlikely to help.

```python
torch.set_num_threads(8)
from scipy.linalg import qr
_scale120 = np.where(map_mask82, 6.0, 30.0)
_u120, _sv120, _vt120 = np.linalg.svd(J82 / _scale120[:, None], full_matrices=False)
_raw120 = np.column_stack([
    _vt120[:12].T,
    np.asarray(direction82) / np.linalg.norm(direction82),
    np.asarray(direction58) / np.linalg.norm(direction58)
])
Dout120, _r120 = np.linalg.qr(_raw120)
Dout120 = Dout120[:, :14].astype(np.float32)
h0_120 = hnew46.detach().clone()
y0_120, o0_120 = coda46(h0_120)
mean0_120 = next_means45(y0_120, o0_120).detach().numpy()
Jslot120 = np.empty((len(mean0_120), 4, Dout120.shape[1]), dtype=np.float64)
_eps120 = 0.08
for _slot120 in range(4):
    for _dim120 in range(Dout120.shape[1]):
        _pert120 = torch.zeros_like(h0_120)
        _pert120[:, _slot120, :] = _eps120 * torch.from_numpy(Dout120[:, _dim120])
        _hp120, _op120 = coda46(h0_120 + _pert120)
        _hm120, _om120 = coda46(h0_120 - _pert120)
        _mp120 = next_means45(_hp120, _op120).detach().numpy()
        _mm120 = next_means45(_hm120, _om120).detach().numpy()
        Jslot120[:, _slot120, _dim120] = (_mp120 - _mm120) / (2 * _eps120)

print("readouts / MAP readouts:", len(mean0_120), int(np.sum(map_mask82)))
print("output-direction singular values:", _sv120[:14])
print("slot Jacobian RMS by slot:", np.sqrt(np.mean(Jslot120 ** 2, axis=(0, 2))))
print("MAP slot-Jacobian singular values:",
      np.linalg.svd(Jslot120[map_mask82].reshape(np.sum(map_mask82), -1),
                    compute_uv=False)[:12])
print("glucose slot-Jacobian singular values:",
      np.linalg.svd(Jslot120[glucose_mask82].reshape(np.sum(glucose_mask82), -1),
                    compute_uv=False)[:12])
print("local original mean range:", np.quantile(mean0_120, [0, .25, .5, .75, 1]))

print("\n=== GENERALISING THE LOCAL DATASET ===")
print("head_meta columns:", list(head_meta.columns))
print(head_meta.head(2).to_string(index=False))
print("head-sized arrays:", [
    (n, tuple(o.shape)) for n, o in list(globals().items())
    if not n.startswith("_") and isinstance(o, (np.ndarray, torch.Tensor))
    and o.ndim >= 1 and o.shape[0] == len(head_meta)
][:30])
print("make_core_memory globals:", make_core_memory.__code__.co_names)
print("make_core_memory locals:", make_core_memory.__code__.co_varnames)
print("core config:", core_config45)
print("memory types:", {k: (type(v).__name__, len(v),
      [tuple(x.shape) if hasattr(x, "shape") else type(x).__name__ for x in v])
      for k, v in core_memories.items()})
with open("/work/notes.md", "a") as _f120:
    _f120.write("\n## Run 120: per-slot, multi-direction output sensitivities\n"
                "Computed central finite-difference coda/reading sensitivities for 14 recurrent directions in each of four slots "
                "on retained local contexts; no gradients and no weight changes. Preparing a larger output-constrained "
                "closed-form regression if sensitivities provide useful selectivity. Python runs: 120; forwards: 15.\n")
```