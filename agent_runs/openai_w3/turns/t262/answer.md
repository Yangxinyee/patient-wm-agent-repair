The ordinary raw state has RMS about 61, so the current dose transport is not a small perturbation. The last calculation also stopped on a NumPy/Torch type mismatch; no weights changed. I will fix that and compare after-FFN responses at the next reading’s delay. Smaller, symmetric responses would support replacing the present transport.

```python
import time
torch.set_num_threads(8)

r234_Odiff = (as_numpy(candidate206["core.prelude.0.attn.out.weight"])
             - as_numpy(load_original("core.prelude.0.attn.out.weight")))
r234_map = np.linalg.lstsq(directions186, r234_Odiff, rcond=None)[0]
r234_residual = r234_Odiff - directions186 @ r234_map
print("attention total/unexplained norms",
      np.linalg.norm(r234_Odiff), np.linalg.norm(r234_residual))
print("replay indices", i169, "current gaps", as_numpy(dt169))
for n in ["attr_static105", "transition97", "centres105", "codes97"]:
    v = globals()[n]
    print(n, getattr(v, "shape", np.asarray(v).shape))
print("register helper dependencies", isolated_record.__code__.co_names)
print("four-sensor targets", T4_228)

def response_raw234(injections, next_seconds=3300.0, individual=False):
    inj = tensor153(injections)
    m = len(inj)
    raw = r233_raw.repeat(m, 1, 1) + inj.repeat_interleave(n186, dim=0)[:, None, :]
    ep = F.layer_norm(raw, (192,),
                      load_original("core.prelude_norm.weight"),
                      load_original("core.prelude_norm.bias"))
    hp, op = state_update82(
        hf169.repeat(m, 1, 1), ep, mem169.repeat(m, 1, 1),
        mask169.repeat(m, 1), age_emb169.repeat(m, 1, 1), 0)
    gap = torch.full((m*n186,), float(next_seconds))
    pred = arr183(phys171(view171(hp, op, gap))).reshape(m, n186, 4)
    return pred if individual else pred.mean(axis=1)

r234_t0 = time.time()
r234_eps = 4.0
r234_J = np.zeros((4, 192))
r234_basis = np.eye(192, dtype=np.float32)
for start in range(0, 192, 16):
    b = r234_basis[start:start+16] * r234_eps
    r = response_raw234(np.concatenate([b, -b]))
    r234_J[:, start:start+len(b)] = ((r[:len(b)]-r[len(b):])/(2*r234_eps)).T
print("raw-J singular values", np.linalg.svd(r234_J, compute_uv=False),
      "seconds", round(time.time()-r234_t0, 1))

r234_baseline = response_raw234(np.zeros((1, 192)))[0]
r234_directions = {}
for ridge in [1e-6, 1e-5, 1e-4]:
    d = r234_J.T @ np.linalg.solve(
        r234_J @ r234_J.T + ridge*np.eye(4), T4_228)
    r234_directions[ridge] = d
    tests = np.stack([-d[:, 0], d[:, 0], -d[:, 1], d[:, 1]])
    print("ridge", ridge, "norms", np.linalg.norm(d, axis=0),
          "signed responses", np.round(response_raw234(tests)-r234_baseline, 3))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 234; forwards 16. Run 233 ended on NumPy/Torch subtraction; no edit changes. Recovered attention decomposition and tested after-FFN transport at next-reading delay by finite differences, without gradients or weight optimization. Candidates are diagnostics only.\n")
```