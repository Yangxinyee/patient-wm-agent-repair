The full architecture text came through in the last output, and I will not print it again. Two points matter most for the edit:
- The value head reads dose information only through the state (`view`). Its "last" register covers only the same code's previous value.
- In the rollout the doses are fixed and the readings are sampled, so any edit acts partly through the model's own sampled glucose.

The kernel holds an earlier replica of the per-bundle state update (`step`, `bvd`, `one_step_err`). There is no source for it and no implementation of the heads.

Next I will record the current state in the notebook. Then I will test whether the existing replica reproduces the tool's states, by calling `one_step_err` on a few stays and bundles and printing only summary statistics.
- **Replica works:** errors around 1e-6 mean I can use it to compute states under edited doses locally.
- **Replica does not work:** large errors mean I must rebuild it, this time saved to a file.

```python
add3 = """
## Session 4 (runs 90+): state of kernel
- Kernel holds an earlier replica of the per-bundle state update (fns: build_bundles, bundle_vecs/bvd, encode, step, mha,
  tlayer, block, ffn, one_step_err; dicts W/Wt of weights; BUN = bundles of the 100 sel1 tuning stays; Xt = fwd_3 states).
  Source is NOT recoverable (defined in kernel). o_chk=8.9e-7 suggests out matched. No head implementation yet.
- fwd_3 = states, tuning sel1 100 stays, 5116 bundles (x: out | 4 h slots | z post mean).
- KEEP OUTPUTS SHORT (content filter refused long raw listings). Save new code to /work/*.py files.
"""
with open('/work/notes.md', 'a') as fh:
    fh.write(add3)
print(inspect.signature(one_step_err), inspect.signature(bvd), inspect.signature(step))
sts = list(BUN.keys())[:3]
print('BUN key sample', sts, type(BUN[sts[0]]).__name__, len(BUN[sts[0]]))
res_chk = []
for s_ in sts:
    nb_ = int((sid_ == s_).sum()) if hasattr(sid_, '__len__') else -1
    for k_ in [1, 2, 5, 10, nb_ - 1]:
        try:
            r_ = one_step_err(s_, k_)
            res_chk.append((s_, k_, r_ if not hasattr(r_, '__len__') else str(r_)[:80]))
        except Exception as ex_:
            res_chk.append((s_, k_, 'ERR ' + str(ex_)[:80]))
for r_ in res_chk:
    print(r_)
```