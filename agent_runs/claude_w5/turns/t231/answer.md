My local per-reading NLL does not reproduce the tool's: correlation only 0.29–0.41 and an offset of about 0.4. So my ΔNLL estimate (+0.0023) may not reflect the judged metric. My notes say an earlier local version matched the tool (1.434–1.439 versus 1.4364). Two candidate errors:
- the target bin: I used H's 'target' (161.26) where the tool probably uses the recorded value (162.0);
- the z-prior's second half: my notes say it is a log-std, while probs_from_v0 defaults to 'logvar'.

The float32 round trip is harmless (2e-4 relative). The edit is saved in four tensors.

Next I recompute the local NLL under these variants and keep the one whose per-row values track the tool closely. The right setup should give a mean near 1.436 and a high per-row correlation. Then I redo the edited-versus-original comparison with it.

```python
tgt_obs = torch.tensor([int(np.asarray(fbv(H[i]['code'], float(H[i]['obs']))).item()) for i in IDX])
tool_arr = np.array([f3k.loc[(int(H[i]['sid']), float(H[i]['hour']), H[i]['code']), 'nll'] for i in IDX], dtype=float)
def nll_var(V, tgt, lv, nz=4, seed=0):
    p = torch.as_tensor(probs_from_v0(V, fixed[IDX], TRr[IDX], nz=nz, lv=lv, seed=seed))
    return -torch.log(p[torch.arange(len(IDX)), tgt].clamp_min(1e-12)).numpy()
for tg_name, tg in (('target', tgt_), ('obs', tgt_obs)):
    for lv in ('logvar', 'logstd', 'std'):
        try:
            nn_ = nll_var(VO, tg, lv)
            print(tg_name, lv, 'mean', nn_.mean().round(4), 'corr w tool', np.corrcoef(nn_, tool_arr)[0,1].round(4), 'mean abs diff', np.abs(nn_-tool_arr).mean().round(4))
        except Exception as ex:
            print(tg_name, lv, 'error', str(ex)[:80])
```