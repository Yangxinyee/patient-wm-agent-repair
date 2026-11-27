The sel1 offline result is alarming. map_arterial rows get dNLL +2.05 and predicted means move with sd 10.6. Yet the tool measured only +0.0053 per map_arterial reading on the f3 tuning set. Glucose rows on sel1 actually improve (glucose_finger −0.0044, lab −0.0087).

My leading explanation is that globals used by `gated3` were changed after the edit was saved. Those are the NE gate `Gn2`, `B_n`, `M_n` and `ROn4`, which I was reworking in run 175. If so, `gated3(-1.9,3)` now builds a different edit from the saved one (hash 262704e88f). That would also explain why the zero edit isn't neutral.

To test this, I'll compare the tensors from `gated3(-1.9,3)` with the saved edit (`load`) and with the originals. Then I'll re-run the sel1 evaluation with the saved tensors themselves. If the saved edit gives a small map_arterial cost on sel1, the globals changed and the stored edit is fine.

```python
E_g3 = gated3(-1.9, 3.0)
print(type(E_g3), list(E_g3.keys()) if isinstance(E_g3, dict) else None)
SAVED = {n: np.asarray(load(n), np.float64) for n in edited()}
for n in edited():
    a_ = np.asarray(E_g3[n], np.float64) if n in E_g3 else None
    o_ = np.asarray(load_original(n), np.float64)
    print(n, 'saved-vs-gated3 maxdiff', None if a_ is None else float(np.abs(a_-SAVED[n]).max()),
          '| saved-vs-orig maxdiff', float(np.abs(SAVED[n]-o_).max()),
          '| n changed entries', int((np.abs(SAVED[n]-o_)>0).sum()))
Wd_saved = {n: torch.as_tensor(SAVED[n]) if isinstance(next(iter(E_g3.values())), torch.Tensor) else SAVED[n] for n in edited()}
o_s = head_run(X, RR, Wd_saved)
RRc['dn_s'] = np.asarray(o_s[1], float) - nz_; RRc['dpm_s'] = np.asarray(o_s[0],float) - np.asarray(o_z[0],float)
print('saved edit on sel1: overall dNLL %.4f' % RRc.dn_s.mean())
print(RRc.groupby('code').agg(n=('dn_s','size'), dn=('dn_s','mean'), dpm_sd=('dpm_s','std')).round(4).to_string())
```