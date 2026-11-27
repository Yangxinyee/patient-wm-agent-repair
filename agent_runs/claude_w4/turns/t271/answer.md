Two facts from this run:

1. **Most readings come straight after a drug bundle.** That holds for 1751 of 1931 finger readings and 1489 of 1657 arterial MAP readings. So the state right after a drug bundle is what predicts nearly every reading, which is ideal for an injection there.
2. **`step_full` with default settings is a poor replica.** R² for the new slots is negative and the relative error is about 16%, even though it runs fast (0.4 s) and handles subsets.

My notes say the best‑fitting variant added the age embedding to the memory keys *and* values ("ageKV", R² 0.998). That was probably implemented in `step_var`, not `step_full`. The time‑decayed slots may also have been stored as `htft` rather than `htf`.

I'll inspect `step_var`'s arguments and variant names, then score it, and `step_full` with `htft`, against the exact states. The combination that reaches R² ≥ 0.99 becomes my step replica. If none does, I'll design empirically.

```python
for nm in ['step_var','step_cv','mha_m','build_mem']:
    f=globals()[nm]; co=f.__code__
    print('==',nm,'args',co.co_varnames[:co.co_argcount],'defaults',[d if not isinstance(d,(dict,torch.Tensor,np.ndarray)) else type(d) for d in (f.__defaults__ or [])])
    print('   consts',[c for c in co.co_consts if isinstance(c,(str,int,float))][:40])
for hn_ in ['htf','htft','hprev']:
    try:
        rs=step_full(np.arange(2319),torch.tensor(Ep,dtype=torch.float32),torch.tensor(A(globals()[hn_]),dtype=torch.float32))
        print('step_full with',hn_,'R2 y %.4f out %.4f'%(r2f(A(rs[0]),yt_),r2f(A(rs[1]),ot_)))
    except Exception as ex: print(hn_,'err',repr(ex)[:200])
```