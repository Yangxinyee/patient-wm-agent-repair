`build_v3` is now fully understood from its bytecode:
- It adds three rank-1 terms to the head-1 value rows of the first prelude attention (`kv.weight` rows 240:288). Each term pairs a 48-dim value direction (aI3, aN3, aP3) with a readout of the normalised action token: `tI` for the insulin rate, `fN` for the NE rate and `pI` for insulin presence.
- It then changes the head-1 output columns (`out.weight[:, 48:96]`) so that each new value direction maps onto a chosen target in the 192-dim slot residual. The targets look like gI·dI, gN·dN and an sP-scaled `qd`.

So v3 adds gI·rate·dI to the slot residual for insulin, and similarly for NE. As written it injects the rate, not the surprise. The presence term only subtracts a constant, not a state-dependent expected dose, so factual fit may suffer.

Next I'll rebuild v3 on the CPU and check four things:
1. Only those rows and columns change.
2. The edited output maps aI3→dI, aN3→dN and aP3→the sP·qd target.
3. The three value directions are orthonormal.
4. If the Jacobian `Jm` and the glucose/MAP slot directions (`nuG`/`nuM`) are still in memory, dI and dN produce new-slot shifts along the intended glucose and MAP directions.

If all of this holds, I'll forward-test v3 for dose response before building the state-dependent subtraction.

```python
kv_o = load_original('core.prelude.0.attn.kv.weight').numpy(); wo_o = load_original('core.prelude.0.attn.out.weight').numpy()
print('kv',kv_o.shape,'wo',wo_o.shape)
for (gI_,gN_,sP_) in [(1.,1.,1.),(2.,1.,0.5)]:
    kv_n, wo_n = build_v3(gI_,gN_,sP_)
    kv_n=np.asarray(kv_n); wo_n=np.asarray(wo_n)
    dkv=kv_n-kv_o; dwo=wo_n-wo_o
    print(f'g=({gI_},{gN_},{sP_}) kv rows changed', np.where(np.abs(dkv).sum(1)>0)[0][[0,-1]], 'wo cols changed', np.where(np.abs(dwo).sum(0)>0)[0][[0,-1]])
    W1n=wo_n[:,48:96]
    for nm,a_,d_ in [('I',aI3,dI),('N',aN3,dN),('P',aP3,qd)]:
        v=W1n@a_
        print(f'  {nm}: cos(W1n a, d)={v@d_/np.linalg.norm(v)/np.linalg.norm(d_):+.4f}  |W1n a|/|d|={np.linalg.norm(v)/np.linalg.norm(d_):.4f}')
A3=np.stack([aI3,aN3,aP3]); print('gram a:',np.round(A3@A3.T,4).tolist())
cs=lambda u,v: float(u@v/np.linalg.norm(u)/np.linalg.norm(v))
print('cos qd,dI %.3f qd,dN %.3f dI,dN %.3f'%(cs(qd,dI),cs(qd,dN),cs(dI,dN)))
print('BB3 norms', {k:round(float(np.linalg.norm(v)),3) for k,v in BB3.items()})
print([n for n in NAMES if 'prelude' in n])
for nm in ['nuG','nuM','Jm','J','Jm_']:
    print(nm, nm in globals(), getattr(globals().get(nm),'shape',None))
if 'nuG' in globals() and 'Jm' in globals():
    yI=(Jm@dI).reshape(4,192); yN=(Jm@dN).reshape(4,192)
    print('Jm@dI vs -nuG cos per slot', [round(cs(yI[s],-nuG[s]),3) for s in range(4)], 'scale', [round(float(yI[s]@(-nuG[s])/(nuG[s]@nuG[s])),3) for s in range(4)])
    print('Jm@dN vs nuM cos per slot', [round(cs(yN[s],nuM[s]),3) for s in range(4)], 'scale', [round(float(yN[s]@(nuM[s])/(nuM[s]@nuM[s])),3) for s in range(4)])
```