Slot-major is the right layout: the stored `out` equals out_norm(mean of slots) exactly. So the saved states are consistent and the error is inside the head.

Readings with no previous value are also off, so the register is not the main problem. The weakest assumption is that every reading uses the same `type_emb` index. Types may differ by code or by measurement kind.

This run searches the best type index separately for each of the four events. If the best type differs by event and drops each event's MAE to well under 1, that was the bug. If every event still prefers type 1 with large errors, the problem lies elsewhere (gap units or activations).

```python
R,lb=regs2(F,'B',1.0); Vm=CTX+L(zmu,'z_head')
for ev in CID:
    mk=(F.event==ev).values; res_=[]
    for T in range(15):
        p=head_p(F[mk],Vm[mk],T,R[mk],lb[mk],code_ids[mk]); pm_=(p[:,1:]*cen[mk]).sum(1); res_.append(np.abs(pm_-tool[mk]).mean())
    o_=np.argsort(res_); print(ev, "best T:",[(int(t),round(res_[t],3)) for t in o_[:3]])
```