Run 241 listed what is in the kernel, but the output was cut off before /work/inj.py was printed. What survived is useful:
- The 1,500 training stays from run 240 (`S2`), with 21,243 drug rows (`DR`) and 35,250 edited-dose rows (`UR`).
- The policy fits (`POL`, `PIH`, `PHI`).
- The exact-replica functions (`bstep`, `bstep_z`, `zparts`, `rollout`, `make_zfn`/`mk_zfn`) and the IRF arrays.

I conclude that run 240 built training data for a z_post edit. The edit is meant to compute the unexpected part of the dose (dose minus what the policy would give) from ctx and u, and to pass it through the z token. What I don't yet know is the exact row format, and what inj.py does with it.

Next I'll print only the source of inj.py and the structure of one `DR` and one `UR` row, keeping the output short. If inj.py already has the injection construction (the z_post and z_in edit that copies u and adds w times the residual), I'll reuse it. If it doesn't, I'll write that construction next.

```python
src = open('/work/inj.py').read()
print(len(src)); print(src[:6000])
def brief(x):
    if isinstance(x, dict): return {k: (getattr(v,'shape',None) or (v if np.isscalar(v) else type(v).__name__)) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [getattr(v,'shape',None) or (v if np.isscalar(v) else type(v).__name__) for v in x]
    return getattr(x,'shape',None) or x
print('DR0', brief(DR[0])); print('UR0', brief(UR[0]))
print('POL', {k: type(v).__name__ for k,v in POL.items()}, 'PIH', list(PIH.keys())[:4], 'PHI ex', list(PHI.items())[:1])
```