The e encoder is now exact for every bundle (maximum error 2.6e-6).

The u error of 1.88 appears only on bundles with no drug start, and it is constant there. So the true model's u for an empty action set is not zero, as the spec suggests. It is most likely the empty-set encoding of the action encoder (the size term passed through mlp2 and the norm), which is presumably what I used in run 105.

I'll test that. If both e and u are then exact everywhere, I can verify the full core step against the tool's states.

```python
u_empty=set_enc(W0,[],'encode_action')
errU2=[]
for i,r in enumerate(recs):
    atoms=BUN[r[0]][r[1]][1]
    has=any(a['act'] and a['kind']!=2 for a in atoms)
    if has: e,u=enc_eu(atoms)
    else: u=u_empty
    errU2.append(np.abs(Pu@u.detach().numpy()-r[5]).max())
errU2=np.array(errU2); print('u (empty-set enc): median %.2e max %.2e frac>1e-4 %.4f'%(np.median(errU2),errU2.max(),(errU2>1e-4).mean()))
with open('/work/notes.md','a') as f:
    f.write("- ENCODER FIXED (run 133): finger unit=0, lab unit=5, MAP art/cuff unit=2; drug END atoms -> e encoder, BARE (no num/bins/units) => M holds NO dose info.\n"
            "  e exact (2.6e-6) all bundles. u: action WORLD atoms -> encode_action.\n")
```