The run stopped before collecting anything. The u-consistency check assumed the first sampled stay had an infusion, and that stay apparently has none, so `next()` raised. Stays without drugs are possible in this ICU.

I'll make the check search the sampled stays for the first drug bundle. I'll also print a couple of bundle times to confirm they are in hours like BUN; the replica's step sizes depend on that. Then I run the same 300-stay collection.

Confirmation would be a u-consistency of about 1e-12, hourly times like 0.0833 and 1.0, and around 30k base rows plus the edited branches.

```python
def first_drug(Bs):
    for b in Bs:
        if any(a['act'] and a['kind']==0 for a in b[1]): return b
    return None
for s in samp:
    B_t = build_bundles(GB[s]); raw0 = first_drug(B_t)
    if raw0 is not None: break
print('times sample', [round(b[0], 4) for b in B_t[:5]])
ra = [a for a in raw0[1] if a['act'] and a['kind']==0]
print('u consistency', float((set_enc(Wc, [fix_atom2(a) for a in ra], 'encode_action') - set_enc(Wc, split_fix(raw0[1])[1], 'encode_action')).abs().max()))
t0c = time.time(); DEC = []; nst = 0
for i_, s in enumerate(samp):
    DEC += collect_stay(Wc, build_bundles(GB[s]), s); nst += 1
    if time.time()-t0c > 700: print('stopped at', i_); break
print('stays', nst, 'rows', len(DEC), 'time %.0fs' % (time.time()-t0c), 'nan pihat rows', sum(1 for r in DEC if np.isnan(r[3]) or np.isnan(r[4])))
```