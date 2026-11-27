END atoms now match exactly (2e-7). An END atom is code + END kind + drug-admin event kind + kq row 0 + numeric bias, with no units and no rate or dose. It goes into the measurement-side vector e_k. So the doubled END atoms in a dose edit carry no dose information. Only start atoms move the action vector u_k.

Next I'll put everything into one general bundle encoder. The admission records are the only untested atoms. I'll assume they are code + WORLD/END kind + visit or visit-detail event kind + kq 0 + numeric bias. I'll check z on every bundle except each stay's first, all 100 stays. A median error around 1e-7 with no outliers would confirm the whole embedding stage. Outliers would be the admission atoms.

```python
def visit_atom(ev, kind, ekind=None, kq=0):
    ek = {'hospital_admission':3, 'icu_admission':4}[ev] if ekind is None else ekind
    kid = 0 if kind=='start' else 2
    return Wt['embed.code.weight'][code_id[ev]] + Wt['embed.kind.weight'][kid] + Wt['embed.ekind.weight'][ek] + Wt['embed.kq.weight'][kq] + Wt['embed.numeric.bias']
def bundle_vecs(evs):
    e_atoms, u_atoms = [], []
    for ev, kd, r_ in evs:
        if ev in ('insulin','norepinephrine'):
            if kd=='start': u_atoms.append(drug_atom(ev, 'start', r_.rate, r_.dose, *CFG[ev]))
            else: e_atoms.append(drug_end(ev))
        elif ev in ('hospital_admission','icu_admission'):
            e_atoms.append(visit_atom(ev, kd))
        else:
            e_atoms.append(meas_atom(ev, r_.value))
    e_ = encode(e_atoms, 'encode') if e_atoms else e_empty
    u_ = encode(u_atoms, 'encode_action') if u_atoms else u_empty
    return e_, u_
# per stay bundle lists for all 100 stays (in states order)
BUN = {}
row = 0; ROWS = {}
for s_ in stays_u:
    bb = build_bundles(tu[tu.stay_id==s_]); keys = sorted(bb.keys())
    BUN[s_] = [(k_, bb[k_]) for k_ in keys]; ROWS[s_] = row; row += len(keys)
errs_all = []
for s_ in stays_u[:40]:
    r0_ = ROWS[s_]
    for i_ in range(1, len(BUN[s_])):
        k_, evs = BUN[s_][i_]; dt = (k_-BUN[s_][i_-1][0])*60.0
        e_, u_ = bundle_vecs(evs)
        rp = r0_+i_-1
        out_p = Xs[rp, :192]; h_p = Xs[rp, 192:960].reshape(4,192)
        dec = torch.exp(-dt*torch.exp(Wt['core.flow.a']))
        ctx = out_p + ln((h_p*dec).mean(0), 'core.out_norm') + mlp(torch.tensor([math.log(dt)/10]), 'gap')
        zp = mlp(torch.cat([ctx, e_, u_]), 'z_post')[:32]
        errs_all.append(((zp - Xs[r0_+i_, 960:992]).abs().max().item(), s_, i_, '+'.join(sorted(set(e2+':'+k2 for e2,k2,_ in evs)))))
E_ = pd.DataFrame(errs_all, columns=['err','stay','i','what'])
print('median %.2e  max %.2e  n %d' % (E_.err.median(), E_.err.max(), len(E_)))
print(E_[E_.err>1e-4].what.value_counts().head(10))
```