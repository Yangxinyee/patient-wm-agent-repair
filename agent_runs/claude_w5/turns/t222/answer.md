E has many extremely quiet directions. Apart from the LayerNorm's blocked 1/γ direction, the smallest eigenvalues are about 1.3–2.7e-7, against a largest of 132. Along them q·E varies with sd of only about 4e-4 and mean about 1e-7, and the prelude carrier can fully reach them (reach = 1.0). So a carrier of about 0.1 in E along such a direction would be read by a rank-1 b_in edit with a signal-to-noise ratio of a few hundred. Since quiet directions are effectively unused by the model, writing there should barely disturb anything else.

The risk is that a direction quiet on factual data is not quiet once doses are doubled or halved. Next I add E from all x2/x½ drug bundles to the moment matrix and pick the two quietest reachable directions (q_I, q_N). Then I compute the mean belief directions v̄_I and v̄_N and check how consistent they are across bundles. A good outcome is q·E sd still around 1e-3 or less on scaled bundles, and a reasonably high median cosine of per-bundle v with v̄.

```python
E_sc = []
for r in DB:
    d = steps_all[r['sid']][r['k']]
    for key, uu in r['u'].items():
        if key == 1.0: continue
        code = 'insulin' if key[0] == 'I' else 'norepinephrine'
        e2, u2 = enc2(scale_atoms(bundles_B[r['sid']][r['k']][1], code, key[1]))
        d2 = dict(d); d2['e'] = e2; d2['u'] = u2; R = {}; mystep(d2, R=R); E_sc.append(R['E'].numpy())
E_sc = np.concatenate(E_sc)
S_c = (Eall.T @ Eall + E_sc.T @ E_sc) / (len(Eall) + len(E_sc))
ev_c, V_c = np.linalg.eigh(S_c)
qs = []
for i in range(12):
    q = V_c[:, i]; qg = q * gam_pn
    if np.linalg.norm(qg - qg.mean()) / np.linalg.norm(qg) < 0.9: continue
    qs.append(q)
    if len(qs) == 2: break
q_I, q_N = qs
for nm, q in (('q_I', q_I), ('q_N', q_N)):
    print(nm, 'sd fact', f"{np.std(Eall@q):.2e}", 'sd scaled', f"{np.std(E_sc@q):.2e}", 'max|qE| fact', f"{np.abs(Eall@q).max():.2e}", 'scaled', f"{np.abs(E_sc@q).max():.2e}")
print('n eig < 1e-6:', int((ev_c < 1e-6).sum()), '< 1e-5:', int((ev_c < 1e-5).sum()))
VIa = np.array([v.reshape(4,192) for v in VI if v is not None]); VNa = np.array([v.reshape(4,192) for v in VN if v is not None])
vbar_I = VIa.mean((0,1)); vbar_N = VNa.mean((0,1))
def cosv(a, b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
cI_list = [cosv(v.mean(0), vbar_I) for v in VIa]; cN_list = [cosv(v.mean(0), vbar_N) for v in VNa]
print('v_I vs vbar cos median', round(np.median(cI_list),3), 'p10', round(np.percentile(cI_list,10),3), '| slot-consistency', round(np.median([cosv(v[0], v[3]) for v in VIa]),4))
print('v_N vs vbar cos median', round(np.median(cN_list),3), 'p10', round(np.percentile(cN_list,10),3), '| |vbar_I|', round(np.linalg.norm(vbar_I),3), '|vbar_N|', round(np.linalg.norm(vbar_N),3), 'cos(vI,vN)', round(cosv(vbar_I, vbar_N),3))
print('fixed', desc(fixed), 'TRr', desc(TRr), 'cen', desc(cen), '| view0', str(inspect.signature(view0)), '| probs_from_v0', str(inspect.signature(probs_from_v0))[:120])
```