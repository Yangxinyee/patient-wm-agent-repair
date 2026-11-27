The tool's insulin and norepinephrine flags mean "a START atom is in the bundle", and they agree exactly with my START atoms. The earlier 0% was a pandas aggregation slip. So the bundling is confirmed, including END atoms at the end of each infusion and the stay-end bundle.

Next I'll write a clean vectorised encoder for atoms, sets, e_k and u_k, plus the z posterior, and test conventions against the tool's z posterior mean. The test uses bundles whose previous state is known from fwd_3, so ctx can be built from the reported out and h. The grid covers:
- activation: ReLU or GELU;
- the gap-input scaling;
- unit-index orderings.

Errors are reported separately for measurement-only and action-only bundles. An error near 1e-4 on measurement-only bundles would lock down the measurement encoder and ctx. Action bundles would then test the drug encoding on their own.

```python
NUMV = {(n_["code"], n_["attribute"]): n_ for n_ in VOCAB["numeric"]}
def zs_(code, attr, x):
    n_ = NUMV[(code, attr)]; return np.clip((x - n_["median"]) / n_["iqr"], -10, 10)
def fb_(code, attr, x):
    e_ = np.array(NUMV[(code, attr)]["fine_bin_edges"]); x = np.asarray(x, float)
    return np.where(np.isnan(x), 0, 1 + (e_[None, :] < x[:, None]).sum(1))
EKc = {c["name"]: {"measurement":1,"drug_admin":2,"visit":3,"visit_detail":4}.get(c.get("event_kind"),0) for c in VOCAB["codes"]}
def act_(x, kind): 
    return torch.relu(x) if kind=='relu' else torch.nn.functional.gelu(x)
def lin_(x, pre): 
    b = pre + '.bias'
    return x @ Wt(pre + '.weight').T + (Wt(b) if b in SHAPES else 0)
def mlp_(x, pre, kind): return lin_(act_(lin_(x, pre + '.0'), kind), pre + '.2')
def atoms_embed(A_, cfg):
    n = len(A_); V = torch.zeros(n, 192, dtype=torch.float64)
    code = A_.code.values; kind = A_.kind.values
    V += Wt('embed.code.weight')[[CID[c] for c in code]] + Wt('embed.kind.weight')[kind] + Wt('embed.ekind.weight')[[EKc[c] for c in code]] + Wt('embed.kq.weight')[cfg.get('kq',0)]
    num = np.zeros((n, 6)); bins_v = np.zeros(n, int); bins_d = np.zeros(n, int); bins_r = np.zeros(n, int)
    unit = np.zeros(n, int); runit = np.zeros(n, int)
    for c in set(code):
        ix_ = np.where(code == c)[0]
        if c in ('glucose_finger','glucose_lab','map_arterial','map_cuff'):
            v_ = A_.value.values[ix_]; num[ix_, 0] = zs_(c,'value',v_); num[ix_, 1] = 1; bins_v[ix_] = fb_(c,'value',v_); unit[ix_] = cfg['unit'][c]
        elif c in ('insulin','norepinephrine'):
            unit[ix_] = cfg['unit'][c]; runit[ix_] = cfg['runit'][c]
            ok_ = (kind[ix_] == 0) | cfg['end_attrs']; j = ix_[ok_]
            d_ = A_.dose.values[j]; r_ = A_.rate.values[j]
            num[j, 2] = zs_(c,'dose',d_); num[j, 3] = 1; num[j, 4] = zs_(c,'rate',r_); num[j, 5] = 1
            bins_d[j] = fb_(c,'dose',d_); bins_r[j] = fb_(c,'rate',r_)
    V += Wt('embed.bins.value.weight')[bins_v] * torch.tensor(bins_v>0)[:,None] + Wt('embed.bins.dose.weight')[bins_d] * torch.tensor(bins_d>0)[:,None] + Wt('embed.bins.rate.weight')[bins_r] * torch.tensor(bins_r>0)[:,None]
    V += Wt('embed.categorical.unit.weight')[unit] * torch.tensor(unit>0)[:,None] + Wt('embed.categorical.rate_unit.weight')[runit] * torch.tensor(runit>0)[:,None]
    V += torch.tensor(num) @ Wt('embed.numeric.weight').T + Wt('embed.numeric.bias')
    return V
print('ok defined')
```