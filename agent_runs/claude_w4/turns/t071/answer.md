The row norms tell me which embedding rows were ever trained. Rows with norm 0.28 are untouched initialisations (0.02 × √192 ≈ 0.277), so those indices never occur:
- REVEAL in `kind`;
- `kq` rows 1–3;
- `text` row 1;
- `unit` row 1 and `rate_unit` row 1.

So an insulin atom uses `kind` 0, presumably `ekind` 2 (drug_admin), and one of the trained `unit` rows (2–5) and `rate_unit` rows (2–3). It may also use `kq` row 0 and the single non-padding row of action, route, status or discharged_to.

That leaves under 2,000 combinations. Next I'll evaluate all of them in one batch for the 11 observed insulin rates. Each candidate goes through mlp1, `size(log 2)`, mlp2 and LayerNorm, and I compare it to the probe's 20 exact u_k coordinates. A best combination with error near 1e-3, far below the rest, confirms the insulin atom embedding. If nothing gets below about 0.05, a term or the normalisation is still wrong.

```python
import itertools
for t in ['action','discharged_to','route','status']:
    print(t, load(f'embed.categorical.{t}.weight').norm(dim=1).numpy().round(2), end='; ')
print()
def fbin(x, edges): return 1 + int(np.sum(np.array(edges) < x))
def zsc(x, d): return float(np.clip((x - d['median'])/d['iqr'], -10, 10))
E_ = {k: load(f'embed.{k}.weight') for k in ['code','kind','ekind','kq']}
CAT = {t: load(f'embed.categorical.{t}.weight') for t in ['action','discharged_to','route','status','unit','rate_unit']}
BR, BD = load('embed.bins.rate.weight'), load('embed.bins.dose.weight')
NW, NB = load('embed.numeric.weight'), load('embed.numeric.bias')
def drug_atom_parts(code_name, rate, dose):
    dr, dd = num[(code_name,'rate')], num[(code_name,'dose')]
    numv = torch.tensor([0., 0., zsc(dose, dd), 1., zsc(rate, dr), 1.])
    return BR[fbin(rate, dr['fine_bin_edges'])] + BD[fbin(dose, dd['fine_bin_edges'])] + NW @ numv + NB
rates_t = sorted(tgt); TG = torch.tensor(np.stack([tgt[r] for r in rates_t]))
core_parts = torch.stack([drug_atom_parts('insulin', r, r) for r in rates_t])  # dose = rate*1h
def u_from_atoms(A, n=1):  # A: (...,192) single-atom sets
    h1 = gelu(A @ load('encode_action.mlp1.0.weight').T + load('encode_action.mlp1.0.bias')) @ load('encode_action.mlp1.2.weight').T + load('encode_action.mlp1.2.bias')
    s = h1/np.sqrt(n) + load('encode_action.size.weight')[:,0]*np.log1p(n) + load('encode_action.size.bias')
    h2 = gelu(s @ load('encode_action.mlp2.0.weight').T + load('encode_action.mlp2.0.bias')) @ load('encode_action.mlp2.2.weight').T + load('encode_action.mlp2.2.bias')
    return LNf(h2, load('encode_action.norm.weight'), load('encode_action.norm.bias'))
results = []
for ek, un, ru, kqi, a_, d_, r_, s_ in itertools.product([1,2,3,4],[0,2,3,4,5],[0,2,3],[None,0],[0,1],[0,1],[0,1],[0,1]):
    base = E_['code'][8] + E_['kind'][0] + E_['ekind'][ek] + CAT['unit'][un] + CAT['rate_unit'][ru] + CAT['action'][a_] + CAT['discharged_to'][d_] + CAT['route'][r_] + CAT['status'][s_]
    if kqi is not None: base = base + E_['kq'][kqi]
    uu_ = u_from_atoms(core_parts + base)[:, u_c]
    results.append((float((uu_ - TG).abs().max()), ek, un, ru, kqi, a_, d_, r_, s_))
results.sort(key=lambda t: t[0])
for r_ in results[:6]: print(np.round(r_[0],4), r_[1:])
```