The last request was a "states" pass with both models on 100 tuning stays. It finished and wrote its files. I have not analysed it yet, because my previous reply was withheld. My notebook also stops before fwd_14, so I don't know what that pass showed for the current edit (v9: channel injection lowered from c=6 to 0.4, b_in gain raised 15x so the intended response is unchanged).

My hypothesis is that most of v7's factual-loss damage did not depend on the gain setting. I think it came from the large channel injection disturbing the prelude, and that v9 removes it.

Next I will spend one Python run collecting everything as aggregates only (no raw rows, no stay ids):
- fwd_14: per-event NLL change, size of the prediction shift, correlation of that shift with the residuals, and how much of the damage sits in the top 1% of readings.
- fwd_10 and fwd_11: the added dose effect by hour, comparing news-direction targets with predicted-mean targets, to see whether injected effects persist and build up.
- fwd_15: how large the state perturbation is at drug bundles and at increasing distances from the last drug bundle.

**What would confirm or refute it:**
- **Confirm:** state perturbations near zero far from drug bundles, and a factual NLL change close to 0.
- **Refute:** large perturbations far from any drug bundle, or a heavy-tailed NLL change, would mean another leak remains.

```python
import os, json, inspect, numpy as np, pandas as pd
S = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); S.append(s)

info = {k: sorted(os.listdir(f'/work/fwd_{k}')) for k in range(8, 16) if os.path.isdir(f'/work/fwd_{k}')}
print('FILES', info)

def fcols(df):
    c = list(df.columns); ev = None
    for x in c:
        if df[x].dtype == object:
            s = df[x].astype(str)
            if s.str.contains('glucose|map', regex=True).any(): ev = x; break
    nll = [x for x in c if 'nll' in x.lower()]
    pm = [x for x in c if ('mean' in x.lower() or 'pred' in x.lower()) and 'nll' not in x.lower()]
    val = [x for x in c if x.lower() in ('value','reading','y','obs','observed','actual','val')]
    return c, ev, nll, pm, val

def fact_stats(k, tag):
    d = f'/work/fwd_{k}'
    fo = pd.read_csv(f'{d}/factual_original.csv'); fe = pd.read_csv(f'{d}/factual_edited.csv')
    c, ev, nll, pm, val = fcols(fo)
    print('fact cols', c, '| ev', ev, 'nll', nll, 'pm', pm, 'val', val)
    nl, p = nll[0], pm[0]; v = val[0] if val else None
    P(f'[{tag} fwd_{k} factual] n {len(fo)}/{len(fe)} NLL orig {fo[nl].mean():.4f} edit {fe[nl].mean():.4f} d {fe[nl].mean()-fo[nl].mean():+.4f}')
    for e_, g in fo.groupby(ev):
        idx = g.index; dn = (fe.loc[idx, nl]-fo.loc[idx, nl]).values; dp = (fe.loc[idx, p]-fo.loc[idx, p]).values
        a = np.abs(dn); o = np.argsort(-a); top = max(1, len(a)//100)
        line = (f'  {e_}: n={len(idx)} dNLL {dn.mean():+.4f} med|dn| {np.median(a):.4f} top1%part {dn[o[:top]].sum()/len(a):+.4f}'
                f' dp_rms {np.sqrt((dp**2).mean()):.2f} med|dp| {np.median(np.abs(dp)):.3f}')
        if v is not None:
            r = (fo.loc[idx, v]-fo.loc[idx, p]).values
            line += f' res_sd {r.std():.2f} corr(dp,res) {np.corrcoef(dp, r)[0,1]:+.3f}'
        P(line)

def dose_stats(k, tag):
    d = f'/work/fwd_{k}'
    do = pd.read_csv(f'{d}/dose_edit_original.csv'); de = pd.read_csv(f'{d}/dose_edit_edited.csv')
    c = list(do.columns)
    ed = next((x for x in c if do[x].dtype == object and do[x].astype(str).str.contains('x2|insulin|nore', regex=True).any()), None)
    sh = [x for x in c if 'shift' in x.lower()]; hr = next((x for x in c if 'hour' in x.lower()), None)
    print('dose cols', c, '| edit', ed, 'shift', sh, 'hour', hr)
    if ed is not None:
        s = sh[0]; mo = do.groupby(ed)[s].mean(); me = de.groupby(ed)[s].mean()
        P(f'[{tag} fwd_{k} dose] ' + ' | '.join(f'{e_}: o {mo[e_]:+.3f} e {me[e_]:+.3f}' for e_ in mo.index))
        if hr:
            po = do.groupby([ed, hr])[s].mean(); pe = de.groupby([ed, hr])[s].mean()
            for e_ in mo.index:
                P(f'   added/h {e_}: ' + ' '.join(f'{x:+.2f}' for x in (pe[e_]-po[e_]).values))
    else:
        for s in sh:
            P(f'[{tag} fwd_{k} dose] {s}: o {do[s].mean():+.3f} e {de[s].mean():+.3f}')
            if hr: P('   added/h: ' + ' '.join(f'{x:+.2f}' for x in (de.groupby(hr)[s].mean()-do.groupby(hr)[s].mean()).values))

def state_stats(k, tag):
    so = np.load(f'/work/fwd_{k}/states_original.npz'); se = np.load(f'/work/fwd_{k}/states_edited.npz')
    xo = so['x']; xe = se['x']; dx = xe - xo
    hi = so['has_insulin'].astype(bool); hn = so['has_norepinephrine'].astype(bool); sid = so['stay_id']
    lag = np.full(len(sid), 99); prev = None; last = None
    for i in range(len(sid)):
        if sid[i] != prev: prev = sid[i]; last = None
        if hi[i] or hn[i]: last = i
        if last is not None: lag[i] = min(i-last, 50)
    P(f'[{tag} fwd_{k} states] rows {len(sid)} ins {hi.sum()} ne {hn.sum()} (median|dx| / p99)')
    groups = [('ins', hi), ('ne', hn & ~hi), ('lag1', lag==1), ('lag2', lag==2), ('lag3-5', (lag>=3)&(lag<=5)),
              ('lag6+', (lag>=6)&(lag<99)), ('preDrug', lag==99)]
    for nm, sl in [('out', slice(0,192)), ('h', slice(192,960)), ('z', slice(960,992))]:
        no = np.linalg.norm(xo[:, sl], axis=1); nd = np.linalg.norm(dx[:, sl], axis=1)
        P(f'  {nm} |x|med {np.median(no):.1f}: ' + ' '.join(f'{g}:{np.median(nd[m]):.3g}/{np.percentile(nd[m],99):.3g}(n{m.sum()})' for g, m in groups if m.sum() > 0))
    return lag

for k, tag in [(14, 'cur'), (10, 'v6'), (11, 'v7')]:
    L = info.get(k, [])
    try:
        if 'factual_edited.csv' in L: fact_stats(k, tag)
    except Exception as ex: print('fact err', k, repr(ex))
    try:
        if 'dose_edit_edited.csv' in L: dose_stats(k, tag)
    except Exception as ex: print('dose err', k, repr(ex))
try: lag15 = state_stats(15, 'cur')
except Exception as ex: print('state err', repr(ex))

for n in edited():
    D = (load(n) - load_original(n)).reshape(SHAPES[n][0], -1).astype(np.float64)
    sv = np.linalg.svd(D, compute_uv=False)
    P(f'  edit {n}: |dW| {np.linalg.norm(D):.3g} rank(>1e-4 max) {(sv > 1e-4*sv[0]).sum()} top sv {np.round(sv[:4],3).tolist()}')

fn = []
for k_, v_ in list(globals().items()):
    if k_.startswith('_') or not callable(v_) or isinstance(v_, type): continue
    if getattr(v_, '__module__', '') != '__main__': continue
    try: sig = str(inspect.signature(v_))
    except Exception: sig = '?'
    fn.append(f'{k_}{sig}')
print('FUNCS:', '; '.join(fn)[:2500])

with open('/work/notes.md', 'a') as f:
    f.write('\n## Restart 11 (run 288): fwd_14/15 aggregates (current edit = v9 build8(B,tI5,tN5,1.0,2.0,0.4))\n' + '\n'.join('- ' + s for s in S) + '\n')
print('notes appended')
```