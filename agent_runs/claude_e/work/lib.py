
import torch, numpy as np
torch.set_grad_enabled(False)
_cache = {}
def T(n):
    return torch.as_tensor(np.asarray(load(n)), dtype=torch.float64)
def ln(x, name, eps=1e-5):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), T(name+'.weight'), T(name+'.bias'), eps=eps)
def lin(x, name, bias=True):
    y = x @ T(name+'.weight').T
    return y + T(name+'.bias') if bias else y
ACT = {'gelu': torch.nn.functional.gelu, 'relu': torch.relu, 'silu': torch.nn.functional.silu, 'tanh': torch.tanh}
def mlp2(x, name, act='gelu'):
    return lin(ACT[act](lin(x, name+'.0')), name+'.2')
def ctx_from_state(out, h4, dt_s, gap_div=10.0, act='gelu'):
    dec = torch.exp(-dt_s[:,None,None] * torch.exp(T('core.flow.a'))[None,None,:])
    hd = h4 * dec
    g = mlp2(torch.log(dt_s)[:,None]/gap_div, 'gap', act)
    return out + ln(hd.mean(1), 'core.out_norm') + g
def z_draws(ctx, nz, gen, act='gelu'):
    o = mlp2(ctx, 'z_prior', act); mu, lsd = o[:, :32], o[:, 32:]
    eps = torch.randn((nz,)+mu.shape, generator=gen, dtype=torch.float64)
    return mu[None] + torch.exp(lsd)[None]*eps

# ---- validated value-head reconstruction (run 72-73): MAE vs tool 1.32 = noise floor of 4 z-draws
# type_idx for measurements = 1; last_numeric features: col0 has_last, col3 log1p(age_s)/10, col6 z_last_value (clipped), others 0
# gap input log(dt_s)/10; z from prior (first 32 = mu, last 32 = log sd); GELU everywhere; predicted mean = sum_{b>=1} p_b*centre[b-1]
def q_from_rows(Xs, prev_idx, dt_s, nz=8, seed=0, type_idx=1):
    gen = torch.Generator().manual_seed(seed)
    xs = Xs[prev_idx]; out = xs[:, :192]; h4 = xs[:, 192:960].reshape(-1,4,192)
    ctx = ctx_from_state(out, h4, dt_s)
    zs = z_draws(ctx, nz, gen)
    view = ctx[None] + lin(zs, 'z_head')
    return ln(view + T('type_emb.weight')[type_idx], 'query_norm')
def reg_feats(has, age_s, zl):
    f = torch.zeros(len(has), 8, dtype=torch.float64)
    f[:,0] = has; f[:,3] = torch.log1p(age_s)/10; f[:,6] = zl
    return f
def value_logits(Q, code, lvb, f, W0=None, b0=None, W2=None, b2=None):
    W0 = T('attr_heads.value_bin.0.weight') if W0 is None else W0; b0 = T('attr_heads.value_bin.0.bias') if b0 is None else b0
    W2 = T('attr_heads.value_bin.2.weight') if W2 is None else W2; b2 = T('attr_heads.value_bin.2.bias') if b2 is None else b2
    const = T('embed.code.weight')[code] + T('embed.bins.value.weight')[lvb] + f @ T('last_numeric.weight').T + T('last_numeric.bias')
    wc = Q + const[None]
    hid = torch.nn.functional.gelu(wc @ W0.T + b0)
    return hid @ W2.T + b2 + T('embed_transition.value_bin.weight')[code*17 + lvb][None], wc, hid

KEYS = {'glucose_finger':'gf','glucose_lab':'gl','map_arterial':'ma','map_cuff':'mc'}
CODE = {'glucose_finger':6,'glucose_lab':10,'map_arterial':7,'map_cuff':5}
def build_reading_rows(npz, G):
    zz = np.load(npz, allow_pickle=True)
    Xs = torch.as_tensor(np.array(zz['x']), dtype=torch.float64)
    st = pd.DataFrame({'stay': zz['stay_id'], 'hour': zz['hour']}); st['row'] = np.arange(len(st))
    rows = []
    for sid, g in st.groupby('stay', sort=False):
        hrs_ = g.hour.values; rws = g.row.values
        for ev, k in KEYS.items():
            ser = G[k].loc[sid]; ser = ser[ser.notna()]
            m_, iq_, edges = med[ev]
            last_t = None; last_v = None
            for t, v in ser.items():
                i = np.searchsorted(hrs_, t-1e-6) - 1
                if i < 0: 
                    last_t, last_v = t, v; continue
                if last_t is None: hs, age, zl_, lvb_ = 0.0, 0.0, 0.0, 0
                else: hs, age, zl_, lvb_ = 1.0, (t-last_t)*3600, float(np.clip((last_v-m_)/iq_,-10,10)), 1+int(np.sum(np.array(edges) < last_v))
                rows.append((sid, t, ev, CODE[ev], rws[i], (t-hrs_[i])*3600, hs, age, zl_, lvb_, v, 1+int(np.sum(np.array(edges) < v))))
                last_t, last_v = t, v
    R_ = pd.DataFrame(rows, columns=['stay','hour','event','code','prev','dt','has','age','zl','lvb','obs','obin'])
    return Xs, R_
def head_eval(Xs, R_, nz=16, seed=0, W=None, chunk=4000):
    W = W or {}
    pm_all, nll_all, P_all = [], [], []
    for a in range(0, len(R_), chunk):
        r = R_.iloc[a:a+chunk]
        Q = q_from_rows(Xs, torch.as_tensor(r.prev.values), torch.as_tensor(r.dt.values, dtype=torch.float64), nz=nz, seed=seed+a)
        f = reg_feats(torch.as_tensor(r.has.values), torch.as_tensor(r.age.values), torch.as_tensor(r.zl.values))
        code = torch.as_tensor(r.code.values); lvb = torch.as_tensor(r.lvb.values)
        lg, wc, hid = value_logits(Q, code, lvb, f, **W)
        p = torch.softmax(lg, -1).mean(0)
        cent = torch.stack([CENT[int(c)] for c in r.code.values])
        pm_all.append((p[:,1:]*cent).sum(1).numpy())
        nll_all.append(-torch.log(p[torch.arange(len(r)), torch.as_tensor(r.obin.values)]).numpy())
        P_all.append(p.numpy())
    return np.concatenate(pm_all), np.concatenate(nll_all), np.concatenate(P_all)
