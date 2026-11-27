Run 68 settles the conventions:
- 4 heads, d_head 48, FFN width 768, 4 slots, memory of 128.
- Time is in seconds: the decay timescales span 600 s to 6.4e7 s, and `SB.dts` holds dt in seconds.
- `SB.enc_err` (about 5e-7 per bundle) is the stored check from run 63, so the bundle encoders are exact.
- `uALL` has no zero rows. Bundles without drug atoms use a nonzero "empty action" vector, so that is what z_post saw.

Next I'll write a CPU replica of one core step: time flow → prelude over [h+slots, e, u, z_in(z)] → two blocks with self- and cross-attention → linear update → c_out → coda → out_norm. I'll validate it against the true next state (`HS[k]`, `OUT[k]`), starting each step from the true previous state. The architecture text leaves three things open: how the age embedding enters the cross-attention, whether age is scaled by 1/10, and whether the memory includes the current bundle. I'll test those combinations.

An error near float precision (≲1e-4 against a state scale of ~1) for one variant would confirm the replica. If every variant gives errors of order 0.1, a structural detail is wrong (for example, the u token's handling), and I'll look at which stage diverges.

```python
import math, itertools, time
Po = {k: load_original(k) for k in NAMES}
def LNp(x, pre): return F.layer_norm(x, (192,), Po[pre+'.weight'], Po[pre+'.bias'], 1e-5)
def mha(xq, xkv, pre, kadd=None, kin=None, vin=None, P_=None):
    P_ = Po if P_ is None else P_
    Wkv = P_[pre+'.kv.weight']; q = xq @ P_[pre+'.q.weight'].T
    kx = xkv if kin is None else xkv + kin; vx = xkv if vin is None else xkv + vin
    k_ = kx @ Wkv[:192].T; v_ = vx @ Wkv[192:].T
    if kadd is not None: k_ = k_ + kadd
    q = q.view(-1,4,48).transpose(0,1); k_ = k_.view(-1,4,48).transpose(0,1); v_ = v_.view(-1,4,48).transpose(0,1)
    a = torch.softmax(q @ k_.transpose(1,2)/math.sqrt(48), -1)
    return (a @ v_).transpose(0,1).reshape(-1,192) @ P_[pre+'.out.weight'].T
def ffn(x, pre): return F.linear(F.gelu(F.linear(x, Po[pre+'.ffn.0.weight'], Po[pre+'.ffn.0.bias'])), Po[pre+'.ffn.2.weight'], Po[pre+'.ffn.2.bias'])
def age_emb(age_s, mode):
    x = torch.log1p(age_s)[:,None]
    if mode=='div10': x = x/10
    return F.linear(F.gelu(F.linear(x, Po['core.age.0.weight'], Po['core.age.0.bias'])), Po['core.age.2.weight'], Po['core.age.2.bias'])
def core_step(h, e, u, z, dt, M, Mage, var):
    hf = h * torch.exp(-dt*torch.exp(Po['core.flow.a']))
    zt = F.linear(z, Po['core.z_in.weight'], Po['core.z_in.bias'])
    T = torch.cat([hf + Po['core.slots'], e[None], u[None], zt[None]], 0)
    Tn = LNp(T,'core.prelude.0.norm1'); T = T + mha(Tn, Tn, 'core.prelude.0.attn')
    T = T + ffn(LNp(T,'core.prelude.0.norm3'), 'core.prelude.0')
    E = LNp(T[:4], 'core.prelude_norm'); x = hf + E
    ae = age_emb(Mage, var['age'])
    for L in ['core.blocks.0.0','core.blocks.0.1']:
        xn = LNp(x, L+'.norm1'); x = x + mha(xn, xn, L+'.attn')
        xn = LNp(x, L+'.norm2'); km = var['km']
        if km=='kadd': x = x + mha(xn, M, L+'.cross', kadd=ae)
        elif km=='kin': x = x + mha(xn, M, L+'.cross', kin=ae)
        else: x = x + mha(xn, M, L+'.cross', kin=ae, vin=ae)
        x = x + ffn(LNp(x, L+'.norm3'), L)
    sp = F.softplus(Po['core.delta_in']); abar = torch.exp(-sp*torch.exp(Po['core.a_in']))
    hn = abar*hf + sp*(E @ Po['core.b_in.weight'].T) + (x - (hf + E))
    y = hn @ Po['core.c_out.weight'].T
    yn = LNp(y,'core.coda.0.norm1'); y = y + mha(yn, yn, 'core.coda.0.attn')
    y = y + ffn(LNp(y,'core.coda.0.norm3'), 'core.coda.0')
    return y, LNp(y.mean(0), 'core.out_norm')
print("SB.row==arange:", bool((SB['row'].values==np.arange(len(SB))).all()))
sids_ = SB['stay_id'].values; mins_ = SB['minute'].values.astype(np.float32)
test_rows = [i for i in range(len(SB)) if not np.isnan(SB['prev'].values[i]) and sids_[i] in set(np.unique(sids_)[:2])][:80]
res_v = []
for km, ag, mem in itertools.product(['kadd','kin','kvin'], ['raw','div10'], ['prev','incl']):
    var = {'km':km,'age':ag}; ey=[]; eo=[]
    for r in test_rows:
        pr_i = int(SB['prev'].values[r])
        same = np.where(sids_==sids_[r])[0]; mem_rows = same[same<r] if mem=='prev' else same[same<=r]; mem_rows = mem_rows[-128:]
        M = eALL[mem_rows]; Mage = torch.tensor((mins_[r]-mins_[mem_rows])*60.0)
        y, o = core_step(HS[pr_i], eALL[r], uALL[r], ZP[r], float(SB['dts'].values[r]), M, Mage, var)
        ey.append(float((y-HS[r]).abs().mean())); eo.append(float((o-OUT[r]).abs().mean()))
    res_v.append((km,ag,mem,np.mean(ey),np.max(ey),np.mean(eo)))
for rv in sorted(res_v, key=lambda t:t[3]): print(rv)
print("scale |HS|", float(HS.abs().mean()), "|OUT|", float(OUT.abs().mean()))
```