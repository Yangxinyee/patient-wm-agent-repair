The tensor layout supports a direct CPU replay of the recurrent update. That would let me measure how altered doses change the head’s inputs, rather than assume a latent direction has the right effect. I’ll validate one-step replay against exported states first; only numerical agreement would justify using it to design an edit.

```python
import torch.nn.functional as F
import itertools, time
torch.set_num_threads(8)
torch.set_grad_enabled(False)

rx_W = rr_original
def rx_linear(x, p):
    return F.linear(x, rx_W[p+".weight"], rx_W.get(p+".bias"))

def rx_norm(x, p):
    return F.layer_norm(x, (192,), rx_W[p+".weight"], rx_W[p+".bias"], 1e-5)

def rx_mlp(x, p, act="gelu"):
    a = rx_linear(x, p+".0")
    return rx_linear(F.gelu(a) if act == "gelu" else F.silu(a), p+".2")

def rx_attn(x, p):
    q = rx_linear(x, p+".q")
    k, v = rx_linear(x, p+".kv").chunk(2, -1)
    B, L, D = q.shape
    q = q.reshape(B,L,4,48).transpose(1,2)
    k = k.reshape(B,L,4,48).transpose(1,2)
    v = v.reshape(B,L,4,48).transpose(1,2)
    a = F.scaled_dot_product_attention(q,k,v)
    return rx_linear(a.transpose(1,2).reshape(B,L,D), p+".out")

def rx_cross(x, m, age, valid, p, mode, age_scale, act):
    q = rx_linear(x, p+".q")
    av = rx_mlp(torch.log1p(age.clamp_min(0)).unsqueeze(-1)/age_scale, "core.age", act)
    if mode == "before_kv":
        k, v = rx_linear(m+av, p+".kv").chunk(2,-1)
    else:
        k, v = rx_linear(m, p+".kv").chunk(2,-1)
        k = k+av
    B,L,D = q.shape
    S = k.shape[1]
    a = F.scaled_dot_product_attention(
        q.reshape(B,L,4,48).transpose(1,2),
        k.reshape(B,S,4,48).transpose(1,2),
        v.reshape(B,S,4,48).transpose(1,2),
        attn_mask=valid[:,None,None,:]
    )
    return rx_linear(a.transpose(1,2).reshape(B,L,D), p+".out")

def rx_layer(x, p, act, memory=None, age=None, valid=None,
             mode="after_k", age_scale=1.0):
    x = x + rx_attn(rx_norm(x,p+".norm1"), p+".attn")
    if memory is not None:
        x = x + rx_cross(rx_norm(x,p+".norm2"), memory, age, valid,
                         p+".cross", mode, age_scale, act)
    return x + rx_mlp(rx_norm(x,p+".norm3"), p+".ffn", act)

rx_keys = [(int(s), int(m)) for s,m in rr_keys4]
rx_hist = {}
rx_history4 = []
for i,(s,m) in enumerate(rx_keys):
    rx_history4.append(list(rx_hist.get(s, [])))
    rx_hist.setdefault(s, []).append(i)

rx_rng = np.random.default_rng(124)
rx_pick = rx_rng.choice(np.flatnonzero(rr_valid4 & (rr_prev4 >= 0)), 48, replace=False)
rx_prev = torch.as_tensor(rr_state4["x"][rr_prev4[rx_pick]], dtype=torch.float32)
rx_dt = torch.tensor([
    (rx_keys[i][1]-rx_keys[rr_prev4[i]][1])*60.0 for i in rx_pick
], dtype=torch.float32)
rx_hflow = rx_prev[:,192:960].reshape(-1,4,192) * torch.exp(
    -rx_dt[:,None,None]*rx_W["core.flow.a"].exp()[None,None,:]
)
rx_e = rr_test_e111[rx_pick]
rx_u = rr_base_u112[rx_pick]
rx_z = rr_zexport4[rx_pick]
rx_targeth = rr_x4[rx_pick,192:960].reshape(-1,4,192)
rx_targetout = rr_x4[rx_pick,:192]

def rx_memory_for(indices, include_current=False):
    histories = [(rx_history4[i]+([int(i)] if include_current else []))[-128:] for i in indices]
    S = max(1, max(map(len,histories)))
    mem = torch.zeros(len(indices),S,192)
    ages = torch.zeros(len(indices),S)
    valid = torch.zeros(len(indices),S,dtype=torch.bool)
    for b,(i,hh) in enumerate(zip(indices,histories)):
        if hh:
            mem[b,:len(hh)] = rr_test_e111[hh]
            ages[b,:len(hh)] = torch.tensor([
                (rx_keys[i][1]-rx_keys[j][1])*60.0 for j in hh
            ])
            valid[b,:len(hh)] = True
    return mem,ages,valid

rx_replay_tests = []
rx_t0 = time.time()
for act, include, mode, age_scale in itertools.product(
        ["gelu","silu"], [False,True], ["after_k","before_kv"], [1.0,10.0]):
    tokens = torch.cat([
        rx_hflow+rx_W["core.slots"][None],
        rx_e[:,None], rx_u[:,None], rx_linear(rx_z,"z_in")[:,None]
    ],dim=1)
    pre = rx_layer(tokens,"core.prelude.0",act)
    E = rx_norm(pre[:,:4],"core.prelude_norm")
    mem,ages,valid = rx_memory_for(rx_pick,include)
    t = rx_hflow+E
    for p in ["core.blocks.0.0","core.blocks.0.1"]:
        t = rx_layer(t,p,act,mem,ages,valid,mode,age_scale)
    delta = F.softplus(rx_W["core.delta_in"])
    abar = torch.exp(-delta*rx_W["core.a_in"].exp())
    h = abar[None,None]*rx_hflow + delta[None,None]*rx_linear(E,"core.b_in") + (t-(rx_hflow+E))
    h = rx_layer(rx_linear(h,"core.c_out"),"core.coda.0",act)
    out = rx_norm(h.mean(1),"core.out_norm")
    rx_replay_tests.append((
        float((h-rx_targeth).square().mean().sqrt()),
        float((out-rx_targetout).square().mean().sqrt()),
        act,include,mode,age_scale
    ))
print("=== ONE-STEP REPLAY ERRORS ===")
for result in sorted(rx_replay_tests)[:8]:
    print(result)
print("seconds:", round(time.time()-rx_t0,2))
print("dt range:", rx_dt.min().item(),rx_dt.max().item())

with open("/work/notes.md","a") as f:
    f.write("\n\n## Run 124\nConstructed a gradient-free CPU one-step recurrent replay and tested activation, memory inclusion, and age-key conventions against original exported states. Best tests: "+repr(sorted(rx_replay_tests)[:4])+". No weights edited. Counts: 124 Python runs, 6 forward passes.\n")
```