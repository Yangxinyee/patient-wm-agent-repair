The rate helper’s shared dictionary is stale relative to the edited encoder. I’ll now test finite-dose routing with all current weights and embeddings. If halving carries less than half the signal, that could explain the early rollout asymmetry; proportional signals would shift attention back to feedback and centering.

```python
import torch.nn.functional as F
current162 = {n: torch.as_tensor(load(n), dtype=torch.float32) for n in NAMES}
savedW0162 = W0

def exact_u162(q, p):
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=bool)
    n = p.sum(1)
    total = torch.zeros((len(q), 192))
    for j, event in enumerate(("insulin", "norepinephrine")):
        idx = np.flatnonzero(p[:, j])
        if len(idx):
            atoms = action_atom_exact(event, q[idx, j])
            hidden = F.gelu(F.linear(atoms, current162["encode_action.mlp1.0.weight"],
                                   current162["encode_action.mlp1.0.bias"]))
            total[idx] += F.linear(hidden, current162["encode_action.mlp1.2.weight"],
                                  current162["encode_action.mlp1.2.bias"])
    total /= torch.as_tensor(np.sqrt(np.maximum(n, 1)), dtype=torch.float32)[:, None]
    size = torch.as_tensor(np.log1p(n), dtype=torch.float32)[:, None]
    total += F.linear(size, current162["encode_action.size.weight"],
                      current162["encode_action.size.bias"])
    hidden = F.gelu(F.linear(total, current162["encode_action.mlp2.0.weight"],
                            current162["encode_action.mlp2.0.bias"]))
    raw = F.linear(hidden, current162["encode_action.mlp2.2.weight"],
                   current162["encode_action.mlp2.2.bias"])
    u = F.layer_norm(raw, (192,), current162["encode_action.norm.weight"],
                     current162["encode_action.norm.bias"])
    u[n == 0] = 0
    return u

try:
    W0 = current162
    testq162 = np.array([[i, ne] for i, ne in
        [(0, 0), (.25, 0), (.5, 0), (1, 0), (2, 0), (4, 0), (8, 0),
         (0, .05), (0, .1), (0, .2), (0, .4), (0, .8), (0, 1.6),
         (1, .2), (2, .4), (4, .8)]])
    testp162 = testq162 > 0
    # Keep an infusion atom present even for the zero-rate test.
    testp162[0, 0] = True
    u162 = exact_u162(testq162, testp162)

    st162 = dict(np.load("/work/fwd_15/states_edited.npz"))
    j162 = int(np.flatnonzero(st162["has_insulin"] | st162["has_norepinephrine"])[0])
    dt162 = float((st162["hour"][j162] - st162["hour"][j162-1]) * 3600)
    h162 = torch.as_tensor(st162["x"][j162-1, 192:960]).reshape(4, 192)
    h162 *= torch.exp(-dt162 * torch.exp(current162["core.flow.a"]))
    z162 = torch.as_tensor(st162["x"][j162, 960:992])
    tokens162 = torch.cat([
        (h162 + current162["core.slots"])[None].expand(len(testq162), -1, -1),
        torch.zeros(len(testq162), 1, 192),
        u162[:, None],
        F.linear(z162, current162["z_in.weight"], current162["z_in.bias"])
          [None, None].expand(len(testq162), -1, -1)
    ], dim=1)

    print("config", VOCAB["config"])
    print("wm_attention globals", wm_attention.__code__.co_names)
    print("wm_layer globals", wm_layer.__code__.co_names)
    print("wm_norm globals", wm_norm.__code__.co_names)

    # Inspect the reserved head's action-token value signal.
    normkeys162 = [n for n in current162 if n.startswith("core.prelude.0") and "norm" in n]
    print("prelude norm keys", normkeys162)
    prefix162 = "core.prelude.0.norm1"
    if prefix162 + ".weight" not in current162:
        prefix162 = normkeys162[0].rsplit(".", 1)[0]
    tn162 = F.layer_norm(tokens162, (192,), current162[prefix162 + ".weight"],
                         current162[prefix162 + ".bias"])
    qq162 = F.linear(tn162, current162["core.prelude.0.attn.q.weight"])
    kv162 = F.linear(tn162, current162["core.prelude.0.attn.kv.weight"])
    kk162, vv162 = kv162.chunk(2, dim=-1)
    nh162 = 6
    dh162 = 192 // nh162
    qh162 = qq162.reshape(-1, 7, nh162, dh162).transpose(1, 2)
    kh162 = kk162.reshape(-1, 7, nh162, dh162).transpose(1, 2)
    vh162 = vv162.reshape(-1, 7, nh162, dh162).transpose(1, 2)
    att162 = torch.softmax(qh162 @ kh162.transpose(-1, -2) / np.sqrt(dh162), dim=-1)
    routed162 = (att162[:, 3, :4] @ vh162[:, 3]).mean(1)
    signal162 = F.linear(routed162,
        current162["core.prelude.0.attn.out.weight"][[61, 21, 15, 66], 96:128])
    print(pd.DataFrame({
        "I": testq162[:, 0], "NE": testq162[:, 1],
        "action_attention": as_np(att162[:, 3, :4, 5].mean(1)),
        "I_pair_signal": as_np(signal162[:, 0] - signal162[:, 1]),
        "NE_pair_signal": as_np(signal162[:, 2] - signal162[:, 3])
    }).round(7).to_string(index=False))
finally:
    W0 = savedW0162
```
