The sampling test used the wrong type index: a retained loop variable held 14, not the index used to build the query. Its result therefore says nothing about the reconstruction. I’ll recover the index from the query itself and repeat the test. A close match after averaging would validate the readout; a systematic gap would not.

```python
import numpy as np, torch
type_errors127 = [
    float(torch.max(torch.abs(
        layer_norm(view121 + W["type_emb.weight"][t127], "query_norm") - query121)))
    for t127 in range(W["type_emb.weight"].shape[0])
]
type127 = int(np.argmin(type_errors127))
print("EXACT QUERY TYPE", type127, type_errors127[type127])
static127 = (attrs121[jbest121] - query121).clone()
trans127 = torch.as_tensor(trans120).clone()
centres127 = torch.as_tensor(cent120).clone()
base_logits127 = mlp(query121 + static127, "attr_heads.value_bin") + trans127
base_means127 = (torch.softmax(base_logits127, -1) * centres127).sum(-1)
print("CANDIDATE REPLAY ERROR", float(torch.max(torch.abs(
    base_means127-torch.as_tensor(predictions121[jbest121])))))
rng127 = torch.Generator().manual_seed(9127)
mu127 = prior121[:, :32]
sd127 = prior121[:, 32:].clamp(-8, 4).exp()
samples127 = []
for draw127 in range(256):
    z127 = mu127 + sd127 * torch.randn(mu127.shape, generator=rng127)
    v127 = ctx121 + linear(z127, "z_head")
    q127 = layer_norm(v127 + W["type_emb.weight"][type127], "query_norm")
    l127 = mlp(q127 + static127, "attr_heads.value_bin") + trans127
    samples127.append((torch.softmax(l127, -1)*centres127).sum(-1).numpy())
samples127 = np.stack(samples127)
truth127 = as_numpy(truth120).reshape(-1)
codes127 = as_numpy(code120).reshape(-1)
print("SAMPLING", {
    "prior_mean_rmse": float(np.sqrt(np.mean((base_means127.numpy()-truth127)**2))),
    "averaged_rmse": float(np.sqrt(np.mean((samples127.mean(0)-truth127)**2))),
    "four_draw_sd": float(np.sqrt(np.mean(samples127.var(0)/4))),
    "mean_bias": float(np.mean(samples127.mean(0)-truth127)),
})
print("BY CODE", {
    int(c127): [float(np.mean(samples127.mean(0)[codes127==c127]-truth127[codes127==c127])),
                float(np.sqrt(np.mean((samples127.mean(0)[codes127==c127]-truth127[codes127==c127])**2)))]
    for c127 in np.unique(codes127)
})
readout_cache127 = {
    "type": type127, "context": ctx121.clone(), "view": view121.clone(),
    "static": static127, "transition": trans127, "centres": centres127,
    "codes": codes127.copy(), "truth": truth127.copy(),
}
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 127; forwards 7. Guarding against mutable-global contamination: prior sampling is repeated using the exact query type and cached register contribution. Original weights remain unchanged.\n")
```