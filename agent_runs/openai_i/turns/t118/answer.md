The diagnostics show that posterior states retain more insulin innovation than the prediction heads, while norepinephrine needs an earlier feature. Before editing, I’ll test one unresolved encoding hypothesis: infusion END markers may enter the observation encoder rather than the action encoder. Agreement with exposed posterior states would confirm it; otherwise I’ll stop using that reconstruction.

```python
import torch, torch.nn.functional as F, numpy as np, time
torch.set_num_threads(8)
torch.set_grad_enabled(False)

def mlp109(x, prefix):
    return repair_linear(F.gelu(repair_linear(x, prefix + ".0")), prefix + ".2")

def pool109(atoms, prefix, empty_mode="encoded"):
    if not atoms and empty_mode == "zero":
        return torch.zeros(192, dtype=torch.float32)
    if atoms:
        s = mlp109(torch.stack(atoms), prefix + ".mlp1").sum(0) / np.sqrt(len(atoms))
    else:
        s = torch.zeros(192, dtype=torch.float32)
    size = torch.tensor([np.log1p(len(atoms))], dtype=torch.float32)
    return repair_norm(mlp109(s + repair_linear(size, prefix + ".size"),
                             prefix + ".mlp2"), prefix + ".norm")

def posterior109(ctx, e, u):
    hidden = F.gelu(repair_linear(torch.cat([ctx, e, u], dim=-1), "z_post.0"))
    return hidden, repair_linear(hidden, "z_post.2")[:, :32]

cases109 = _probe_cases99
ctx109 = latent_context(np.array([c[0] for c in cases109]))
target109 = torch.from_numpy(latent_state[[c[0] for c in cases109], -32:])
start_atoms109 = [[latent_start_atom(r) for r in c[1]] for c in cases109]
u_start109 = {
    mode: torch.stack([pool109(a, "encode_action", mode) for a in start_atoms109])
    for mode in ["encoded", "zero"]
}
cfgs109 = list(dict.fromkeys(entry[2] for entry in _grid99))
results109 = []
t109 = time.time()
for cfg109 in cfgs109:
    e109 = torch.stack([
        pool109([end_atom99(r, cfg109) for r in c[2]], "encode")
        for c in cases109
    ])
    for mode109, u109 in u_start109.items():
        _, z109 = posterior109(ctx109, e109, u109)
        err109 = torch.sqrt(((z109 - target109) ** 2).mean(1))
        results109.append((float(torch.sqrt(((z109-target109)**2).mean())),
                           float(err109.max()), cfg109, mode109))
results109.sort(key=lambda x: x[0])
print("END-as-observation candidates:", len(results109))
print("Best five:", results109[:5])
print("Elapsed seconds:", round(time.time()-t109, 1))

best109 = results109[0]
check_cases109 = [c for c in latent_bundles
                  if c[2] and c[0] not in set(_probe_indices99)][:160]
if check_cases109:
    ci109 = np.array([c[0] for c in check_cases109])
    ce109 = torch.stack([
        pool109([end_atom99(r, best109[2]) for r in c[2]], "encode")
        for c in check_cases109
    ])
    cu109 = torch.stack([
        pool109([latent_start_atom(r) for r in c[1]], "encode_action", best109[3])
        for c in check_cases109
    ])
    ch109, cz109 = posterior109(latent_context(ci109), ce109, cu109)
    ct109 = torch.from_numpy(latent_state[ci109, -32:])
    print("Independent END-bundle RMS:",
          float(torch.sqrt(((cz109-ct109)**2).mean())))
print("Edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 109 — decisive END-routing test\n"
            "- Tested END markers through the observation encoder, with starts through the action encoder.\n"
            f"- Best probe result: {best109}; separate-bundle validation printed above.\n"
            "- No weights changed. Reject this pathway unless exposed-state agreement is close.\n")
```