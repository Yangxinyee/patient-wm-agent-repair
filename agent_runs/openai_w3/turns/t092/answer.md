The full-stream posterior agrees closely, and action tokens receive substantial attention—unlike the latent tokens. That supports moving the repair into the prelude. I’ll first reproduce its downstream state update, so the correction’s direction and scale can be calibrated without guessing.

```python
def attention82(x, prefix, memory=None, mask=None, age=None, age_mode=0):
    q82 = linear(x, prefix + ".q")
    source82 = x if memory is None else memory
    if memory is not None and age is not None and age_mode == 0:
        source82 = source82 + age
    k82, v82 = linear(source82, prefix + ".kv").chunk(2, dim=-1)
    if memory is not None and age is not None:
        if age_mode == 1:
            k82 = k82 + age
        elif age_mode == 2:
            k82 = linear(memory + age, prefix + ".kv")[..., :192]
    def split82(t):
        return t.reshape(len(t), t.shape[1], 4, 48).transpose(1, 2)
    q82, k82, v82 = split82(q82), split82(k82), split82(v82)
    scores82 = q82 @ k82.transpose(-1, -2) / np.sqrt(48)
    if mask is not None:
        scores82 = scores82.masked_fill(~mask[:, None, None, :], -1e9)
    a82 = torch.softmax(scores82, dim=-1)
    joined82 = (a82 @ v82).transpose(1, 2).reshape(len(x), x.shape[1], 192)
    return linear(joined82, prefix + ".out")

def layer82(x, prefix, memory=None, mask=None, age=None, age_mode=0):
    x = x + attention82(layer_norm(x, prefix + ".norm1"),
                        prefix + ".attn")
    if memory is not None:
        x = x + attention82(layer_norm(x, prefix + ".norm2"),
                            prefix + ".cross", memory, mask, age, age_mode)
    return x + mlp(layer_norm(x, prefix + ".norm3"), prefix + ".ffn")

def state_update82(h, E, memory, mask, age, age_mode=0):
    x = h + E
    recurrent = layer82(x, "core.blocks.0.0",
                        memory, mask, age, age_mode)
    recurrent = layer82(recurrent, "core.blocks.0.1",
                        memory, mask, age, age_mode)
    delta82 = torch.nn.functional.softplus(W["core.delta_in"])
    abar82 = torch.exp(-delta82 * torch.exp(W["core.a_in"]))
    newh82 = abar82 * h + delta82 * linear(E, "core.b_in") + recurrent - x
    newh82 = layer82(linear(newh82, "core.c_out"), "core.coda.0")
    out82 = layer_norm(newh82.mean(dim=1), "core.out_norm")
    return newh82, out82

stay_indices82 = {
    sid: np.flatnonzero(np.asarray(S["stay_id"]) == sid)
    for sid in np.unique(S["stay_id"])
}
positions82 = np.flatnonzero(active81)
sel82 = positions82[np.linspace(0, len(positions82) - 1, 32).astype(int)]
target_h82 = torch.as_tensor(
    S["x"][idx81[sel82], 192:960].reshape(-1, 4, 192),
    dtype=torch.float32
)
target_out82 = torch.as_tensor(
    S["x"][idx81[sel82], :192], dtype=torch.float32
)

def memory82(indices, include_current):
    histories82 = []
    for i82 in indices:
        history82 = stay_indices82[S["stay_id"][i82]]
        history82 = history82[history82 <= i82 if include_current
                             else history82 < i82][-128:]
        histories82.append(history82)
    length82 = max(map(len, histories82))
    mem82 = np.zeros((len(indices), length82, 192), dtype=np.float32)
    ages82 = np.zeros((len(indices), length82, 1), dtype=np.float32)
    masks82 = np.zeros((len(indices), length82), dtype=bool)
    for j82, hist82 in enumerate(histories82):
        mem82[j82, :len(hist82)] = e80[hist82]
        ages82[j82, :len(hist82), 0] = hours79[indices[j82]] - hours79[hist82]
        masks82[j82, :len(hist82)] = True
    return (torch.from_numpy(mem82), torch.from_numpy(masks82),
            torch.from_numpy(ages82))

tests82 = []
for include82 in (False, True):
    mem82, mask82, age_hours82 = memory82(idx81[sel82], include82)
    for seconds82 in (1.0, 3600.0):
        for divisor82 in (1.0, 10.0):
            age_emb82 = mlp(
                torch.log1p(age_hours82 * seconds82) / divisor82, "core.age"
            )
            for mode82 in range(3):
                nh82, no82 = state_update82(
                    h81[sel82], E81[sel82], mem82, mask82, age_emb82, mode82
                )
                err82 = float((nh82 - target_h82).square().mean().sqrt())
                oerr82 = float((no82 - target_out82).square().mean().sqrt())
                tests82.append((err82, oerr82, include82,
                                seconds82, divisor82, mode82))
print("Best update conventions (h RMSE, out RMSE, current, time, divisor, mode):")
for result82 in sorted(tests82)[:8]:
    print(result82)

with open("/work/notes.md", "a") as f:
    f.write(
        "\nPython run 82; forwards 7. Full posterior RMSE 2.42e-5; action "
        "attention ~0.232 versus latent ~3e-7 on WORLD drug bundles. "
        f"Reconstructing recurrent update; best convention {min(tests82)}. "
        "No weights edited.\n"
    )
```