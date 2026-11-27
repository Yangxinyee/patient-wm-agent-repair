The refit removes the large mean offset. At penalty 0.01, it still closely matches the desired decay and has a manageable write route. I’ll connect that controller and the dose wire, leaving the value head unchanged, and check one-step factual leakage before a GPU audit. Large changes outside the private channels would refute this construction.

```python
penalty160 = 0.01
P160 = controllers159[penalty160]
U160 = routes159[penalty160]
C160 = Cperp145 + R118 @ P160.T

D160 = np.zeros((192, 4), dtype=np.float64)
D160[:, 0] = D160[:, 1] = QE125[:, 0] / wire_slopes153["insulin"]
D160[:, 2] = D160[:, 3] = QE125[:, 2] / wire_slopes153["norepinephrine"]
delta160 = np.asarray(
    torch.nn.functional.softplus(
        torch.as_tensor(load_original("core.delta_in"))), dtype=np.float64)
B160 = (np.asarray(load_original("core.b_in.weight"), dtype=np.float64)
        + (U160 / delta160[:, None]) @ D160.T)

names160 = ["core.prelude.0.attn.kv.weight",
            "core.prelude.0.attn.out.weight",
            "core.c_out.weight", "core.b_in.weight"]
cache160 = {name: w63(name).clone() for name in names160}
candidate160 = {
    names160[0]: np.asarray(load(names160[0]), dtype=np.float32),
    names160[1]: np.asarray(wire_out152, dtype=np.float32),
    names160[2]: np.asarray(C160, dtype=np.float32),
    names160[3]: np.asarray(B160, dtype=np.float32),
}

ids160 = np.asarray(ids152[:24], dtype=np.int64)
ctx160, hf160, mem160, ages160, mask160 = bundle_inputs74(ids160)
e160 = torch.as_tensor(e_valid60[ids160], dtype=torch.float32)
u160 = torch.as_tensor(u_valid60[ids160], dtype=torch.float32)
z160 = mlp63(torch.cat([ctx160, e160, u160], dim=-1), "z_post")[:, :32]

try:
    base160 = transition64(hf160, e160, u160, z160, mem160, ages160, mask160)
    for name, value in candidate160.items():
        w63_cache[name] = torch.as_tensor(value)
    test160 = transition64(hf160, e160, u160, z160, mem160, ages160, mask160)
finally:
    for name, value in cache160.items():
        w63_cache[name] = value

print("=== ONE-STEP FACTUAL LEAKAGE ===")
for i, (base, test) in enumerate(zip(base160, test160)):
    diff = (test-base).numpy()
    print("output", i, "shape", diff.shape,
          "RMS", float(np.sqrt(np.mean(diff**2))))
    if diff.shape[-1] == 192:
        private = diff @ R118
        outside = diff-private @ R118.T
        print(" private RMS", np.sqrt(np.mean(private**2, axis=tuple(range(private.ndim-1)))))
        print(" outside RMS", float(np.sqrt(np.mean(outside**2))))
print("B edit norm", float(np.linalg.norm(
    B160-np.asarray(load_original("core.b_in.weight")))))
print("delta range", delta160.min(), delta160.max())

for name, value in candidate160.items():
    save_edit(name, value)

with open("/work/notes.md", "a") as f160:
    f160.write(
        "\n## Run 160 / controller and dose wire staged\n"
        "Used factual-anchored controllers159[.01]; private states receive "
        "one raw-rate unit in both poles for each drug. Saved c_out, b_in, "
        "and wire_out152, retaining the reconstruction kv edit. Value head "
        "is untouched. Pending paired factual audit; these are not yet "
        "validated causal predictions.\n")
print("Staged edits:", edited())
```
