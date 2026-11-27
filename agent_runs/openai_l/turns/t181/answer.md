The local controller fit still introduces substantial factual-state noise. I’ll avoid that controller and test a simpler route: a small dose signal written after the recurrent block, leaving the original dynamics intact. This is viable only if the write stays confined to quiet directions and preserves factual NLL.

```python
C167 = np.asarray(load_original("core.c_out.weight"), dtype=np.float64)
B167_original = np.asarray(load_original("core.b_in.weight"), dtype=np.float64)
R167 = np.asarray(R118, dtype=np.float64)
sv167 = np.linalg.svd(C167, compute_uv=False)
U167 = np.linalg.solve(C167, R167)
amp167 = 0.1
B167 = B167_original + amp167 * (U167 / np.asarray(delta160)[:, None]) @ np.asarray(D160).T

print("C singular range", float(sv167[-1]), float(sv167[0]))
print("write inverse norm", float(np.linalg.norm(U167)))
print("C inverse residual", float(np.linalg.norm(C167 @ U167 - R167)))
print("B increment norm", float(np.linalg.norm(B167-B167_original)))

keys167 = ["core.b_in.weight", "core.c_out.weight",
           "core.prelude.0.attn.kv.weight", "core.prelude.0.attn.out.weight"]
saved167 = {n: w63_cache[n].clone() for n in keys167}
try:
    for n in keys167:
        w63_cache[n] = torch.as_tensor(load_original(n), dtype=torch.float32)
    base167 = transition64(hf160, e160, u160, z160, mem160, ages160, mask160)
    w63_cache["core.prelude.0.attn.kv.weight"] = torch.as_tensor(load("core.prelude.0.attn.kv.weight"), dtype=torch.float32)
    w63_cache["core.prelude.0.attn.out.weight"] = torch.as_tensor(wire_out152, dtype=torch.float32)
    w63_cache["core.b_in.weight"] = torch.as_tensor(B167, dtype=torch.float32)
    test167 = transition64(hf160, e160, u160, z160, mem160, ages160, mask160)
    dh167 = np.asarray(test167[1]-base167[1])
    private167 = dh167 @ R167
    orth167 = dh167-private167 @ R167.T
    print("local private RMS", np.round(np.sqrt(np.mean(private167**2, axis=(0,1))), 6).tolist())
    print("local orthogonal RMS", float(np.sqrt(np.mean(orth167**2))))
    print("local out RMS", float(torch.mean((test167[0]-base167[0])**2).sqrt()))
finally:
    for n, v in saved167.items():
        w63_cache[n] = v

drop_edit("core.c_out.weight")
save_edit("core.b_in.weight", torch.as_tensor(B167, dtype=torch.float32))
save_edit("core.prelude.0.attn.out.weight", torch.as_tensor(wire_out152, dtype=torch.float32))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 167 / one-step quiet write, no persistence controller\n"
            "Abandoned harmful c_out controller. Original c_out retained; analytical "
            "right-inverse U=solve(C_original,R118) writes a 0.1-scale dose feature "
            "after recurrent blocks via b_in. Staged B167 and wire_out152 with "
            "reconstruction kv. Value head untouched. Pending factual isolation audit.\n")
print("CURRENT EDITS", edited())
print("\n=== HEAD HELPER ===")
print(skeleton149(head_inputs96, limit=2200))
print("Jselected118", np.asarray(Jselected118).shape)
print("Vsignal118", np.asarray(Vsignal118).shape)
print("responses118", responses118)
```