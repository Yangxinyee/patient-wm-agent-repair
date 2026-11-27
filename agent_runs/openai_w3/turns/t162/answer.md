The new replay does not match the verified prelude, so its head-selection results are not trustworthy and I have made no edits. I will compare it directly with the retained exact helper. Agreement in both attention and slot outputs is required before testing a transport channel.

```python
import dis

for name146 in ("prelude81", "prelude_cache88", "parts105"):
    fn146 = globals()[name146]
    print(name146, "constants:", fn146.__code__.co_consts)
    ins146 = list(dis.get_instructions(fn146))
    wanted146 = set()
    for i146, op146 in enumerate(ins146):
        if op146.argval in ("chunk", "reshape", "transpose", "softmax", "sqrt"):
            wanted146.update(range(max(0, i146-5), min(len(ins146), i146+9)))
    print("\n".join(
        f"{ins146[i].offset}: {ins146[i].opname} {ins146[i].argrepr}"
        for i in sorted(wanted146)
    )[:2300])

args146 = [torch.as_tensor(x, dtype=torch.float32)
           for x in (h81, e81, u81, z88)]
for name146 in ("prelude81", "prelude_cache88"):
    try:
        result146 = globals()[name146](*args146)
        if not isinstance(result146, (tuple, list)):
            result146 = (result146,)
        print(name146, "output shapes:",
              [getattr(x, "shape", type(x)) for x in result146])
        for j146, x146 in enumerate(result146):
            if tuple(getattr(x146, "shape", ())) == tuple(np.shape(E81)):
                print("E RMSE", j146, float(torch.mean(
                    (torch.as_tensor(x146) - torch.as_tensor(E81))**2).sqrt()))
        globals()[name146 + "_result146"] = result146
    except Exception as ex146:
        print(name146, type(ex146).__name__, str(ex146)[:180])

rawtokens146 = torch.cat((
    args146[0] + wt145("core.slots")[None],
    args146[1][:, None],
    args146[2][:, None],
    F.linear(args146[3], wt145("z_in.weight"), wt145("z_in.bias"))[:, None]
), dim=1)
rawparts146 = prelude_parts145(rawtokens146)
print("tokens81 vs constructed raw RMSE:",
      float(torch.mean((torch.as_tensor(tokens81) - rawtokens146)**2).sqrt()))
print("tokens81 vs constructed normalized RMSE:",
      float(torch.mean((torch.as_tensor(tokens81) - rawparts146[0])**2).sqrt()))
print("Constructed-raw E / attention RMSE:",
      float(torch.mean((rawparts146[6] - torch.as_tensor(E81))**2).sqrt()),
      float(torch.mean((rawparts146[7] - torch.as_tensor(attention81))**2).sqrt()))
print("W cache type:", type(W))
if isinstance(W, dict):
    print("Cached/original Q maximum difference:",
          float(torch.max(torch.abs(
              torch.as_tensor(W["core.prelude.0.attn.q.weight"])
              - wt145("core.prelude.0.attn.q.weight")))))

with open("/work/notes.md", "a") as f146:
    f146.write("\nPython run 146; forwards 9. New prelude replay failed equality checks, so no head edit is justified. Comparing retained exact helper and raw-versus-normalized token conventions; original weights remain unchanged.\n")
```