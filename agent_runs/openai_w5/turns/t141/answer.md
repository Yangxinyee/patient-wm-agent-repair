The model uses four attention heads, and the freed head has enough space for both drug signals. I’ll install an isolated routing candidate: action bundles supply the signal, while rapid decay clears its private recurrent coordinates before the next update. This only establishes transport; factual NLL and dose-response tests can still reject it.

```python
snapshot125 = {n: torch.as_tensor(load(n)).clone() for n in edited()}
route125 = {}

def rw125(name):
    if name not in route125:
        route125[name] = torch.as_tensor(load(name)).clone()
    return route125[name]

R125 = np.asarray(R117, dtype=int)
S125 = 0.01
sentinel125 = torch.tensor([-S125, -S125, S125, S125])

# Non-action tokens have negative carrier scores; the zero action token scores zero.
rw125("core.slots")[:, R125] = sentinel125
rw125("encode.norm.weight")[R125] = 0
rw125("encode.norm.bias")[R125] = sentinel125
zin_names125 = [n for n in NAMES if re.fullmatch(r"(core\.)?z_in\.(weight|bias)", n)]
assert len(zin_names125) == 2, zin_names125
for n in zin_names125:
    if n.endswith(".weight"):
        rw125(n)[R125, :] = 0
    else:
        rw125(n)[R125] = sentinel125

# A layer-normalized vector has an exact constant linear functional.
g125 = torch.as_tensor(load("core.prelude.0.norm1.weight"))
b125 = torch.as_tensor(load("core.prelude.0.norm1.bias"))
inv125 = 1 / g125
const125 = float(torch.sum(b125 * inv125))
assert abs(const125) > 1e-4, const125
constant_row125 = inv125 / const125

def centered_difference125(i, j):
    v = torch.zeros(192)
    v[i], v[j] = inv125[i], -inv125[j]
    return v - torch.dot(v, b125) * constant_row125

q125 = rw125("core.prelude.0.attn.q.weight")
kv125 = rw125("core.prelude.0.attn.kv.weight")
ao125 = rw125("core.prelude.0.attn.out.weight")
q125[144:192] = 0
q125[144] = constant_row125
kv125[144:192] = 0
kv125[336:384] = 0
carrier125 = torch.zeros(192)
carrier125[R125] = torch.tensor([1., 1., -1., -1.]) * inv125[R125]
kv125[144] = 10000 * carrier125
kv125[336] = centered_difference125(R125[0], R125[1])
kv125[337] = centered_difference125(R125[2], R125[3])
ao125[:, 144:192] = 0
ao125[R125, :] = 0
ao125[R125[0], 144], ao125[R125[1], 144] = 0.5, -0.5
ao125[R125[2], 145], ao125[R125[3], 145] = 0.5, -0.5
rw125("core.prelude.0.ffn.2.weight")[R125, :] = 0
rw125("core.prelude.0.ffn.2.bias")[R125] = 0
rw125("core.prelude_norm.bias")[R125] = 0

# Remove generic residual writes to private coordinates.
for n in NAMES:
    if n.startswith("core.blocks.") and (
        n.endswith(".attn.out.weight") or n.endswith(".cross.out.weight")
        or n.endswith(".ffn.2.weight") or n.endswith(".ffn.2.bias")
    ):
        t = rw125(n)
        if t.ndim == 2:
            t[R125, :] = 0
        else:
            t[R125] = 0

# Fresh small paired signals; prior private h vanishes over every positive time gap.
rw125("core.flow.a")[R125] = 0.0
rw125("core.delta_in")[R125] = float(np.log(np.expm1(1.0)))
bi125 = rw125("core.b_in.weight")
bi125[R125, :] = 0
pg125 = torch.as_tensor(load("core.prelude_norm.weight"))
eps125 = 0.1
for a, b in [(R125[0], R125[1]), (R125[2], R125[3])]:
    v = torch.zeros(192)
    v[a], v[b] = eps125 / pg125[a], -eps125 / pg125[b]
    bi125[a], bi125[b] = v, -v
co125 = rw125("core.c_out.weight")
co125[:, R125] = 0
co125[R125, :] = 0
for r in R125:
    co125[r, r] = 1

# Independently test action-token selection, including the zero action token.
nprobe125 = 600
rng125 = np.random.default_rng(125)
tokens125 = torch.tensor(rng125.normal(size=(nprobe125, 7, 192)), dtype=torch.float32)
tokens125[:, :, R125] = sentinel125
u125 = torch.tensor(ucheck123[:nprobe125])
u125[~torch.tensor(active123[:nprobe125])] = 0
tokens125[:, 5, :] = u125
tn125 = F.layer_norm(tokens125, (192,), g125, b125)
qq125 = F.linear(tn125, q125)[:, :4, 144]
kk125 = F.linear(tn125, kv125)[:, :, 144]
att125 = torch.softmax(qq125[:, :, None] * kk125[:, None, :] / np.sqrt(48), dim=-1)
print("constant-query max error:", float((qq125 - 1).abs().max()))
print("minimum action-token attention:", float(att125[:, :, 5].min()))
assert float(att125[:, :, 5].min()) > 0.999
for n, t in route125.items():
    save_edit(n, t)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 125: isolated dose-transport candidate\n"
        "- Routed head 3 to the action token using a carrier discriminator and exact constant query.\n"
        "- Private state coordinates receive paired rate signals and decay before subsequent updates.\n"
        "- No causal value-head correction installed. This is a transport/capacity test only.\n"
        "- Must pass GPU factual NLL and saved-state rate-transport checks before use.\n"
    )
print("new routing tensors:", len(route125))
print("z input tensors:", zin_names125)
print("total edited tensors:", len(edited()))
```
