Memory variations did not remove the error. One basic possibility remains untested: ReLU in the recurrent feed-forward layers. I’ll check it before abandoning local replay. Numerical agreement would validate the replay; otherwise I’ll use the forward tool directly for candidate testing.

```python
def rx_mlp(x,p,act="gelu"):
    a = rx_linear(x,p+".0")
    if act == "gelu":
        a = F.gelu(a)
    elif act == "silu":
        a = F.silu(a)
    elif act == "relu":
        a = F.relu(a)
    else:
        raise ValueError(act)
    return rx_linear(a,p+".2")

rx_tests127 = []
rx_layout = "split"
rx_t0 = time.time()
rx_mem127,rx_age127,rx_valid127 = rx_memory_for(rx_pick,True)
for pre_act,block_act,coda_act,age_act,mode in itertools.product(
    ["relu","gelu","silu"],["relu","gelu","silu"],
    ["relu","gelu","silu"],["relu","gelu","silu"],
    ["before_kv","before_k","after_k"]
):
    tokens = torch.cat([
        rx_hflow+rx_W["core.slots"][None],
        rx_e[:,None],rx_u[:,None],rx_linear(rx_z,"z_in")[:,None]
    ],1)
    E = rx_norm(rx_layer2(tokens,"core.prelude.0",pre_act)[:,:4],"core.prelude_norm")
    t = rx_hflow+E
    for p in ["core.blocks.0.0","core.blocks.0.1"]:
        t = rx_layer2(t,p,block_act,rx_mem127,rx_age127,rx_valid127,mode,age_act)
    delta = F.softplus(rx_W["core.delta_in"])
    abar = torch.exp(-delta*rx_W["core.a_in"].exp())
    h = abar[None,None]*rx_hflow+delta[None,None]*rx_linear(E,"core.b_in")+(t-(rx_hflow+E))
    h = rx_layer2(rx_linear(h,"core.c_out"),"core.coda.0",coda_act)
    out = rx_norm(h.mean(1),"core.out_norm")
    rx_tests127.append((
        float((h-rx_targeth).square().mean().sqrt()),
        float((out-rx_targetout).square().mean().sqrt()),
        pre_act,block_act,coda_act,age_act,mode
    ))
print("=== RELU-INCLUSIVE REPLAY TESTS ===")
for result in sorted(rx_tests127)[:8]:
    print(result)
print("seconds:",round(time.time()-rx_t0,2))
rx_replay_best127 = min(rx_tests127)

with open("/work/notes.md","a") as f:
    f.write("\n\n## Run 127\nMemory conventions did not validate replay. Added the previously omitted ReLU option for all feed-forward/age layers. Best tests: "+repr(sorted(rx_tests127)[:3])+". No edits. Counts: 127 Python runs, 6 forward passes.\n")
```