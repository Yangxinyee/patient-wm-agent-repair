That probe used six attention heads, but the model has four. Its zero-signal result is therefore invalid, and no weights changed. I’ll correct the head count and verify the attention layout against the existing implementation before drawing a transport conclusion.

```python
print("=== ATTENTION CONVENTIONS ===")
print("n_heads", VOCAB["config"]["n_heads"],
      "kv_layout", wm_kv_layout, "memory_mode", wm_memory_mode)

savedW0163 = W0
try:
    W0 = current162
    nh163 = VOCAB["config"]["n_heads"]
    dh163 = 192 // nh163
    qh163 = qq162.reshape(-1, 7, nh163, dh163).transpose(1, 2)
    helper163 = wm_attention(tn162, "core.prelude.0.attn", weights=current162)
    choices163 = {}
    for mode163 in ("split", "interleaved", "pairlast"):
        if mode163 == "split":
            k163, v163 = kv162.chunk(2, dim=-1)
            k163 = k163.reshape(-1, 7, nh163, dh163).transpose(1, 2)
            v163 = v163.reshape(-1, 7, nh163, dh163).transpose(1, 2)
        elif mode163 == "interleaved":
            kv163 = kv162.reshape(-1, 7, nh163, 2, dh163)
            k163 = kv163[:, :, :, 0].transpose(1, 2)
            v163 = kv163[:, :, :, 1].transpose(1, 2)
        else:
            kv163 = kv162.reshape(-1, 7, nh163, dh163, 2)
            k163 = kv163[..., 0].transpose(1, 2)
            v163 = kv163[..., 1].transpose(1, 2)
        a163 = torch.softmax(qh163 @ k163.transpose(-1, -2) / np.sqrt(dh163), -1)
        oh163 = a163 @ v163
        output163 = F.linear(oh163.transpose(1, 2).reshape(-1, 7, 192),
                             current162["core.prelude.0.attn.out.weight"])
        error163 = float((output163 - helper163).abs().max())
        choices163[mode163] = (error163, a163, oh163)
        print(mode163, "maxdiff_from_helper", error163)
    best163 = min(choices163, key=lambda k: choices163[k][0])
    assert choices163[best163][0] < 1e-4
    att163 = choices163[best163][1]
    head163 = choices163[best163][2][:, 3, :4].mean(1)
    sig163 = F.linear(head163,
        current162["core.prelude.0.attn.out.weight"][[61, 21, 15, 66], 144:192])
    print("=== CORRECTED PRIVATE-HEAD TRANSPORT ===")
    print(pd.DataFrame({
        "I": testq162[:, 0], "NE": testq162[:, 1],
        "action_attention": as_np(att163[:, 3, :4, 5].mean(1)),
        "I_pair_signal": as_np(sig163[:, 0] - sig163[:, 1]),
        "NE_pair_signal": as_np(sig163[:, 2] - sig163[:, 3])
    }).round(7).to_string(index=False))
finally:
    W0 = savedW0163
```
