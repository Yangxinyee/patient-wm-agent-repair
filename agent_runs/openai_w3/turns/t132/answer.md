The factual outputs and states cover both training and tuning stays, so they must be matched carefully. The remaining readout error could also come from the query type or register fields. I’ll test those on a fixed batch. A close match would support the readout reconstruction; a large error would mean another convention is still missing.

```python
# Recover what base97 already contains, rather than guessing register fields.
B120 = as_numpy(base97).astype(np.float64)
code120 = as_numpy(W["embed.code.weight"])[codes97]
vb120 = as_numpy(W["embed.bins.value.weight"])[bins97]
rb120 = as_numpy(W["embed.bins.rate.weight"])[0]
lw120 = as_numpy(W["last_numeric.weight"]).astype(np.float64)
lb120 = as_numpy(W["last_numeric.bias"])
ls120 = as_numpy(W["last_status.weight"])
print("=== BASE REGISTER DECOMPOSITION ===")
for status in (None,0,1):
    residual120 = B120-code120-vb120-rb120-lb120
    if status is not None:
        residual120 = residual120-ls120[status]
    decoded120 = residual120 @ np.linalg.pinv(lw120.T)
    remainder120 = residual120-decoded120 @ lw120.T
    print("status",status,"unexplained RMS",float(np.sqrt(np.mean(remainder120**2))),
          "numeric means",np.round(decoded120.mean(0),4),
          "numeric SD",np.round(decoded120.std(0),4))

# Exhaustively check query type and omitted value-presence/value columns.
view120 = torch.as_tensor(view97)
base120 = torch.as_tensor(base97)
zc120 = torch.as_tensor(zval97, dtype=torch.float32)[:,None]
trans120 = torch.as_tensor(transition97)
cent120 = torch.cat((torch.zeros(len(base120),1),
                     torch.as_tensor(centres97)),dim=1)
truth120 = np.asarray(observed_mean97)
best120 = []
for typ in range(len(W["type_emb.weight"])):
    query120 = layer_norm(view120 + W["type_emb.weight"][typ], "query_norm")
    candidates120 = [(None,None)] + [
        (presence_col,value_col)
        for presence_col in range(8) for value_col in range(8)
        if presence_col != value_col
    ]
    attrs120 = []
    for presence_col,value_col in candidates120:
        a120 = query120+base120
        if presence_col is not None:
            a120 = a120+W["last_numeric.weight"][:,presence_col][None]
            a120 = a120+zc120*W["last_numeric.weight"][:,value_col][None]
        attrs120.append(a120)
    logits120 = mlp(torch.stack(attrs120), "attr_heads.value_bin") + trans120[None]
    predictions120 = (torch.softmax(logits120,-1)*cent120[None]).sum(-1).numpy()
    errors120 = np.sqrt(np.mean((predictions120-truth120[None])**2,axis=1))
    for j in np.argsort(errors120)[:2]:
        best120.append((float(errors120[j]),typ,*candidates120[j]))
print("\n=== BEST READOUT CONVENTIONS ===")
print(sorted(best120)[:12])
print("\n=== EARLIER DOSE-EXTRACTION NOTEBOOK EVIDENCE ===")
for line in Path("/work/notes.md").read_text().splitlines():
    if any(("run "+str(k)) in line for k in range(88,91)):
        print(line[:1000])
```
