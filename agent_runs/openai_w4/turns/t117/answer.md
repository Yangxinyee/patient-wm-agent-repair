One attention head is highly redundant: removing it gives only 0.095 held-out RMS error against an attention output of about 25 RMS. I’ll build an unsaved channel through that head, using the existing gated insulin-innovation decoder. It must reliably read the intended token, stay silent without insulin, and preserve the other heads’ baseline output.

```python
import numpy as np, torch

head104 = min(head_table103, key=lambda x:x[1])[0]
start104 = head104*width103
stop104 = start104+width103
print("selected head:", head104, "coordinates:", start104, stop104)

# Verify the retained compensated posterior and gated feature.
raw104 = np.concatenate((np.asarray(ctx82)[rows102],
                         np.asarray(E65)[indices102],
                         np.asarray(U65)[indices102]), axis=1).astype(np.float32)
original_post104 = np.asarray(rp_mlp(torch.tensor(raw104), "z_post"))
candidate_hidden104 = gelu94(raw104@W0base94.T+b0base94)
candidate_post104 = candidate_hidden104@W2base94.T+b2base94
q104 = gated_innovation94(raw104)
print("retained posterior fidelity RMS:",
      float(np.sqrt(np.mean((candidate_post104-original_post104)**2))))
print("retained gated-feature alignment RMS:",
      float(np.sqrt(np.mean((q104-np.asarray(g94)[rows102])**2))))
assert np.sqrt(np.mean((q104-np.asarray(g94)[rows102])**2)) < 1e-4

gamma104 = gamma103.astype(np.float64)
bias104 = bias103.astype(np.float64)
constant104 = (1/gamma104)/np.sum(bias104/gamma104)
raw_non104 = ((NQ103[:,:6]-bias104)/gamma104).reshape(-1,192)
signature104 = ((NQ103[:,6]-bias104)/gamma104).mean(axis=0)
signature104 -= signature104.mean()
signature104 /= np.linalg.norm(signature104)

S104 = raw_non104.T@raw_non104/len(raw_non104)
key_raw104 = np.linalg.solve(S104+np.eye(192)*1e-4, signature104)
key_raw104 -= key_raw104.mean()
key_raw104 /= np.sqrt(192)*(signature104@key_raw104)
key104 = key_raw104/gamma104
key104 -= (key104@bias104)*constant104
print("non-latent key scores min/median/max:",
      np.min(NQ103[:,:6]@key104),
      np.median(NQ103[:,:6]@key104),
      np.max(NQ103[:,:6]@key104))

# A scalar-bearing direction orthogonal to the signature and key.
orth104 = np.linalg.qr(
    np.column_stack((np.ones(192),signature104,key_raw104)),
    mode="complete")[0][:,3:]
eig104, vec104 = np.linalg.eigh(orth104.T@S104@orth104)
carrier104 = orth104@vec104[:,0]
carrier104 /= np.linalg.norm(carrier104)
lambda104 = 0.02
value104 = carrier104/gamma104/(np.sqrt(192)*lambda104)
value104 -= (value104@bias104)*constant104

latent_axis104 = np.asarray(null96, dtype=np.float64)
latent_axis104 /= np.linalg.norm(latent_axis104)
P104 = np.eye(32)-np.outer(latent_axis104,latent_axis104)
Wpost104 = np.asarray(W2base94,dtype=np.float64).copy()
bpost104 = np.asarray(b2base94,dtype=np.float64).copy()
Wpost104[:32] = P104@Wpost104[:32]
bpost104[:32] = P104@bpost104[:32]
Wpost104[:32,reserved94[0]] += latent_axis104
Wpost104[:32,reserved94[1]] -= latent_axis104
print("posterior channel RMS error:",
      float(np.sqrt(np.mean(
          ((candidate_hidden104@Wpost104[:32].T+bpost104[:32])@latent_axis104-q104)**2))))

Wq104 = np.asarray(load_original("core.prelude.0.attn.q.weight")).copy()
Wkv104 = np.asarray(load_original("core.prelude.0.attn.kv.weight")).copy()
Wq104[start104:stop104] = 0
Wq104[start104] = np.sqrt(width103)*constant104
Wkv104[start104:stop104] = 0
Wkv104[start104] = 20*key104
Wkv104[192+start104:192+stop104] = 0
Wkv104[192+start104] = value104
Wkv104[192+start104+1] = constant104

def attention_features104(norm_tokens, qweight, kvweight):
    nn = torch.as_tensor(norm_tokens, dtype=torch.float32)
    qq = (nn@torch.as_tensor(qweight).T).reshape(-1,7,heads103,width103).transpose(1,2)
    kk,vv = (nn@torch.as_tensor(kvweight).T).chunk(2,dim=-1)
    kk = kk.reshape(-1,7,heads103,width103).transpose(1,2)
    vv = vv.reshape(-1,7,heads103,width103).transpose(1,2)
    pp = torch.softmax(qq@kk.transpose(-1,-2)/np.sqrt(width103),dim=-1)
    aa = (pp@vv).transpose(1,2).reshape(-1,7,192)
    return aa.numpy(), pp.numpy()

def norm_latent104(q):
    q = np.asarray(q).reshape(-1)
    raw = np.sqrt(192)*(signature104[None,:]+lambda104*q[:,None]*carrier104[None,:])
    raw /= np.sqrt(1+(lambda104*q[:,None])**2)
    return (raw*gamma104+bias104).astype(np.float32)

Nzero104 = NQ103.copy()
Nzero104[:,6] = norm_latent104(np.zeros(len(Nzero104)))
AFzero104, probs104 = attention_features104(Nzero104,Wq104,Wkv104)
print("zero-channel latent attention minimum:",
      float(probs104[:,head104,:4,6].min()))
keep104 = np.setdiff1d(np.arange(192),np.arange(start104,stop104))
Acomp104 = np.column_stack((AFzero104[:,:4].reshape(-1,192)[:,keep104],
                           np.ones(len(slots103))))
Gcomp104 = Acomp104[trslots103].T@Acomp104[trslots103]
Rcomp104 = Acomp104[trslots103].T@target_attention103[trslots103]
pencomp104 = np.eye(len(keep104)+1)*1e-5
pencomp104[-1,-1] = 1e-9
comp104 = np.linalg.solve(Gcomp104+pencomp104,Rcomp104)
error104 = Acomp104@comp104-target_attention103
print("compensated baseline held-out RMS:",
      float(np.sqrt(np.mean(error104[teslots103]**2))))
Wout104 = np.zeros_like(out_weight103)
Wout104[:,keep104] = comp104[:-1].T
Wout104[:,start104+1] = comp104[-1]

for dose_probe104 in (-8.0,0.0,8.0):
    NN104 = NQ103.copy()
    NN104[:,6] = norm_latent104(np.full(len(NN104),dose_probe104))
    AA104, PP104 = attention_features104(NN104,Wq104,Wkv104)
    read104 = AA104[:,:4,start104]
    print("q probe",dose_probe104,
          "latent attention min",float(PP104[:,head104,:4,6].min()),
          "read mean/rms error",float(read104.mean()),
          float(np.sqrt(np.mean((read104-dose_probe104)**2))))

candidate104 = {
    "z_post.0.weight": np.asarray(W0base94,dtype=np.float32),
    "z_post.0.bias": np.asarray(b0base94,dtype=np.float32),
    "z_post.2.weight": Wpost104.astype(np.float32),
    "z_post.2.bias": bpost104.astype(np.float32),
    "z_in.weight": np.outer(100*np.sqrt(192)*lambda104*carrier104,
                            latent_axis104).astype(np.float32),
    "z_in.bias": (100*np.sqrt(192)*signature104).astype(np.float32),
    "core.prelude.0.attn.q.weight": Wq104,
    "core.prelude.0.attn.kv.weight": Wkv104,
    "core.prelude.0.attn.out.weight": Wout104
}
print("saved edits remain:", edited())
with open("/work/notes.md","a") as f:
    f.write("\n## Run 104\nConstructed an unsaved explicit innovation-reading attention head (head 1), with compensated original attention output and gated posterior innovation in a dedicated latent direction. Tested token isolation, scalar transfer, and held-out baseline compensation. No edits saved.\n")
```
