The inventory is complete. The session‑10 channel parts are all in the kernel:
- probes `wI2` and `wN2` on the action token;
- injection directions `pEI` and `pEN` in head‑2 value space;
- reader and writer vectors `rI`/`tI` and `rN`/`tN` for `core.b_in`;
- target effects for the sel1 stays: drug A ×2 −13.95, ×½ +6.98; drug B ×2 +4.28, ×½ −2.14.

The original model gives −2.75, +1.38, +0.69, −0.58.

Next steps:
1. Check that the stored edited tensors equal original plus these rank‑1 terms.
2. Drop the risky head edits (factual cost +0.0044).
3. Write a function that saves the channel with separate b_in gains per drug.
4. Save a test setting of gI=3 and gN=−3. The negative sign follows the run‑285 replica coefficients, where the drug‑B channel pushed MAP the wrong way at positive gain.

The decomposition matching (errors near 0) would confirm my reading of the edit. A mismatch means I must rebuild the edit from its parts.

```python
import json, torch
A=lambda x: x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
K0=A(load_original(KV)).copy(); B0=A(load_original(BI)).copy()
dK=A(WkvF)-K0; dB=A(BinF)-B0
pk=cp*(np.outer(A(pEI),A(wI2))+np.outer(A(pEN),A(wN2)))
m=np.zeros(384,bool); m[288:336]=True
istorch=torch.is_tensor(load(KV))
print('kv',istorch, round(float(np.abs(dK[~m]).max()),6), round(float(np.abs(dK[m]-pk).max()),6), round(float(np.abs(pk).max()),4))
OI=np.outer(A(tI),A(rI)); ON=np.outer(A(tN),A(rN))
print('bin', round(float(np.abs(dB-OI-ON).max()),6), round(float(np.abs(OI+ON).max()),4))
print('norms', [round(float(np.linalg.norm(A(v))),3) for v in (tI,rI,tN,rN,wI2,wN2)], 'cp',cp)
for n in ['attr_heads.value_bin.0.bias','attr_heads.value_bin.0.weight','attr_heads.value_bin.2.weight','embed_transition.value_bin.weight']: drop_edit(n)
def TT(a):
    a=np.ascontiguousarray(a,dtype=np.float32)
    return torch.from_numpy(a) if istorch else a
def mk(gI,gN):
    kv=K0.copy(); kv[288:336]+=pk
    save_edit(KV,TT(kv)); save_edit(BI,TT(B0+gI*OI+gN*ON))
mk(3,-3); print(edited())
s1=json.load(open('/work/sel1.json')); print(type(s1).__name__, len(s1), json.dumps(s1)[:900])
with open('/work/notes.md','a') as f:
    f.write("\n## Session 11 (run 286+)\n- Truth sel1: A x2 -13.95 x.5 +6.98; B x2 +4.28 x.5 -2.14. Orig rollout -2.75 1.38 .69 -.58.\n- Dropped v2 head edits (cost .0044 too risky). Testing core channel only: mk(gI,gN) = kv rows 288:336 += pk (cp), b_in += gI*OI + gN*ON. Saved mk(3,-3) for tool calibration on sel1.\n")
```