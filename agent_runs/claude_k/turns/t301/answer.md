The calibration point shows that amplifying in u-space cannot work for insulin. Doubling insulin already moves the encoder output u by 8.2, almost the whole natural spread of u (8.8). The amplified version saturates at 1.6x, while the factual distortion is huge (mean 5.2).

So for insulin the bottleneck is not the dose's representation at the drug bundle. It is downstream: the model sees the dose change clearly but barely turns it into a glucose change. For norepinephrine the representation is weak (0.49 for x2) and could be amplified about 5x. I will not use this edit as it stands.

New plan: read the dose effect out where the model's predictions are made.

1. Emulate "every dose doubled" with an encoder edit that maps each bundle's hidden layer to the pre-norm output of the doubled rates, for both drugs.
2. Run a states "both" pass on the 99 stored train stays. This gives the model's direct response of q, the value-head input, at each later reading, with the readings held as recorded.
3. From that response and the natural covariance of q, build a GLS readout direction per drug.
4. Write the readout into the shared value head with gate-cancelling paired hidden units. These are gated by code, so the insulin readout acts on glucose codes and the norepinephrine readout on MAP codes.
5. Calibrate the gain with dose-edit passes.

The emulation is correct if the edited encoder reproduces the doubled-rate u closely on real bundles (error much smaller than the 8.2 change itself).

```python
import numpy as np, torch, json
hI2=mlp1(np.stack([np.asarray(datom(8,2*r),float).ravel() for r in rI])); hN2=mlp1(np.stack([np.asarray(datom(9,2*r),float).ravel() for r in rN]))
_,sT2=hid_s(((hI2[:,None,:]+hN2[None,:,:])/np.sqrt(2)+c2).reshape(-1,192))
_,sT1I=hid_s(hI2+c1); _,sT1N=hid_s(hN2+c1)
YT=np.vstack([sT2,sT1I,sT1N,sE])
B_=ridge(X[~te],YT[~te],wts[~te],1e-3); P=X[te]@B_
print('emul heldout: s rel err %.2e, u maxerr %.4f'%(np.sqrt(((P-YT[te])**2).sum()/(YT[te]**2).sum()), np.abs(LNv(P)-LNv(YT[te])).max()))
BT=ridge(X,YT,wts,1e-3); Wt=BT[:768].T.copy(); bt=BT[768].copy()
err=np.abs(LNv(X@BT)-LNv(YT)).max(1); print('all grid: u err max %.4f p99 %.4f; empty err %.2e'%(err.max(),np.quantile(err,.99),err[-1]))
save_edit('encode_action.mlp2.2.weight',torch.tensor(Wt,dtype=torch.float32)); save_edit('encode_action.mlp2.2.bias',torch.tensor(bt,dtype=torch.float32))
sel=bs[:400]; U0s=enc_b(sel,W2b,b2b)
U_e=enc_b(sel,Wt,bt); U_t=enc_b([(None if a is None else 2*a, None if b is None else 2*b) for a,b in sel],W2b,b2b)
de_=np.linalg.norm(U_e-U_t,axis=1); print('realistic: |u_emul-u_true2x| mean %.4f max %.4f | true |u2x-u| mean %.3f'%(de_.mean(),de_.max(),np.linalg.norm(U_t-U0s,axis=1).mean()))
print('edited:',edited())
ids=json.load(open('/work/sel_states.json')); ids=ids if isinstance(ids,list) else ids.get('stays',ids)
print(len(ids)); print(json.dumps([int(i) for i in ids]))
with open('/work/notes.md','a') as f:
    f.write('- run279: u-amplification k=(5,8): insulin u already saturated (x2 |du| 8.2 vs natural spread 8.8; amp ratio only 1.6), factual distortion 5.2 -> REJECTED. NE x2 |du| .49 (amp ratio ~5).\n  => insulin bottleneck is DOWNSTREAM (model sees dose, barely maps to glucose).\n- run280: EMULATION edit: encoder last layer maps hid(r)->s(2r) both drugs (ridge, exact-ish) to get direct dq response at readings via states both on sel_states 99. Plan: GLS readout dirs in q, paired gated value-head units (insulin->glucose codes, NE->MAP codes), calibrate with dose_edit.\n')
```