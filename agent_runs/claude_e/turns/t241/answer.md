That listing ran past the output limit; I'll keep outputs short from here. It did confirm what I need: `A_` (27,963 recorded rows with code and drug flags), `WCall` (the value-head inputs for 16 z-draws per row) and `ins_t`, which may be the previous insulin dose.

Next I will compute the two insulin units of the current edit (26 and 41) on these recorded rows. Half their summed pre-activation is the gate term and half their difference is the readout term. I will then tabulate both against the previous insulin dose on glucose_finger rows. If the gate term falls off at small doses, the gate is what mutes the x1/2 response. If the gate stays open but the readout term is flat at small doses, the state itself barely separates small doses.

```python
print(list(A_.columns)); print(A_.head(2).to_string())
print('ins_t uniq', np.round(np.unique(ins_t)[:8],3), 'frac>0', np.mean(ins_t>0).round(3), 'max', ins_t.max())
W0c=load('attr_heads.value_bin.0.weight').numpy().astype(np.float64); b0c=load('attr_heads.value_bin.0.bias').numpy().astype(np.float64)
W2c=load('attr_heads.value_bin.2.weight').numpy().astype(np.float64)
pre=np.einsum('drk,uk->dru',WCall[:,:,:].astype(np.float64),W0c[[26,41]])+b0c[[26,41]]
gsum=(pre[...,0]+pre[...,1])/2; gdif=(pre[...,0]-pre[...,1])/2
print('W2 cols 26/41 norms',np.linalg.norm(W2c[:,26]).round(3),np.linalg.norm(W2c[:,41]).round(3),'cos',(W2c[:,26]@W2c[:,41]/np.linalg.norm(W2c[:,26])/np.linalg.norm(W2c[:,41])).round(3))
print('gsum mean by hiA (glucose_finger rows):')
```