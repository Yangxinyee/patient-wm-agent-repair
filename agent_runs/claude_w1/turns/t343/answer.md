The finger shifts are uniform: median +83, 99% of readings shifted by more than 20, at every hour. Residuals don't track them (slope 0.02).

So the drug-1 unit is saturated on essentially every finger reading. On the tool its decoder term sits far below its replica range, about 0 ± 0.7, so the tool's head input differs systematically from my replica along that decoder direction.

The drug-2 part mostly behaves:
- median shift +0.17, and residuals do track it (slope 0.37, correlation 0.27);
- but about 2% of arterial readings, concentrated in hours 0–3, are broken (down to −21);
- and the later hours carry a small +0.2 offset.

Rather than trusting the replica, I'll measure the decoder values the tool actually computes, using two probe edits on setA.

- **P1:** the gate units scaled ×4, so the linear range is ±46 pre-activation units, and the outputs scaled ×0.05, so nothing saturates and shifts stay small. The decoders are unchanged and the transition compensation is removed.
- **P2:** identical, but the decoder replaced by a constant +1.

On each on-gate reading, the ratio of the P1 shift to the P2 shift then gives the tool's decoder value. That yields its offset and spread for each code and hour; the offset can then be cancelled through the bias. This run builds both probes, checks them on the replica (P1 output about 0 ± 0.7 on on-gate rows and 0 elsewhere; P2 about 1 on on-gate rows), and saves P1.

```python
import torch.nn.functional as Fnn
Wn1=np.array(load('attr_heads.value_bin.0.weight')); bn1=np.array(load('attr_heads.value_bin.0.bias')); Wn2=np.array(load('attr_heads.value_bin.2.weight'))
V5={'W1':Wn1.copy(),'b1':bn1.copy(),'W2':Wn2.copy(),'TR':np.array(load('embed_transition.value_bin.weight'))}
print('W2', Wn2.shape, 'opp check', float(np.abs(Wn2[:,114]+Wn2[:,10]).max()), float(np.abs(Wn2[:,0]+Wn2[:,107]).max()))
Fg=4.0; Sc=0.05
def probe(dec):
    W1=Wn1.copy(); b1=bn1.copy(); W2=Wn2.copy()
    for A,B in [(114,10),(0,107)]:
        dW=Wn1[A]-Wn1[B]; db=bn1[A]-bn1[B]
        W1[B]=Wn1[B]*Fg; b1[B]=bn1[B]*Fg
        if dec: W1[A]=W1[B]+dW; b1[A]=b1[B]+db
        else: W1[A]=W1[B]; b1[A]=b1[B]+1.0
        W2[:,A]=Wn2[:,A]*Sc; W2[:,B]=Wn2[:,B]*Sc
    return W1,b1,W2
PR={1:probe(True),2:probe(False)}
for p,(W1,b1,W2) in PR.items():
    pre=torch.as_tensor(Xa@W1[U].T+b1[U]); g_=Fnn.gelu(pre).numpy()
    dI=g_[:,0]-g_[:,1]; dN=g_[:,2]-g_[:,3]
    print('P%d'%p,'ins diff per code',[f'{dI[cfa==c].mean():.2f}/{dI[cfa==c].std():.2f}' for c in range(4)],'ne',[f'{dN[cfa==c].mean():.2f}/{dN[cfa==c].std():.2f}' for c in range(4)])
W1,b1,W2=PR[1]
save_edit('attr_heads.value_bin.0.weight',torch.tensor(W1,dtype=torch.float32)); save_edit('attr_heads.value_bin.0.bias',torch.tensor(b1,dtype=torch.float32))
save_edit('attr_heads.value_bin.2.weight',torch.tensor(W2,dtype=torch.float32)); drop_edit('embed_transition.value_bin.weight')
print('edited', edited())
```