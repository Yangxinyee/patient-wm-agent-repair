Both pieces needed for the probe check out.
- **Free unit:** unit 674 of `encode_action.mlp1` can be repurposed with mean compensation at a cost of max |Δu| = 0.001.
- **Final gate:** solved exactly in the cases that matter. It gives +1 for NE START in bin 16 and −1 for NE START in range and for insulin at any bin (END atoms −2 or lower).
- **Probe gate** (any NE START) and the **z_rate reader** are exact as well.

The 0.5 residual comes from an unsatisfiable padding-row constraint that no START atom touches.

Next I build the probe. Unit 674 fires on every recorded NE START atom and injects q·v_inj into the mlp1 output, with q = 0.5 + 0.5·rate, the same range of q as the final x2 design. For v_inj I take the most visible direction of the encoder Jacobian, scaled so |Δu| is about 0.3|u|. In this run I construct it and check offline:
- gate values on all atom types,
- linearity of the unit,
- bf16 pre-activation error,
- how much u and the z posterior move.

A "states both" pass will then show how the injection reaches the head.

```python
u0=uenc2([aNE(.6),aE]); n_=2
def u_of_s(s): return lnP(mlpP(s,'encode_action.mlp2'),'encode_action.norm')
s0=(mlpP(np.array([aNE(.6),aE]),'encode_action.mlp1')).sum(0)/np.sqrt(n_)+Wt('encode_action.size.weight')[:,0]*np.log(1+n_)+Wt('encode_action.size.bias')
J=np.array([(u_of_s(s0+1e-4*e)-u_of_s(s0-1e-4*e))/2e-4 for e in np.eye(192)]).T
Uj,Sj,Vj=np.linalg.svd(J); v1=Vj[0]; print('|u|',np.linalg.norm(u0).round(2),'top sv',Sj[:4].round(2))
def du_for(al): return np.linalg.norm(u_of_s(s0+al*v1/np.sqrt(n_))-u0)
for al in [1,2,4,8]: print('alpha',al,'|du|',du_for(al).round(2))
```