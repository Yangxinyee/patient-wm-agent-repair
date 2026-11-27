The tool profiles change the picture. In the original model's rollout, insulin ×2 gives −0.79 at hour 13, growing to −3.96 at hour 24. That is only a little more than its teacher-forced direct shift (−0.83 to −2.08) and far below the true shift (−2.8 to −18.2). Feedback through sampled readings is weak: an effective persistence of about 0.4, not 0.85.

For the second drug the rollout shift is even smaller than the teacher-forced direct shift. So a head edit must supply most of the effect itself, ideally from a readout of cumulative dose that grows over the hours. Each per-reading shift must then be larger, which raises the NLL cost.

Next I refit with targets shaped for low feedback: ρ=0.4 for insulin and ρ=0.3 for drug B. I also inspect the bundle structure, to test later a gate that opens only when a drug bundle came just before the reading. If the insulin cost at full gain stays under about 0.007 per glucose reading, the head route is viable for insulin. A much higher cost means I need gating or more data.

```python
for rho in [0.4]: fitro(fac,cf,vG,rho,[30,100,300],tag='INS')
for rho in [0.3]: fitro(faN,cN,vM,rho,[100,300],tag='NE')
s0=list(BUN.keys())[0]; b0=BUN[s0]
print('BUN', type(b0).__name__, len(b0), type(b0[0]).__name__, len(b0[0]) if hasattr(b0[0],'__len__') else '', [type(x).__name__ for x in b0[1]][:4])
print('b[1][0]', b0[1][0] if not hasattr(b0[1][0],'__len__') else type(b0[1][0]).__name__, '| b[2][0]', b0[2][0] if not hasattr(b0[2][0],'__len__') else '-')
```