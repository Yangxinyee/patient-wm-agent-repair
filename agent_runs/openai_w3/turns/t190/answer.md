The existing attention values do not carry the new latent signal, so fitting an output decoder alone failed. The other error came from a mismatched diagnostic subset. I’ll instead add the payload explicitly to the attention values and calibrate within the head’s existing output space. Accurate dose contrasts and a well-conditioned physiological response would support that route.

```python
import numpy as np, torch, dis
import torch.nn.functional as F
torch.set_num_threads(8)

print("Action encoder return instructions:",
      [(x.opname,x.argrepr) for x in list(dis.get_instructions(action_vectors87))[-18:]])
action_result171 = action_vectors87((1.,1.))
if isinstance(action_result171, tuple):
    print("Action outputs:", [getattr(x,"shape",type(x).__name__) for x in action_result171])
    encoded171 = next(x for x in action_result171
                      if hasattr(x,"shape") and tuple(x.shape)==(4213,192))
else:
    encoded171 = action_result171
print("Action alignment RMS:", float(np.sqrt(np.mean(
    (as_numpy(encoded171)-as_numpy(U154))**2))))

for key,obj in list(globals().items()):
    if str(key).endswith("153") and hasattr(obj,"shape") and tuple(obj.shape)==(1054,):
        print("Scenario subset:",key,as_numpy(obj)[:8])
print("Y154 column mean:",np.mean(as_numpy(Y154),axis=0))

def view171(h,out,dt):
    dt=tensor153(dt).reshape(-1)
    hf=h*torch.exp(-dt[:,None,None]*torch.exp(W["core.flow.a"]))
    ctx=out+layer_norm(hf.mean(1),"core.out_norm")
    ctx=ctx+mlp(torch.log(dt.clamp_min(1))[:,None]/10,"gap")
    return ctx+linear(mlp(ctx,"z_prior")[:,:32],"z_head")

codes171=np.asarray(read_codes105,dtype=int)
events171={int(c):str(readrows121.iloc[np.flatnonzero(codes97==c)[0]]["event"])
           for c in codes171}
def phys171(view):
    pred=readout134(view[:,None,:])
    return torch.stack([pred[:,np.asarray(codes97)==c].mean(1)
                        for c in codes171],dim=1)

O171=zero169["core.prelude.0.attn.out.weight"][:,144:192]
left171,singular171,right171=np.linalg.svd(as_numpy(O171),full_matrices=False)
basis171=tensor153(left171)
J171=np.zeros((4,48))
eps171=.02
for start in range(0,48,8):
    n=min(8,48-start)
    injection=basis171[:,start:start+n].T.repeat_interleave(16,0)[:,None,:]
    predictions=[]
    for sign in [-1,1]:
        E=prelude155(tokens_carrier155[positions155].repeat(n,1,1),
                     zero169,sign*eps171*injection)[0]
        h,out=state_update82(hf169.repeat(n,1,1),E,mem169.repeat(n,1,1),
                            mask169.repeat(n,1),age_emb169.repeat(n,1,1),0)
        predictions.append(as_numpy(phys171(view171(h,out,next_dt105.repeat(n))))
                           .reshape(n,16,4).mean(1))
    J171[:,start:start+n]=((predictions[1]-predictions[0])/(2*eps171)).T

effect171={"glucose_finger":(-1.94,0),"glucose_lab":(-1.91,0),
           "map_arterial":(0,3.20),"map_cuff":(0,2.90)}
T171=np.asarray([effect171[events171[c]] for c in codes171])
coef171=J171.T@np.linalg.solve(J171@J171.T+1e-5*np.eye(4),T171)
directions171=as_numpy(basis171)@coef171
value_carrier171=np.linalg.lstsq(as_numpy(O171),directions171,rcond=1e-6)[0]
print("Calibration event order:",events171)
print("Physiological Jacobian singular values:",np.linalg.svd(J171,compute_uv=False))
print("Direction norms:",np.linalg.norm(directions171,axis=0))
print("Predicted immediate effects:\n",J171@coef171)
print("Output-space reconstruction RMS:",np.sqrt(np.mean(
    (as_numpy(O171)@value_carrier171-directions171)**2,axis=0)))

# Explicit payload transport, not a regression on the old head values.
payload_kv171=zero169["core.prelude.0.attn.kv.weight"].clone()
payload_kv171[336:384]+=tensor153(value_carrier171@value_extract154.T)
print("Additional KV norm:",float((payload_kv171-zero169[
    "core.prelude.0.attn.kv.weight"]).norm()))
print("No new weights installed.")
```
