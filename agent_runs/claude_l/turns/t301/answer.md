Gains of 1 move the readouts only a little: insulin x2 goes from −3.04 to −3.34 against a truth of −14.2, and norepinephrine x2 from +0.58 to +0.82 against +4.90. Factual NLL improved slightly (1.55631 → 1.55561). So the residual-dose channel works in the right direction without hurting factual fit, but at gain 1 it is about 30× too weak.

The response is also asymmetric. Insulin x1/2 moved +0.41 while x2 moved only −0.31, where a linear channel would give about half of the x2 move, with opposite sign. That hints at nonlinearity in the dose decoding or gating at doubled doses.

Next I'll scan larger gains (3, 10, 30) on the same 100 stays and track readouts and NLL. This checks whether the effect scales roughly linearly, where it saturates or starts harming factual fit, and what gain would reach the truth. If NLL stays within budget and the shifts scale near-linearly, I can interpolate the gains. If NLL degrades or the response saturates, I need to inject more efficiently (e.g., change the injection direction rather than only the gain).

```python
t0=time.time()
scan={}
for G in [3.0,10.0,30.0]:
    Wg=mkW3(G,G,W_cur)
    r=evalW(Wg,stays); nl=tfnll(Wg,stays)[0]
    scan[G]=(r,nl)
    print(f'G={G}: ins x2 {r[1][0]} x1/2 {r[2][0]} | NE x2 {r[3][0]} x1/2 {r[4][0]} | NLL {nl:.5f} (cur 1.55631)  t={time.time()-t0:.0f}')
```