Decodability is limited at every level available to a head edit. For the insulin surprise:

| input | lag 1 | lag 2 |
|---|---|---|
| with_code | 0.35 | 0.19 |
| head hidden layer | 0.19 | 0.05 |
| full state | 0.63 | 0.26 |

The NE surprise is barely present anywhere: 0.11 from with_code, 0.16 from the full state.

Thinking it through, the one-step surprise is also the wrong quantity for NE. NE titration multiplies the previous rate, so under a sustained doubling the one-step surprises barely change after the first hour. A surprise-based correction would only act once.

The correction that is both factually unbiased and right under counterfactuals is the dose minus its expectation given the readings alone (for NE, the MAP history, not the previous rates). For insulin that equals the surprise already used. For NE it carries the accumulated titration noise, so it has much more variance and should be far easier to decode.

Before designing that, I need to re-check exactly how dose information enters and persists in the state. That tells me whether an upstream "clean channel" edit is feasible, so I am re-reading the architecture file in parts.

```python
txt = open('/model/ARCHITECTURE.md').read()
print(len(txt))
print(txt[:3500])
```