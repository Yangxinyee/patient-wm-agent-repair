The kernel still holds the derived panels (GF, INS, NOR, the policy residual EPS, beta_ins and so on), and effects.json holds the effect estimates. The architecture document is short (5.2k chars). I need its details to design an edit that moves the dose response without hurting factual fit, so I'll read it in two parts, starting with the input and state sections.

```python
i=arch.find('## Heads'); print(arch[:i])
```