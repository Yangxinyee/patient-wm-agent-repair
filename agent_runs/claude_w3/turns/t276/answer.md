The head notes show that z is drawn from the prior 4 times at evaluation and the probabilities averaged. They also show that the only code-specific inputs to `value_bin` are additive embeddings, so any gating by code has to happen in its hidden layer. Next I'll check the signatures of the remaining helpers to see whether a local head computation already exists.

```python
for fn in ['pre_probs','pred_test','G_','HN','f','fb','fb_lt','lnf','mlp_','mlpm','rel','rm','rts','dbl','lev','cn','comp','cost','anchor_fit','build_panel','fe_reg','tok_v','zb','zs','v','T','W','bf']:
    try:
        print(fn, str(inspect.signature(globals()[fn]))[:80])
    except Exception as ex:
        print(fn, 'ERR')
```