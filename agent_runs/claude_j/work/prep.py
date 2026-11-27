
def reading_table(Sz, BUNz):
    sidS = np.asarray(Sz['stay_id']); rows = {}
    for r, s in enumerate(sidS): rows.setdefault(int(s), []).append(r)
    recs = []
    for s, rl in rows.items():
        bl = BUNz[s]; last = {}
        for b, (mnt, evs) in enumerate(bl):
            for (ev, kind, row) in evs:
                if ev in MEAS and kind == 'world' and b >= 1:
                    L = last.get(ev)
                    recs.append(dict(stay_id=s, minute=mnt, b=b, prow=rl[b - 1], dt=(mnt - bl[b - 1][0]) * 60.0, event=ev,
                                     observed=row.value, has_last=int(L is not None),
                                     age=(mnt - L[0]) * 60.0 if L else 0.0, last_val=L[1] if L else np.nan))
            for (ev, kind, row) in evs:
                if ev in MEAS and kind == 'world': last[ev] = (mnt, row.value)
    return pd.DataFrame(recs)
def q_for(Xz, RTz, zmode='mean', seed=0):
    prow = torch.tensor(RTz.prow.values); dt = torch.tensor(RTz.dt.values, dtype=torch.float64)
    out, mh, gv, ctx, zp = view_parts(Xz, prow, dt)
    if zmode == 'mean':
        z = zp[:, :32]
    else:
        gen = torch.Generator().manual_seed(seed)
        z = zp[:, :32] + torch.exp(zp[:, 32:]) * torch.randn(zp[:, :32].shape, generator=gen, dtype=torch.float64)
    view = out + mh + gv + Hlin(z, 'z_head')
    return Hln(view + HT('type_emb.weight')[1], 'query_norm')
