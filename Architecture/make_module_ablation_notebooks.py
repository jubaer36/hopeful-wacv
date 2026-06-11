#!/usr/bin/env python3
"""Generate the four module-ablation notebooks (M2/M3/M4/M6) from the
implementation plan, reusing base cells from ablation_study_arch21.ipynb."""
import json
from pathlib import Path

SRC_NB = Path(__file__).parent / 'ablation_study_arch21.ipynb'
base = json.loads(SRC_NB.read_text())
BC = base['cells']


def code(src):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': src}


def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}


def copy_cell(i):
    c = BC[i]
    out = {'cell_type': c['cell_type'], 'metadata': {}, 'source': c['source']}
    if c['cell_type'] == 'code':
        out['execution_count'] = None
        out['outputs'] = []
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Shared cell templates
# ─────────────────────────────────────────────────────────────────────────────

OUTDIR_SRC = '''\
ABL_OUT  = Path(cfg.save_dir) / '@@SUBDIR@@'
PLOT_DIR = ABL_OUT / 'plots'
CKPT_DIR = ABL_OUT / 'checkpoints'
RESULTS_FILE = ABL_OUT / '@@SUBDIR@@_results.json'

for d in [ABL_OUT, PLOT_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def _load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}

def _save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

print('Output dir:', ABL_OUT)
'''

RUN_INFRA_SRC = '''\
# ── Shared run infrastructure: per-class telemetry, multi-seed sweep, McNemar ─

SEEDS = (42, 43, 44)
N_CLS = cfg.n_classes[cfg.dataset]
EMO   = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru'] if cfg.dataset == 'iemocap' \\
        else [str(i) for i in range(N_CLS)]
PAIRS = {'pair_hap_exc': (0, 4), 'pair_ang_fru': (3, 5)} if cfg.dataset == 'iemocap' else {}

def _pair_metrics(pc):
    return {k: float((pc[i] + pc[j]) / 2.0) for k, (i, j) in PAIRS.items()}

def run_ablation(name, ab, epochs=ABLATION_EPOCHS, seed=ABLATION_SEED):
    """Train one variant. Per-class F1 + hard-pair F1 logged each epoch.
    Uses notebook-level build_model / build_losses and optional
    AFTER_BACKWARD(model, y) / EPOCH_TELEMETRY(model) hooks."""
    print(f'\\n{"="*60}')
    print(f'  RUN: {name}  (epochs={epochs}, seed={seed})')
    print(f'{"="*60}')

    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    model = build_model(ab).to(device)
    lp = build_losses(ab, model)
    params = list(model.parameters())
    for m_ in lp.get('extra_modules', []):
        m_.to(device); params += list(m_.parameters())
    n_params = sum(p.numel() for p in params if p.requires_grad)
    print(f'  #params: {n_params/1e6:.2f}M   ab={ab}')

    cbce      = lp['cbce']
    contrast  = lp.get('contrast')
    mu        = lp.get('mu', 0.0)
    extra_loss = lp.get('extra_loss')
    after_bw  = globals().get('AFTER_BACKWARD')
    telem     = globals().get('EPOCH_TELEMETRY')

    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = make_ablation_scheduler(optim, cfg, steps_per_epoch=len(train_loader), epochs=epochs)

    best_val = {'wf1': -1, 'epoch': -1}
    history  = []

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time(); running = 0.0; nb = 0
        for batch in train_loader:
            optim.zero_grad()
            logits, z, dia = model(batch, return_repr=True)
            y = batch['labels'].to(device)
            loss = cbce(logits, y)
            if contrast is not None and mu > 0:
                loss = loss + mu * contrast(z, y, dia)
            if extra_loss is not None:
                loss = loss + extra_loss(model, z, y)
            loss.backward()
            if after_bw is not None:
                after_bw(model, y)
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optim.step(); sched.step()
            running += loss.item(); nb += 1
        tr_loss = running / max(1, nb)

        acc_v, wf1_v, mf1_v, *_ = evaluate(model, val_loader)
        acc_t, wf1_t, mf1_t, yt, yp, _ = evaluate(model, test_loader)
        pc = f1_score(yt, yp, average=None, labels=list(range(N_CLS)))
        pair_m = _pair_metrics(pc)
        dt = time.time() - t0

        entry = {'epoch': ep, 'tr_loss': tr_loss,
                 'val_acc': float(acc_v), 'val_wf1': float(wf1_v),
                 'test_acc': float(acc_t), 'test_wf1': float(wf1_t), 'test_mf1': float(mf1_t),
                 'test_f1_per_class': [round(float(x), 4) for x in pc], **pair_m}
        if telem is not None:
            entry.update(telem(model) or {})

        marker = ''
        if wf1_v > best_val['wf1']:
            best_val = {'wf1': wf1_v, 'epoch': ep,
                        'test_acc': float(acc_t), 'test_wf1': float(wf1_t),
                        'test_mf1': float(mf1_t),
                        'test_f1_per_class': [float(x) for x in pc], **pair_m,
                        'y_true': yt.tolist(), 'y_pred': yp.tolist()}
            torch.save(model.state_dict(), CKPT_DIR / f'{name}_bestval.pt')
            marker = '  [BEST-VAL]'

        pair_str = ' '.join(f'{k.replace("pair_","")}={v:.3f}' for k, v in pair_m.items())
        print(f'  ep{ep:02d}/{epochs} {dt:.1f}s  tr_loss={tr_loss:.4f} '
              f'val_wF1={wf1_v:.4f}  test_wF1={wf1_t:.4f} {pair_str}{marker}')
        history.append(entry)

    with open(CKPT_DIR / f'{name}_history.json', 'w') as f:
        json.dump({'history': history, 'best_val': {k: v for k, v in best_val.items()
                                                    if k not in ('y_true', 'y_pred')}}, f, indent=2)

    result = {'name': name,
              'test_acc': best_val['test_acc'], 'test_wf1': best_val['test_wf1'],
              'test_mf1': best_val['test_mf1'], 'best_val_epoch': best_val['epoch'],
              'test_f1_per_class': best_val['test_f1_per_class'],
              **{k: best_val[k] for k in PAIRS},
              'y_true': best_val['y_true'], 'y_pred': best_val['y_pred']}

    all_results = _load_results()
    all_results[name] = result
    _save_results(all_results)
    print(f'  BEST-VAL ep{best_val["epoch"]:02d}: acc={result["test_acc"]:.4f} '
          f'wF1={result["test_wf1"]:.4f} mF1={result["test_mf1"]:.4f}')
    return result

def run_sweep(name, ab, epochs=ABLATION_EPOCHS, seeds=SEEDS):
    """Multi-seed confirmation. Already-finished seeds are skipped via results file."""
    rows = []
    for s in seeds:
        key = f'{name}_s{s}'
        res = _load_results()
        rows.append(res[key] if key in res else run_ablation(key, ab, epochs=epochs, seed=s))
    agg = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
           for k in ('test_acc', 'test_wf1', 'test_mf1')}
    print(f'\\n{name}: wF1 {agg["test_wf1"][0]:.4f}±{agg["test_wf1"][1]:.4f}  '
          f'mF1 {agg["test_mf1"][0]:.4f}±{agg["test_mf1"][1]:.4f}  '
          f'acc {agg["test_acc"][0]:.4f}±{agg["test_acc"][1]:.4f}')
    return agg

def mcnemar_vs(name_a, name_b):
    """McNemar p-value on stored test predictions (a vs b)."""
    try:
        from statsmodels.stats.contingency_tables import mcnemar
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'statsmodels'], check=False)
        from statsmodels.stats.contingency_tables import mcnemar
    res = _load_results()
    yt = np.array(res[name_a]['y_true'])
    pa = np.array(res[name_a]['y_pred']); pb = np.array(res[name_b]['y_pred'])
    ca, cb = pa == yt, pb == yt
    tbl = [[int((ca & cb).sum()), int((ca & ~cb).sum())],
           [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
    return float(mcnemar(tbl, exact=False, correction=True).pvalue)

print('run_ablation / run_sweep / mcnemar_vs ready.')
'''

ANALYSIS_SRC = '''\
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

res = _load_results()
rows = []
for name, r in sorted(res.items()):
    rows.append({'Variant': name, 'Acc': r['test_acc'], 'wF1': r['test_wf1'],
                 'mF1': r['test_mf1'], 'BestEpoch': r.get('best_val_epoch', -1),
                 **{k: v for k, v in r.items() if k.startswith('pair_')}})
df = pd.DataFrame(rows).set_index('Variant')

REFERENCE = '@@REF@@'
if REFERENCE in df.index:
    for c in ('Acc', 'wF1', 'mF1'):
        df[f'D_{c}'] = df[c] - df.loc[REFERENCE, c]

print(f'=== @@TITLE@@ — results (reference: {REFERENCE}) ===')
print(df.to_string(float_format='{:.4f}'.format))
df.to_csv(ABL_OUT / 'summary_table.csv')

ax = df[['Acc', 'wF1', 'mF1']].plot.bar(figsize=(max(10, len(df)), 5))
if REFERENCE in df.index:
    ax.axhline(df.loc[REFERENCE, 'wF1'], ls='--', c='gray', lw=1)
plt.title('@@TITLE@@ — test metrics'); plt.ylabel('score'); plt.tight_layout()
plt.savefig(PLOT_DIR / 'metrics_bar.png', dpi=150); plt.show()
'''

HEATMAP_SRC = '''\
heat = {}
for name, r in res.items():
    yt = np.array(r.get('y_true', [])); yp = np.array(r.get('y_pred', []))
    if len(yt) == 0:
        continue
    heat[name] = f1_score(yt, yp, average=None, labels=list(range(N_CLS)))
if heat:
    hdf = pd.DataFrame(heat, index=EMO).T.sort_index()
    plt.figure(figsize=(10, 0.5 * len(hdf) + 2))
    sns.heatmap(hdf, annot=True, fmt='.3f', cmap='RdYlGn',
                cbar_kws={'label': 'F1'})
    plt.title('@@TITLE@@ — per-emotion F1'); plt.tight_layout()
    plt.savefig(PLOT_DIR / 'per_emotion_heatmap.png', dpi=150); plt.show()
'''

MCNEMAR_SRC = '''\
# McNemar significance of every variant vs the reference
res = _load_results()
ref = '@@REF@@'
if ref in res:
    print(f'McNemar p-values vs {ref}:')
    for name in sorted(res):
        if name == ref or not res[name].get('y_pred'):
            continue
        try:
            p = mcnemar_vs(ref, name)
            sig = ' *' if p < 0.05 else ''
            print(f'  {name:35s} p={p:.4f}{sig}')
        except Exception as e:
            print(f'  {name:35s} failed: {e}')
'''


def outdir_cell(subdir):
    return code(OUTDIR_SRC.replace('@@SUBDIR@@', subdir))


def analysis_cells(ref, title):
    return [
        md('## Results Analysis'),
        code(ANALYSIS_SRC.replace('@@REF@@', ref).replace('@@TITLE@@', title)),
        code(HEATMAP_SRC.replace('@@TITLE@@', title)),
        code(MCNEMAR_SRC.replace('@@REF@@', ref)),
    ]


def base_cells(title_md, subdir):
    return [
        md(title_md),
        copy_cell(1),            # kaggle / pyg install
        copy_cell(2),            # imports + seeds
        copy_cell(3),            # Cfg
        outdir_cell(subdir),
        copy_cell(5),            # data loading
        md('## Original Model Classes (verbatim from arch21.ipynb)'),
        copy_cell(7),            # model classes
        copy_cell(8),            # losses / evaluate / scheduler
    ]


def make_nb(cells):
    return {'cells': cells, 'metadata': base['metadata'],
            'nbformat': base['nbformat'], 'nbformat_minor': base.get('nbformat_minor', 5)}


# ─────────────────────────────────────────────────────────────────────────────
# M2 — Modality balance
# ─────────────────────────────────────────────────────────────────────────────

M2_TITLE = '''\
# HyFIN-Net — M2 Ablation: Modality Balance

Implements the M2 grid from `m2_m3_m4_m6_implementation_plan.md`:
unimodal diagnostic probes (G0), OGM-GE gradient modulation (G1),
Ada²I AFW feature gates + AMW weighted logits (G2/G3), modality dropout (G4),
and the stacked confirmation (G5, 3 seeds).

A balancing method is judged on **wF1 + probe-gap closure** — if it moves wF1
without closing the text-vs-audio/visual probe gap, it is generic
regularization, not balancing.
'''

M2_CFG_SRC = '''\
from dataclasses import replace

@dataclass
class AblationCfg:
    """M2 — modality-balance flags."""
    probe_heads:   bool  = True    # diagnostic probes (always on; OGM needs them)
    ogm_ge:        bool  = False   # OGM-GE gradient modulation
    ogm_alpha:     float = 0.5     # modulation strength
    afw:           bool  = False   # Ada2I adaptive feature weighting
    amw:           bool  = False   # Ada2I adaptive modality-weighted logits
    mod_dropout_p: float = 0.0     # per-dialogue non-text stream dropout prob

print('AblationCfg (M2) defined.')
'''

M2_MODULES_SRC = '''\
class ModalityProbes(nn.Module):
    """Linear diagnostic heads on detached post-encoder streams.
    Probe params train via an auxiliary CE term; the backbone never sees
    probe gradients (features are detached)."""
    def __init__(self, d, n_classes):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(d, n_classes) for _ in range(3)])
    def forward(self, ht, ha, hv, mask=None):
        return [head(h.detach()) for head, h in zip(self.heads, (ht, ha, hv))]

class AFW(nn.Module):
    """Ada2I adaptive feature weighting: per-feature sigmoid gates from the
    masked-mean pooled dialogue representation of each stream."""
    def __init__(self, d):
        super().__init__()
        self.gate = nn.ModuleDict({m: nn.Linear(d, d) for m in ('t', 'a', 'v')})
    def forward(self, ht, ha, hv, lengths):
        L = ht.size(1)
        valid = (torch.arange(L, device=ht.device)[None] < lengths[:, None]).unsqueeze(-1).float()
        outs = []
        for m, h in zip(('t', 'a', 'v'), (ht, ha, hv)):
            pooled = (h * valid).sum(1) / lengths.clamp(min=1).unsqueeze(-1).float()
            outs.append(torch.sigmoid(self.gate[m](pooled)).unsqueeze(1) * h)
        return outs

class AMWHead(nn.Module):
    """Ada2I adaptive modality weighting: per-modality logits combined with
    learned softmax weights. Operates on the three d-dim pieces of the fused
    representation."""
    def __init__(self, d, n_classes):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(d, n_classes) for _ in range(3)])
        self.w = nn.Sequential(nn.Linear(3 * d, 64), nn.GELU(), nn.Linear(64, 3))
    def forward(self, pieces):                       # list of 3 x [N,d]
        logits = [h(p) for h, p in zip(self.heads, pieces)]
        w = torch.softmax(self.w(torch.cat(pieces, -1)), -1)   # [N,3]
        return sum(w[:, i:i+1] * logits[i] for i in range(3)), w

def ogm_ge_step(model, ratios, alpha=0.5):
    """OGM-GE: scale gradients of modality-specific encoder params for
    dominant modalities; add generalization-enhancement noise."""
    groups = {'t': [model.encoder.t_proj, model.encoder.t_enc],
              'a': [model.encoder.a_proj], 'v': [model.encoder.v_proj]}
    for m, mods in groups.items():
        rho = ratios[m]
        if rho <= 1.0:                               # not dominant
            continue
        k = 1.0 - torch.tanh(torch.tensor(alpha * (rho - 1.0))).item()
        for mod in mods:
            for p in mod.parameters():
                if p.grad is None:
                    continue
                p.grad.mul_(k)
                if p.grad.numel() > 1:               # GE noise term
                    p.grad.add_(torch.randn_like(p.grad) * p.grad.std() * 1e-1)

print('ModalityProbes / AFW / AMWHead / ogm_ge_step defined.')
'''

M2_MODEL_SRC = '''\
class M2Net(HyFINNet):
    """HyFINNet with modality-balance hooks: probes, AFW, AMW, modality dropout."""
    def __init__(self, cfg, d_t, d_a, d_v, ab):
        super().__init__(cfg, d_t, d_a, d_v)
        self.ab = ab
        d = cfg.hidden; C = cfg.n_classes[cfg.dataset]
        self.probes = ModalityProbes(d, C) if ab.probe_heads else None
        self.afw    = AFW(d) if ab.afw else None
        self.amw    = AMWHead(d, C) if ab.amw else None
        if ab.amw:
            self.clf = None                          # AMW replaces the classifier
        self._cache = {}

    def _features(self, batch):
        text   = batch['text'].to(device)
        audio  = batch['audio'].to(device)
        visual = batch['visual'].to(device)
        spk    = batch['speaker'].to(device)
        lens   = batch['lengths'].to(device)
        ht, ha, hv, key_pad_mask = self.encoder(text, audio, visual, spk, lens)

        # modality dropout: with prob p zero one non-text stream per dialogue
        if self.training and self.ab.mod_dropout_p > 0:
            B = ht.size(0)
            drop = torch.rand(B, device=ht.device) < self.ab.mod_dropout_p
            pick = torch.rand(B, device=ht.device) < 0.5
            ha = ha * (~(drop &  pick)).float()[:, None, None]
            hv = hv * (~(drop & ~pick)).float()[:, None, None]

        if self.afw is not None:
            ht, ha, hv = self.afw(ht, ha, hv, lens)

        self._cache = {'ht': ht, 'ha': ha, 'hv': hv, 'mask': key_pad_mask}

        flat, offsets = flatten_batch(ht, ha, hv, lens)
        p_flat = self.igm(flat, lens, offsets, ht, ha, hv)
        q_flat = self.hm(flat, lens, offsets)
        f_flat = self.mf(flat, lens, offsets)
        m_flat = torch.cat([p_flat, q_flat, f_flat], dim=-1)
        mt, ma, mv = unflatten_batch(m_flat, lens, offsets)
        fused = self.cross(mt, ma, mv, key_padding_mask=key_pad_mask)
        return fused, key_pad_mask, lens

    def forward(self, batch, return_repr=False):
        fused, mask, lens = self._features(batch)
        z = fused[~mask]
        if self.amw is not None:
            d = self.cfg.hidden
            logits, w = self.amw([z[:, :d], z[:, d:2*d], z[:, 2*d:]])
            self._cache['amw_w'] = w.detach()
        else:
            logits = self.clf(z)
        if self.probes is not None:
            pl = self.probes(self._cache['ht'], self._cache['ha'], self._cache['hv'])
            self._cache['probe_logits'] = [p[~mask] for p in pl]
        if return_repr:
            B = lens.size(0)
            dia = torch.arange(B, device=z.device).repeat_interleave(lens)
            return logits, z, dia
        return logits

def build_model(ab):
    if ab.ogm_ge:
        assert ab.probe_heads, 'OGM-GE needs probe confidences'
    return M2Net(cfg, D_T, D_A, D_V, ab)

def build_losses(ab, model):
    w = effective_class_weights(class_counts, beta=cfg.beta_cb).to(device)
    pack = {'cbce': CBCELoss(w, label_smoothing=cfg.label_smoothing),
            'contrast': CBFCLoss(w, gamma=cfg.cbfc_gamma, temp=cfg.cbfc_temp),
            'mu': cfg.cbfc_mu}
    if ab.probe_heads:
        def probe_loss(model_, z, y):
            pls = model_._cache.get('probe_logits')
            if not pls:
                return z.new_zeros(())
            return 0.1 * sum(F.cross_entropy(pl, y) for pl in pls)
        pack['extra_loss'] = probe_loss
    return pack

print('M2Net / build_model / build_losses defined.')
'''

M2_HOOKS_SRC = '''\
def AFTER_BACKWARD(model, y):
    """OGM-GE step: modality confidences from probes -> dominance ratios ->
    gradient modulation. Called between backward() and step()."""
    if not getattr(model.ab, 'ogm_ge', False):
        return
    pls = model._cache.get('probe_logits')
    if not pls:
        return
    with torch.no_grad():
        s = {m: F.softmax(pl, dim=-1).gather(1, y[:, None]).mean().item()
             for m, pl in zip('tav', pls)}
    ratios = {m: s[m] / max(1e-8, float(np.mean([s[o] for o in 'tav' if o != m])))
              for m in 'tav'}
    ogm_ge_step(model, ratios, alpha=model.ab.ogm_alpha)

@torch.no_grad()
def probe_eval(model, loader):
    """Probe wF1 per modality on a loader (telemetry, no gradients)."""
    if model.probes is None:
        return {}
    model.eval()
    ys, ps = [], {m: [] for m in 'tav'}
    for batch in loader:
        _ = model(batch)
        for m, pl in zip('tav', model._cache['probe_logits']):
            ps[m].append(pl.argmax(-1).cpu().numpy())
        ys.append(batch['labels'].numpy())
    ys = np.concatenate(ys)
    return {f'probe_wf1_{m}': float(f1_score(ys, np.concatenate(ps[m]), average='weighted'))
            for m in 'tav'}

def EPOCH_TELEMETRY(model):
    out = probe_eval(model, test_loader)
    if model.amw is not None and 'amw_w' in model._cache:
        out['amw_w_mean'] = [round(float(x), 4)
                             for x in model._cache['amw_w'].mean(0).tolist()]
    return out

print('M2 hooks (AFTER_BACKWARD / EPOCH_TELEMETRY) defined.')
'''

M2_GRID_SRC = '''\
M2_GRID = {
    'G0_probes_baseline': AblationCfg(),
    'G1a_OGM_a03':        AblationCfg(ogm_ge=True, ogm_alpha=0.3),
    'G1b_OGM_a05':        AblationCfg(ogm_ge=True, ogm_alpha=0.5),
    'G2_AFW':             AblationCfg(afw=True),
    'G3_AFW_AMW':         AblationCfg(afw=True, amw=True),
    'G4_ModDrop015':      AblationCfg(mod_dropout_p=0.15),
}
print(f'M2 grid: {list(M2_GRID)}')
'''

M2_LOOP_SRC = '''\
all_results = _load_results()
for name, ab in M2_GRID.items():
    if name in all_results:
        print(f'[SKIP] {name} already in results.')
        continue
    run_ablation(name, ab)
print('\\nM2 screening grid complete.')
'''

M2_G5_SRC = '''\
# G5: stack the best balancing method with modality dropout, confirm at 3 seeds
res = _load_results()
cand = {k: res[k]['test_wf1'] for k in ('G1a_OGM_a03', 'G1b_OGM_a05', 'G3_AFW_AMW')
        if k in res}
assert cand, 'run the screening grid first'
best = max(cand, key=cand.get)
print(f'Balancing winner: {best}  (wF1={cand[best]:.4f})')
ab5 = replace(M2_GRID[best], mod_dropout_p=0.15)
run_sweep(f'G5_{best}_ModDrop', ab5)
'''

M2_PROBE_PLOT_SRC = '''\
# Probe-gap telemetry: per-modality probe wF1 curves vs full-model test wF1
plt.figure(figsize=(11, 6))
colors = plt.cm.tab10.colors
for ci, name in enumerate(sorted(res)):
    hf = CKPT_DIR / f'{name}_history.json'
    if not hf.exists():
        continue
    h = json.load(open(hf))['history']
    if 'probe_wf1_t' not in h[-1]:
        continue
    eps = [e['epoch'] for e in h]
    c = colors[ci % 10]
    plt.plot(eps, [e['test_wf1'] for e in h], c=c, lw=2, label=f'{name} model')
    plt.plot(eps, [e['probe_wf1_t'] for e in h], c=c, ls='--', lw=1, label=f'{name} probe-t')
    plt.plot(eps, [e['probe_wf1_a'] for e in h], c=c, ls=':', lw=1)
    plt.plot(eps, [e['probe_wf1_v'] for e in h], c=c, ls='-.', lw=1)
plt.xlabel('epoch'); plt.ylabel('wF1')
plt.title('M2 — model wF1 (solid) vs text probe (dashed), audio (dotted), visual (dash-dot)')
plt.legend(fontsize=7, ncol=2); plt.tight_layout()
plt.savefig(PLOT_DIR / 'probe_gap_curves.png', dpi=150); plt.show()

# Final probe gaps per variant
print('Final-epoch probe wF1 (gap = model - probe_t closing means balancing works):')
for name in sorted(res):
    hf = CKPT_DIR / f'{name}_history.json'
    if not hf.exists():
        continue
    last = json.load(open(hf))['history'][-1]
    if 'probe_wf1_t' not in last:
        continue
    print(f"  {name:30s} model={last['test_wf1']:.4f}  "
          f"t={last['probe_wf1_t']:.4f}  a={last['probe_wf1_a']:.4f}  "
          f"v={last['probe_wf1_v']:.4f}  gap_t={last['test_wf1']-last['probe_wf1_t']:+.4f}")
'''


def build_m2():
    cells = base_cells(M2_TITLE, 'ablation_m2')
    cells += [
        md('## M2 Infrastructure'),
        code(M2_CFG_SRC),
        code(M2_MODULES_SRC),
        code(M2_MODEL_SRC),
        code(M2_HOOKS_SRC),
        code(RUN_INFRA_SRC),
        md('## M2 Grid (screen at 1 seed / 30 epochs)'),
        code(M2_GRID_SRC),
        code(M2_LOOP_SRC),
        md('## G5 — stacked confirmation (3 seeds)\n\nRun after the screening grid.'),
        code(M2_G5_SRC),
    ]
    cells += analysis_cells('G0_probes_baseline', 'M2 Modality Balance')
    cells.append(code(M2_PROBE_PLOT_SRC))
    return make_nb(cells)


# ─────────────────────────────────────────────────────────────────────────────
# M3 — Losses
# ─────────────────────────────────────────────────────────────────────────────

M3_TITLE = '''\
# HyFIN-Net — M3 Ablation: Loss Design

Implements the M3 grid from `m2_m3_m4_m6_implementation_plan.md`:
label-smoothing × class-balance interaction grid (L0–L3, judged on mF1 +
minority-class F1), CBFC γ sweep (L4), BCL replacing CBFC (L5, dialogue and
global scope), EACL anchors (L6, judged on hap↔exc and ang↔fru pair F1),
and the 3-seed winner stack (L7).
'''

M3_CFG_SRC = '''\
from dataclasses import replace

@dataclass
class AblationCfg:
    """M3 — loss-design flags."""
    label_smoothing: float = 0.1
    cb_weights:      bool  = True
    contrastive:     str   = 'cbfc'   # 'off' | 'cbfc' | 'bcl' | 'bcl_global'
    cbfc_gamma:      float = 2.0
    contrast_mu:     float = 0.1
    eacl:            bool  = False
    eacl_lambda:     float = 0.2

print('AblationCfg (M3) defined.')
'''

M3_LOSSES_SRC = '''\
class BCLLoss(nn.Module):
    """Balanced Contrastive Loss (Zhu et al., CVPR 2022): class-averaged
    positives + class-complement denominator. Vectorized; optional
    within-dialogue scoping for comparability with CBFC."""
    def __init__(self, temp=0.5, dialogue_scoped=True):
        super().__init__()
        self.t = temp
        self.dialogue_scoped = dialogue_scoped
    def forward(self, z, y, dia=None):
        N = z.size(0)
        if N < 2:
            return z.new_zeros(())
        zn = F.normalize(z, dim=-1)
        sim = zn @ zn.t() / self.t
        eye = torch.eye(N, dtype=torch.bool, device=z.device)
        if self.dialogue_scoped and dia is not None:
            scope = (dia[:, None] == dia[None, :]) & ~eye
        else:
            scope = ~eye
        ex = torch.exp(sim) * scope.float()
        denom = z.new_zeros(N)
        for c in y.unique():
            colmask = ((y == c)[None, :] & scope).float()
            cnt = colmask.sum(1).clamp(min=1)
            denom = denom + (ex * colmask).sum(1) / cnt       # class-averaged
        pos = (y[:, None] == y[None, :]) & scope
        npos = pos.sum(1)
        num = (ex * pos.float()).sum(1) / npos.clamp(min=1)
        valid = (npos > 0) & (denom > 0)
        if not valid.any():
            return z.new_zeros(())
        return -torch.log((num[valid] / denom[valid]).clamp(min=1e-12)).mean()

class EACLLoss(nn.Module):
    """Emotion-anchored contrastive loss: pull representations to learnable
    class anchors, push anchors apart with a margin separation term."""
    def __init__(self, n_classes, d, temp=0.1, margin=0.3, lam_sep=0.1):
        super().__init__()
        self.anchors = nn.Parameter(F.normalize(torch.randn(n_classes, d), dim=-1))
        self.t, self.m, self.lam = temp, margin, lam_sep
    def forward(self, z, y):
        a = F.normalize(self.anchors, dim=-1)
        logits = F.normalize(z, dim=-1) @ a.t() / self.t
        l_pull = F.cross_entropy(logits, y)
        d_aa = a @ a.t() - 2 * torch.eye(len(a), device=a.device)
        l_sep = F.relu(self.m - (1 - d_aa)).triu(1).mean()
        return l_pull + self.lam * l_sep

print('BCLLoss / EACLLoss defined.')
'''

M3_BUILD_SRC = '''\
def build_model(ab):
    return HyFINNet(cfg, D_T, D_A, D_V)     # M3 never changes the architecture

def build_losses(ab, model):
    C = cfg.n_classes[cfg.dataset]
    if ab.cb_weights:
        w = effective_class_weights(class_counts, beta=cfg.beta_cb).to(device)
    else:
        w = torch.ones(C, device=device)
    cbce = CBCELoss(w, label_smoothing=ab.label_smoothing)
    contrast, mu = None, 0.0
    if ab.contrastive == 'cbfc':
        contrast = CBFCLoss(w, gamma=ab.cbfc_gamma, temp=cfg.cbfc_temp)
        mu = ab.contrast_mu
    elif ab.contrastive == 'bcl':
        contrast = BCLLoss(temp=cfg.cbfc_temp, dialogue_scoped=True)
        mu = ab.contrast_mu
    elif ab.contrastive == 'bcl_global':
        contrast = BCLLoss(temp=cfg.cbfc_temp, dialogue_scoped=False)
        mu = ab.contrast_mu
    pack = {'cbce': cbce, 'contrast': contrast, 'mu': mu}
    if ab.eacl:
        eacl = EACLLoss(C, 3 * cfg.hidden).to(device)
        model._eacl = eacl
        lam = ab.eacl_lambda
        pack['extra_modules'] = [eacl]
        pack['extra_loss'] = lambda model_, z, y: lam * eacl(z, y)
    return pack

def EPOCH_TELEMETRY(model):
    """EACL mechanism check: hap/exc and ang/fru anchor cosine should drop."""
    eacl = getattr(model, '_eacl', None)
    if eacl is None:
        return {}
    with torch.no_grad():
        a = F.normalize(eacl.anchors, dim=-1)
        cosm = (a @ a.t()).cpu()
    return {'anchor_cos_hap_exc': round(float(cosm[0, 4]), 4),
            'anchor_cos_ang_fru': round(float(cosm[3, 5]), 4)}

print('M3 build_model / build_losses / telemetry defined.')
'''

M3_P1_SRC = '''\
# ── Phase 1 (3A): label smoothing x class-balance grid — re-baselines all later runs
L_GRID_3A = {
    'L0_ls00_cbOff': AblationCfg(label_smoothing=0.0, cb_weights=False),
    'L1_ls00_cbOn':  AblationCfg(label_smoothing=0.0, cb_weights=True),
    'L2_ls01_cbOff': AblationCfg(label_smoothing=0.1, cb_weights=False),
    'L3_ls01_cbOn':  AblationCfg(label_smoothing=0.1, cb_weights=True),
}
all_results = _load_results()
for name, ab in L_GRID_3A.items():
    if name in all_results:
        print(f'[SKIP] {name}')
        continue
    run_ablation(name, ab)

# Pick the base: judged on mF1 + minority-class (hap, idx 0) F1, not wF1
res = _load_results()
print('\\n3A grid (pick by mF1; hap F1 shown as the minority-dilution symptom):')
for k in L_GRID_3A:
    r = res[k]
    print(f"  {k:16s} mF1={r['test_mf1']:.4f}  wF1={r['test_wf1']:.4f}  "
          f"hap_F1={r['test_f1_per_class'][0]:.4f}")
BASE_NAME = max(L_GRID_3A, key=lambda k: res[k]['test_mf1'])
BASE_AB   = L_GRID_3A[BASE_NAME]
print(f'\\n=> base config: {BASE_NAME}  ({BASE_AB})')
'''

M3_P2_SRC = '''\
# ── Phase 2: CBFC gamma sweep (L4) + BCL in both scopes (L5), on the 3A base
all_results = _load_results()
phase2 = {}
for g in (0.0, 1.0, 2.0):
    phase2[f'L4_cbfc_g{g:g}'] = replace(BASE_AB, contrastive='cbfc', cbfc_gamma=g)
phase2['L5_bcl_dia']    = replace(BASE_AB, contrastive='bcl')
phase2['L5_bcl_global'] = replace(BASE_AB, contrastive='bcl_global')
phase2['L5_contrast_off'] = replace(BASE_AB, contrastive='off')

for name, ab in phase2.items():
    if name in all_results:
        print(f'[SKIP] {name}')
        continue
    run_ablation(name, ab)

res = _load_results()
print('\\nPhase 2 (contrastive design):')
for k in phase2:
    r = res[k]
    print(f"  {k:18s} wF1={r['test_wf1']:.4f}  mF1={r['test_mf1']:.4f}  "
          f"hap_exc={r['pair_hap_exc']:.4f}  ang_fru={r['pair_ang_fru']:.4f}")
CONTRAST_NAME = max(phase2, key=lambda k: res[k]['test_wf1'])
CONTRAST_AB   = phase2[CONTRAST_NAME]
print(f'\\n=> contrastive winner: {CONTRAST_NAME}')
'''

M3_P3_SRC = '''\
# ── Phase 3: EACL on top of the contrastive winner, plus the interaction cell
# (EACL with contrastive off) — if the pair-F1 gain only appears without the
# contrastive term, the two losses are redundant and the cheaper one stays.
all_results = _load_results()
phase3 = {
    'L6_eacl':            replace(CONTRAST_AB, eacl=True),
    'L6b_eacl_noContrast': replace(BASE_AB, contrastive='off', eacl=True),
}
for name, ab in phase3.items():
    if name in all_results:
        print(f'[SKIP] {name}')
        continue
    run_ablation(name, ab)

res = _load_results()
print('\\nEACL verdict — judged on pair F1, not wF1:')
for k in [CONTRAST_NAME] + list(phase3):
    r = res[k]
    print(f"  {k:22s} wF1={r['test_wf1']:.4f}  "
          f"hap_exc={r['pair_hap_exc']:.4f}  ang_fru={r['pair_ang_fru']:.4f}")
print('\\nCheck anchor_cos_hap_exc in the history files: it must DECREASE over '
      'epochs for the loss to be doing its job.')
'''

M3_P4_SRC = '''\
# ── Phase 4: confirm the winner stack at 3 seeds
res = _load_results()
singles = {k: res[k]['test_wf1'] for k in res
           if k.startswith('L') and '_s4' not in k}
WINNER_NAME = max(singles, key=singles.get)
print(f'Winner by wF1: {WINNER_NAME}  ({singles[WINNER_NAME]:.4f})')
print('(Override WINNER_AB manually if the pair-F1 picture argues otherwise.)')

_ab_lookup = {**L_GRID_3A, **phase2, **phase3}
WINNER_AB = _ab_lookup[WINNER_NAME]
run_sweep('L7_winner', WINNER_AB)
'''


def build_m3():
    cells = base_cells(M3_TITLE, 'ablation_m3')
    cells += [
        md('## M3 Infrastructure'),
        code(M3_CFG_SRC),
        code(M3_LOSSES_SRC),
        code(M3_BUILD_SRC),
        code(RUN_INFRA_SRC),
        md('## Phase 1 — 3A smoothing × class-balance grid'),
        code(M3_P1_SRC),
        md('## Phase 2 — contrastive design (CBFC γ / BCL)'),
        code(M3_P2_SRC),
        md('## Phase 3 — EACL anchors (+ interaction with contrastive)'),
        code(M3_P3_SRC),
        md('## Phase 4 — winner stack, 3 seeds'),
        code(M3_P4_SRC),
    ]
    cells += analysis_cells('L3_ls01_cbOn', 'M3 Loss Design')
    return make_nb(cells)


# ─────────────────────────────────────────────────────────────────────────────
# M4 — Implicit edges
# ─────────────────────────────────────────────────────────────────────────────

M4_TITLE = '''\
# HyFIN-Net — M4 Ablation: Implicit Edge Quality

Implements the M4 grid from `m2_m3_m4_m6_implementation_plan.md`:
E0 off (the reference), E1 shared detector (current design),
E2 per-modality detectors with score-once caching, E3 + Gumbel top-k
selection, E4 CAD counterfactual scoring (closed-form leave-one-out).

**Every variant is judged against E0, not E1** — implicit edges must first
earn their place at all. Prior B4 result: removing implicit edges *helped*
(+0.0090 wF1), so E0 is a strong reference.

**Deviation from the plan, deliberate:** in the baseline the detector scores
are discarded after thresholding, so `ImplicitEdgeDetector.W` never receives
a gradient — it is a frozen random projection. Here detector scores are used
as the edge weights of implicit edges (explicit edges keep `angular_weight`),
which gives every detector a real training signal.
'''

M4_CFG_SRC = '''\
from dataclasses import replace

@dataclass
class AblationCfg:
    """M4 — implicit-edge flags."""
    implicit_mode:     str = 'shared'   # 'off'|'shared'|'per_modality'|'gumbel'|'cad'
    implicit_topk_div: int = 8          # k = max(2, n // this) for topk modes

print('AblationCfg (M4) defined.')
'''

M4_DETECTORS_SRC = '''\
class ThresholdDetector(nn.Module):
    """Original a > 1/n threshold rule, but also returns the softmax attention
    score per kept edge (used as edge weight -> detector gets gradient)."""
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(2 * d, 1, bias=False)
        self.act = nn.LeakyReLU(0.2)
    def _scores(self, f, n):
        fi = f.unsqueeze(1).expand(n, n, -1)
        fj = f.unsqueeze(0).expand(n, n, -1)
        s = self.act(self.W(torch.cat([fi, fj], dim=-1)).squeeze(-1))
        causal = torch.tril(torch.ones(n, n, device=f.device, dtype=torch.bool))
        a = F.softmax(s.masked_fill(~causal, -1e9), dim=-1)
        return a, causal
    def extra_edges(self, f, n):
        if n < 2:
            return None, None, None
        a, causal = self._scores(f, n)
        keep = (a > (1.0 / n)) & causal
        idx = keep.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return None, None, None
        u, w = idx[:, 0], idx[:, 1]
        return u, w, a[u, w]

class GumbelTopKDetector(ThresholdDetector):
    """Gumbel top-k selection over strict causal predecessors: fixed edge
    budget k = max(2, n // topk_div), Gumbel noise at train time."""
    def __init__(self, d, topk_div=8):
        super().__init__(d)
        self.topk_div = topk_div
    def extra_edges(self, f, n):
        if n < 3:
            return None, None, None
        a, _ = self._scores(f, n)
        causal = torch.tril(torch.ones(n, n, device=f.device, dtype=torch.bool),
                            diagonal=-1)
        s = torch.log(a.clamp(min=1e-12)).masked_fill(~causal, -1e9)
        if self.training:
            g = -torch.log(-torch.log(torch.rand_like(s).clamp(1e-9, 1 - 1e-9)))
            s = s + g
        k = min(max(2, n // self.topk_div), n - 1)
        keep_idx = s.topk(k, dim=-1).indices                     # [n,k]
        rows = torch.arange(n, device=f.device)[:, None].expand(n, k)
        valid = causal.gather(1, keep_idx)                       # drop filler picks
        u, w = rows[valid], keep_idx[valid]
        if u.numel() == 0:
            return None, None, None
        return u, w, a[u, w]

class CADDetector(nn.Module):
    """Counterfactual (leave-one-out) influence scoring with a single-head
    attention scorer — closed form, no second forward pass. Influence of
    candidate i on target j = ||out_j - out_j_without_i||. Selection: Gumbel
    top-k over influence scores."""
    def __init__(self, d, topk_div=8):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.topk_div = topk_div
    def extra_edges(self, f, n):
        if n < 3:
            return None, None, None
        d = f.size(-1)
        q, k, v = self.q(f), self.k(f), self.v(f)
        att = (q @ k.t()) / math.sqrt(d)
        causal = torch.tril(torch.ones(n, n, device=f.device, dtype=torch.bool),
                            diagonal=-1)
        e = torch.exp(att.clamp(max=30.0)) * causal.float()      # [n,n]
        S = e.sum(1, keepdim=True)                               # [n,1]
        out = (e @ v) / S.clamp(min=1e-12)                       # [n,d]
        denom = (S - e).clamp(min=1e-12)
        out_wo = (out.unsqueeze(1) * S.unsqueeze(-1)
                  - e.unsqueeze(-1) * v.unsqueeze(0)) / denom.unsqueeze(-1)
        infl = (out.unsqueeze(1) - out_wo).norm(dim=-1)          # [n,n]
        score = F.softmax(infl.masked_fill(~causal, -1e9), dim=-1)
        s = torch.log(score.clamp(min=1e-12)).masked_fill(~causal, -1e9)
        if self.training:
            g = -torch.log(-torch.log(torch.rand_like(s).clamp(1e-9, 1 - 1e-9)))
            s = s + g
        k_ = min(max(2, n // self.topk_div), n - 1)
        keep_idx = s.topk(k_, dim=-1).indices
        rows = torch.arange(n, device=f.device)[:, None].expand(n, k_)
        valid = causal.gather(1, keep_idx)
        u, w = rows[valid], keep_idx[valid]
        if u.numel() == 0:
            return None, None, None
        return u, w, score[u, w]

print('ThresholdDetector / GumbelTopKDetector / CADDetector defined.')
'''

M4_MODEL_SRC = '''\
class M4IGM(InceptionGraphModule):
    """IGM with configurable implicit-edge design. Implicit edges are scored
    ONCE per forward (window-independent) and shared across branches; their
    detector scores become edge weights so detectors receive gradients.
    Edge-count telemetry accumulates during training epochs."""
    def __init__(self, d, windows, n_layers, heads=4, dropout=0.1, ab=None):
        super().__init__(d, windows, n_layers, heads=heads, dropout=dropout)
        self.ab = ab
        mode = ab.implicit_mode
        if mode == 'shared':
            self.dets = nn.ModuleList([ThresholdDetector(d)])      # one, shared
        elif mode == 'per_modality':
            self.dets = nn.ModuleList([ThresholdDetector(d) for _ in range(3)])
        elif mode == 'gumbel':
            self.dets = nn.ModuleList([GumbelTopKDetector(d, ab.implicit_topk_div)
                                       for _ in range(3)])
        elif mode == 'cad':
            self.dets = nn.ModuleList([CADDetector(d, ab.implicit_topk_div)
                                       for _ in range(3)])
        else:                                                       # 'off'
            self.dets = None
        self.implicit = None        # drop the unused inherited detector
        self.reset_edge_counts()

    def reset_edge_counts(self):
        self.edge_counts = {'t': 0, 'a': 0, 'v': 0, 'cand': 0}

    def _implicit_edges(self, lengths, offsets, ht, ha, hv):
        if self.dets is None:
            return None
        feats = (ht, ha, hv)
        srcs, dsts, scrs = [], [], []
        for b, n in enumerate(lengths.tolist()):
            o = offsets[b]
            for m, (mname, bs) in enumerate(zip('tav', (o, o + n, o + 2 * n))):
                det = self.dets[0] if len(self.dets) == 1 else self.dets[m]
                u, w, s = det.extra_edges(feats[m][b, :n], n)
                if u is None:
                    continue
                u = u + bs; w = w + bs
                srcs += [u, w]; dsts += [w, u]; scrs += [s, s]
                if self.training:
                    self.edge_counts[mname] += int(u.numel())
            if self.training:
                self.edge_counts['cand'] += 3 * (n * (n - 1)) // 2
        if not srcs:
            return None
        return (torch.stack([torch.cat(srcs), torch.cat(dsts)], 0),
                torch.cat(scrs))

    def forward(self, flat, lengths, offsets, ht, ha, hv):
        imp = self._implicit_edges(lengths, offsets, ht, ha, hv)
        outs = []
        for (p, f), branch in zip(self.windows, self.branches):
            ei = build_igm_graph(lengths, offsets, ht, ha, hv, p, f,
                                 implicit=None, device=flat.device)
            ew = angular_weight(flat, ei)
            if imp is not None:
                ei = torch.cat([ei, imp[0]], dim=1)
                ew = torch.cat([ew, imp[1]], dim=0)
            outs.append(branch(flat, ei, ew))
        return torch.stack(outs, dim=0).mean(dim=0)

class M4Net(HyFINNet):
    def __init__(self, cfg, d_t, d_a, d_v, ab):
        super().__init__(cfg, d_t, d_a, d_v)
        self.ab = ab
        d = cfg.hidden; ds = cfg.dataset
        self.igm = M4IGM(d, cfg.igm_branches[ds], cfg.igm_layers,
                         heads=cfg.igm_heads, dropout=cfg.dropout[ds], ab=ab)

def build_model(ab):
    return M4Net(cfg, D_T, D_A, D_V, ab)

def build_losses(ab, model):
    w = effective_class_weights(class_counts, beta=cfg.beta_cb).to(device)
    return {'cbce': CBCELoss(w, label_smoothing=cfg.label_smoothing),
            'contrast': CBFCLoss(w, gamma=cfg.cbfc_gamma, temp=cfg.cbfc_temp),
            'mu': cfg.cbfc_mu}

def EPOCH_TELEMETRY(model):
    """Kept-edge counts per modality per epoch. A detector keeping >30% of
    all causal pairs is a noise source regardless of metrics."""
    ec = dict(model.igm.edge_counts)
    model.igm.reset_edge_counts()
    if ec['cand'] > 0:
        ec['kept_frac'] = round((ec['t'] + ec['a'] + ec['v']) / ec['cand'], 4)
    return {f'imp_{k}': v for k, v in ec.items()}

print('M4IGM / M4Net / build_model / build_losses / telemetry defined.')
'''

M4_GRID_SRC = '''\
M4_GRID = {
    'E0_off':           AblationCfg(implicit_mode='off'),
    'E1_shared':        AblationCfg(implicit_mode='shared'),
    'E2_per_modality':  AblationCfg(implicit_mode='per_modality'),
    'E3_gumbel':        AblationCfg(implicit_mode='gumbel'),
}
print(f'M4 grid: {list(M4_GRID)}  (+ E4_cad, conditional)')
'''

M4_LOOP_SRC = '''\
all_results = _load_results()
for name, ab in M4_GRID.items():
    if name in all_results:
        print(f'[SKIP] {name} already in results.')
        continue
    run_ablation(name, ab)

# E4 (CAD) only if E2 or E3 beats E0
res = _load_results()
e0 = res['E0_off']['test_wf1']
if max(res['E2_per_modality']['test_wf1'], res['E3_gumbel']['test_wf1']) > e0:
    if 'E4_cad' not in res:
        run_ablation('E4_cad', AblationCfg(implicit_mode='cad'))
    else:
        print('[SKIP] E4_cad already in results.')
else:
    print(f'E4_cad skipped: neither E2 nor E3 beats E0 (wF1={e0:.4f}).')
print('\\nM4 screening complete.')
'''

M4_CONFIRM_SRC = '''\
# Confirm: top candidate vs E0 at 3 seeds; McNemar on every screened config
res = _load_results()
screened = [k for k in ('E1_shared', 'E2_per_modality', 'E3_gumbel', 'E4_cad')
            if k in res]
top = max(screened, key=lambda k: res[k]['test_wf1'])
e0 = res['E0_off']['test_wf1']
print(f"Top implicit-edge design: {top}  wF1={res[top]['test_wf1']:.4f} "
      f"vs E0 {e0:.4f}  (Δ={res[top]['test_wf1']-e0:+.4f})")

if res[top]['test_wf1'] > e0:
    run_sweep(f'{top}_confirm', M4_GRID.get(top, AblationCfg(implicit_mode='cad')))
    run_sweep('E0_off_confirm', M4_GRID['E0_off'])
else:
    print('Verdict: implicit edges do not earn their place — E0 stands.')
'''

M4_TELEM_SRC = '''\
# Edge-count telemetry across epochs (mechanism evidence for the paper)
plt.figure(figsize=(11, 5))
for name in sorted(res):
    hf = CKPT_DIR / f'{name}_history.json'
    if not hf.exists():
        continue
    h = json.load(open(hf))['history']
    if 'imp_kept_frac' not in h[-1]:
        continue
    plt.plot([e['epoch'] for e in h],
             [e.get('imp_kept_frac', 0) for e in h], label=name)
plt.axhline(0.30, ls='--', c='red', lw=1, label='noise threshold (30%)')
plt.xlabel('epoch'); plt.ylabel('kept fraction of causal pairs')
plt.title('M4 — implicit-edge budget per design'); plt.legend(); plt.tight_layout()
plt.savefig(PLOT_DIR / 'edge_budget_curves.png', dpi=150); plt.show()
'''


def build_m4():
    cells = base_cells(M4_TITLE, 'ablation_m4')
    cells += [
        md('## M4 Infrastructure'),
        code(M4_CFG_SRC),
        code(M4_DETECTORS_SRC),
        code(M4_MODEL_SRC),
        code(RUN_INFRA_SRC),
        md('## M4 Grid (all judged against E0)'),
        code(M4_GRID_SRC),
        code(M4_LOOP_SRC),
        md('## Confirmation (3 seeds) + significance'),
        code(M4_CONFIRM_SRC),
    ]
    cells += analysis_cells('E0_off', 'M4 Implicit Edges')
    cells.append(code(M4_TELEM_SRC))
    return make_nb(cells)


# ─────────────────────────────────────────────────────────────────────────────
# M6 — Fusion
# ─────────────────────────────────────────────────────────────────────────────

M6_TITLE = '''\
# HyFIN-Net — M6 Ablation: Fusion Design

Implements the M6 grid from `m2_m3_m4_m6_implementation_plan.md`:
F0 parameter-free mean (the mandatory control), F1 text-anchored attention
(current design), F2 symmetric pairwise attention, F3 gated additive fusion,
F4 best-of + alternating inter/intra IGM schedule (GraphSmile mechanism),
F5 + cross-utterance inter-modal edges (6C+).

F1–F3 must beat F0 to keep their parameters. Re-screen this grid once after
the M2 winner locks — balancing changes which fusion wins.
'''

M6_CFG_SRC = '''\
from dataclasses import replace

@dataclass
class AblationCfg:
    """M6 — fusion flags."""
    fusion_mode:     str  = 'text_anchor'  # 'mean'|'text_anchor'|'pairwise'|'gated'
    igm_schedule:    str  = 'joint'        # 'joint'|'alternating'|'alternating_rev'
    cross_utt_inter: bool = False          # 6C+: cross-utterance inter-modal edges

print('AblationCfg (M6) defined.')
'''

M6_FUSION_SRC = '''\
class PairwiseCrossModalAttn(nn.Module):
    """Symmetric pairwise attention: all six directed cross-modal attentions
    {V->T, A->T, T->A, V->A, T->V, A->V}. Output width 3*d_h (same as the
    text-anchored design)."""
    def __init__(self, d_in, d_h, heads=4, dropout=0.3):
        super().__init__()
        self.mods = ('t', 'a', 'v')
        self.proj = nn.ModuleDict({m: nn.Linear(d_in, d_h) for m in self.mods})
        self.attn = nn.ModuleDict({
            f'{s}2{t}': nn.MultiheadAttention(d_h, heads, dropout=dropout,
                                              batch_first=True)
            for t in self.mods for s in self.mods if s != t})
    def forward(self, mt, ma, mv, key_padding_mask):
        h = {m: F.gelu(self.proj[m](x)) for m, x in zip(self.mods, (mt, ma, mv))}
        outs = []
        for t in self.mods:
            f = h[t]
            for s in self.mods:
                if s == t:
                    continue
                o, _ = self.attn[f'{s}2{t}'](h[t], h[s], h[s],
                                             key_padding_mask=key_padding_mask)
                f = f + o
            outs.append(f)
        return torch.cat(outs, dim=-1)

class GatedFusion(nn.Module):
    """Gated additive fusion with learned per-utterance modality weights.
    Output cat([fused, t, a, v]) -> width 4*d_h (classifier resized).
    Mean gate weights logged as collapse telemetry (w_text -> 1 is the
    documented failure mode)."""
    def __init__(self, d_in, d_h):
        super().__init__()
        self.proj = nn.ModuleDict({m: nn.Linear(d_in, d_h) for m in ('t', 'a', 'v')})
        self.w = nn.Sequential(nn.Linear(3 * d_h, 64), nn.GELU(), nn.Linear(64, 3))
        self.last_w = None
    def forward(self, mt, ma, mv, key_padding_mask=None):
        t = F.gelu(self.proj['t'](mt))
        a = F.gelu(self.proj['a'](ma))
        v = F.gelu(self.proj['v'](mv))
        w = torch.softmax(self.w(torch.cat([t, a, v], dim=-1)), dim=-1)
        if key_padding_mask is not None:
            self.last_w = w[~key_padding_mask].mean(0).detach().cpu().tolist()
        fused = w[..., 0:1] * t + w[..., 1:2] * a + w[..., 2:3] * v
        return torch.cat([fused, t, a, v], dim=-1)

print('PairwiseCrossModalAttn / GatedFusion defined.')
'''

M6_GRAPH_SRC = '''\
from itertools import combinations

def build_igm_graph_split(lengths, offsets, ht, ha, hv, p_window, f_window,
                          implicit=None, cross_utt_inter=False, device='cpu'):
    """Edge set split for the alternating schedule:
    E_inter = cross-modal edges (same-utterance; + cross-utterance within the
              past window if cross_utt_inter — the 6C+ extension),
    E_intra = intra-modal windowed edges + implicit edges."""
    src_i, dst_i = [], []
    src_a, dst_a = [], []
    feats_for_implicit = (ht, ha, hv)
    for b, n in enumerate(lengths.tolist()):
        o = offsets[b]
        t0, a0, v0 = o, o + n, o + 2 * n
        idx = [torch.arange(n, device=device) + bs for bs in (t0, a0, v0)]
        # inter: same-utterance cross-modal (baseline behavior)
        for x, y in combinations(idx, 2):
            src_i += [x, y]; dst_i += [y, x]
        # 6C+: cross-utterance cross-modal pairs within the past window
        if cross_utt_inter and n > 1:
            ig = torch.arange(n, device=device)
            for shift in range(1, p_window + 1):
                m = ig >= shift
                if not m.any():
                    continue
                pos = ig[m]
                for A, B2 in combinations(idx, 2):
                    src_i += [A[pos], B2[pos - shift], B2[pos], A[pos - shift]]
                    dst_i += [B2[pos - shift], A[pos], A[pos - shift], B2[pos]]
        # intra: windowed edges per modality
        ig = torch.arange(n, device=device)
        for ix in idx:
            for shift in range(1, p_window + 1):
                m = ig >= shift
                if m.any():
                    src_a += [ix[m], ix[m] - shift]; dst_a += [ix[m] - shift, ix[m]]
            for shift in range(1, f_window + 1):
                m = (ig + shift) < n
                if m.any():
                    src_a += [ix[m], ix[m] + shift]; dst_a += [ix[m] + shift, ix[m]]
        # implicit edges -> intra set (they are intra-modal)
        if implicit is not None:
            for mod_idx, bs in enumerate((t0, a0, v0)):
                u, w = implicit.extra_edges(feats_for_implicit[mod_idx][b, :n], n)
                if u is not None:
                    u = u + bs; w = w + bs
                    src_a += [u, w]; dst_a += [w, u]
    def _stack(src, dst):
        if not src:
            return torch.zeros(2, 0, dtype=torch.long, device=device)
        return torch.stack([torch.cat(src), torch.cat(dst)], dim=0)
    return _stack(src_i, dst_i), _stack(src_a, dst_a)

class M6IGM(InceptionGraphModule):
    """IGM with optional alternating inter/intra layer schedule."""
    def __init__(self, d, windows, n_layers, heads=4, dropout=0.1, ab=None):
        super().__init__(d, windows, n_layers, heads=heads, dropout=dropout)
        self.ab = ab
    def forward(self, flat, lengths, offsets, ht, ha, hv):
        if self.ab.igm_schedule == 'joint' and not self.ab.cross_utt_inter:
            return super().forward(flat, lengths, offsets, ht, ha, hv)
        order = ('intra', 'inter') if self.ab.igm_schedule == 'alternating_rev' \\
                else ('inter', 'intra')
        outs = []
        for (p, f), branch in zip(self.windows, self.branches):
            ei_inter, ei_intra = build_igm_graph_split(
                lengths, offsets, ht, ha, hv, p, f, implicit=self.implicit,
                cross_utt_inter=self.ab.cross_utt_inter, device=flat.device)
            ew_inter = angular_weight(flat, ei_inter)
            ew_intra = angular_weight(flat, ei_intra)
            h = flat
            for li, (conv, ln) in enumerate(zip(branch.convs, branch.norms)):
                if self.ab.igm_schedule == 'joint':
                    ei = torch.cat([ei_inter, ei_intra], dim=1)
                    ew = torch.cat([ew_inter, ew_intra], dim=0)
                elif order[li % 2] == 'inter':
                    ei, ew = ei_inter, ew_inter
                else:
                    ei, ew = ei_intra, ew_intra
                h = ln(F.relu(conv(h, ei, ew.unsqueeze(-1))) + h)
            outs.append(h)
        return torch.stack(outs, dim=0).mean(dim=0)

print('build_igm_graph_split / M6IGM defined.')
'''

M6_MODEL_SRC = '''\
class M6Net(HyFINNet):
    """HyFINNet with configurable fusion + IGM schedule."""
    def __init__(self, cfg, d_t, d_a, d_v, ab):
        super().__init__(cfg, d_t, d_a, d_v)
        self.ab = ab
        d = cfg.hidden; ds = cfg.dataset
        self.igm = M6IGM(d, cfg.igm_branches[ds], cfg.igm_layers,
                         heads=cfg.igm_heads, dropout=cfg.dropout[ds], ab=ab)
        if ab.fusion_mode == 'pairwise':
            self.cross = PairwiseCrossModalAttn(3 * d, d, heads=cfg.ca_heads,
                                                dropout=cfg.dropout[ds])
        elif ab.fusion_mode == 'gated':
            self.cross = GatedFusion(3 * d, d)
            self.clf = Classifier(4 * d, cfg.n_classes[ds], cfg.dropout[ds])
        elif ab.fusion_mode == 'mean':
            self.cross = None
        # 'text_anchor' keeps the inherited CrossModalAttn

    def _features(self, batch):
        text   = batch['text'].to(device)
        audio  = batch['audio'].to(device)
        visual = batch['visual'].to(device)
        spk    = batch['speaker'].to(device)
        lens   = batch['lengths'].to(device)
        ht, ha, hv, key_pad_mask = self.encoder(text, audio, visual, spk, lens)
        flat, offsets = flatten_batch(ht, ha, hv, lens)
        p_flat = self.igm(flat, lens, offsets, ht, ha, hv)
        q_flat = self.hm(flat, lens, offsets)
        f_flat = self.mf(flat, lens, offsets)
        m_flat = torch.cat([p_flat, q_flat, f_flat], dim=-1)
        mt, ma, mv = unflatten_batch(m_flat, lens, offsets)
        if self.ab.fusion_mode == 'mean':
            fused = (mt + ma + mv) / 3.0
        else:
            fused = self.cross(mt, ma, mv, key_padding_mask=key_pad_mask)
        return fused, key_pad_mask, lens

def build_model(ab):
    return M6Net(cfg, D_T, D_A, D_V, ab)

def build_losses(ab, model):
    w = effective_class_weights(class_counts, beta=cfg.beta_cb).to(device)
    return {'cbce': CBCELoss(w, label_smoothing=cfg.label_smoothing),
            'contrast': CBFCLoss(w, gamma=cfg.cbfc_gamma, temp=cfg.cbfc_temp),
            'mu': cfg.cbfc_mu}

def EPOCH_TELEMETRY(model):
    """Gate-weight collapse telemetry for the gated fusion."""
    cr = getattr(model, 'cross', None)
    if isinstance(cr, GatedFusion) and cr.last_w is not None:
        return {'gate_w_tav': [round(float(x), 4) for x in cr.last_w]}
    return {}

print('M6Net / build_model / build_losses / telemetry defined.')
'''

M6_GRID_SRC = '''\
F_GRID = {
    'F0_mean':        AblationCfg(fusion_mode='mean'),
    'F1_text_anchor': AblationCfg(fusion_mode='text_anchor'),
    'F2_pairwise':    AblationCfg(fusion_mode='pairwise'),
    'F3_gated':       AblationCfg(fusion_mode='gated'),
}
print(f'M6 grid: {list(F_GRID)}  (+ F4 alternating, F5 cross-utterance)')
'''

M6_LOOP_SRC = '''\
all_results = _load_results()
for name, ab in F_GRID.items():
    if name in all_results:
        print(f'[SKIP] {name} already in results.')
        continue
    run_ablation(name, ab)

res = _load_results()
f0 = res['F0_mean']['test_wf1']
print(f'\\nF0 (mean control) wF1 = {f0:.4f}. Fusions that fail to beat it do '
      f'not earn their parameters:')
for k in ('F1_text_anchor', 'F2_pairwise', 'F3_gated'):
    print(f"  {k:16s} wF1={res[k]['test_wf1']:.4f}  Δ={res[k]['test_wf1']-f0:+.4f}")
'''

M6_F4_SRC = '''\
# F4: best of F0–F3 + alternating inter/intra schedule (both layer orders)
res = _load_results()
BEST_FUSION = max(F_GRID, key=lambda k: res[k]['test_wf1'])
print(f'Fusion winner: {BEST_FUSION}')

f4 = {
    'F4a_alt_interFirst': replace(F_GRID[BEST_FUSION], igm_schedule='alternating'),
    'F4b_alt_intraFirst': replace(F_GRID[BEST_FUSION], igm_schedule='alternating_rev'),
}
for name, ab in f4.items():
    if name in res:
        print(f'[SKIP] {name}')
        continue
    run_ablation(name, ab)
'''

M6_F5_SRC = '''\
# F5: best F4 + cross-utterance inter-modal edges (6C+) — attributed
# independently of the schedule
res = _load_results()
f4_keys = [k for k in ('F4a_alt_interFirst', 'F4b_alt_intraFirst') if k in res]
best_f4 = max(f4_keys, key=lambda k: res[k]['test_wf1'])
base_f4 = f4[best_f4]
print(f'Schedule winner: {best_f4}')

if 'F5_cross_utt' not in res:
    run_ablation('F5_cross_utt', replace(base_f4, cross_utt_inter=True))
else:
    print('[SKIP] F5_cross_utt')

# Also attribute 6C+ without the schedule (joint + cross-utterance)
if 'F5b_cross_utt_joint' not in res:
    run_ablation('F5b_cross_utt_joint',
                 replace(F_GRID[BEST_FUSION], cross_utt_inter=True))
else:
    print('[SKIP] F5b_cross_utt_joint')
'''

M6_CONFIRM_SRC = '''\
# Confirm the overall M6 winner at 3 seeds
res = _load_results()
singles = {k: v['test_wf1'] for k, v in res.items()
           if not any(k.endswith(f'_s{s}') for s in (42, 43, 44))}
winner = max(singles, key=singles.get)
print(f'M6 winner: {winner}  wF1={singles[winner]:.4f}')

_lookup = {**F_GRID, **f4,
           'F5_cross_utt': replace(base_f4, cross_utt_inter=True),
           'F5b_cross_utt_joint': replace(F_GRID[BEST_FUSION], cross_utt_inter=True)}
if winner in _lookup:
    run_sweep(f'{winner}_confirm', _lookup[winner])
'''


def build_m6():
    cells = base_cells(M6_TITLE, 'ablation_m6')
    cells += [
        md('## M6 Infrastructure'),
        code(M6_CFG_SRC),
        code(M6_FUSION_SRC),
        code(M6_GRAPH_SRC),
        code(M6_MODEL_SRC),
        code(RUN_INFRA_SRC),
        md('## M6 Grid — fusion designs vs the mean control'),
        code(M6_GRID_SRC),
        code(M6_LOOP_SRC),
        md('## F4 — alternating inter/intra schedule'),
        code(M6_F4_SRC),
        md('## F5 — cross-utterance inter-modal edges (6C+)'),
        code(M6_F5_SRC),
        md('## Confirmation (3 seeds)'),
        code(M6_CONFIRM_SRC),
    ]
    cells += analysis_cells('F1_text_anchor', 'M6 Fusion')
    return make_nb(cells)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    out = {
        'ablation_m2_modality_balance.ipynb': build_m2(),
        'ablation_m3_losses.ipynb':           build_m3(),
        'ablation_m4_implicit_edges.ipynb':   build_m4(),
        'ablation_m6_fusion.ipynb':           build_m6(),
    }
    here = Path(__file__).parent
    for fname, nb in out.items():
        (here / fname).write_text(json.dumps(nb, indent=1))
        print(f'wrote {fname}  ({len(nb["cells"])} cells)')
