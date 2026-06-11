#!/usr/bin/env python3
"""Smoke-test every variant in the M2/M3/M4/M6 ablation notebooks:
synthetic batch forward + backward, shape checks, hook checks.
No data files or training loops involved."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import make_module_ablation_notebooks as g

# ── base namespace: imports, Cfg, model classes, losses ──────────────────────
ns = {}
exec(''.join(g.BC[2]['source']), ns)                 # imports + seeds + device
ns['IS_KAGGLE'] = False
exec(''.join(g.BC[3]['source']), ns)                 # Cfg
exec(''.join(g.BC[7]['source']), ns)                 # model classes
exec(''.join(g.BC[8]['source']), ns)                 # losses / evaluate / sched

torch = ns['torch']
np = ns['np']
F = ns['F']
device = ns['device']

ns['D_T'], ns['D_A'], ns['D_V'] = 768, 74, 341
ns['class_counts'] = np.array([100, 120, 300, 200, 150, 400])


def synth_batch(B=2, L=7):
    lens = torch.tensor([L, L - 2])
    n_utt = int(lens.sum())
    return {'text': torch.randn(B, L, 768), 'audio': torch.randn(B, L, 74),
            'visual': torch.randn(B, L, 341),
            'speaker': torch.randint(0, 2, (B, L)),
            'lengths': lens,
            'labels': torch.randint(0, 6, (n_utt,))}, n_utt


def check(area, name, ans, ab, grad_module=None):
    model = ans['build_model'](ab).to(device)
    model.train()
    lp = ans['build_losses'](ab, model)
    batch, n_utt = synth_batch()
    logits, z, dia = model(batch, return_repr=True)
    assert logits.shape == (n_utt, 6), f'{name}: logits {tuple(logits.shape)}'
    y = batch['labels'].to(device)
    loss = lp['cbce'](logits, y)
    if lp.get('contrast') is not None and lp.get('mu', 0) > 0:
        loss = loss + lp['mu'] * lp['contrast'](z, y, dia)
    if lp.get('extra_loss') is not None:
        loss = loss + lp['extra_loss'](model, z, y)
    loss.backward()
    after = ans.get('AFTER_BACKWARD')
    if after is not None:
        after(model, y)
    grad_note = ''
    if grad_module is not None:
        gm = grad_module(model)
        gsum = sum(float(p.grad.abs().sum()) for p in gm.parameters()
                   if p.grad is not None)
        assert gsum > 0, f'{name}: detector got zero gradient'
        grad_note = f'  det_grad={gsum:.3e}'
    print(f'  OK [{area}] {name}  loss={float(loss):.4f}{grad_note}')


# ── M2 ────────────────────────────────────────────────────────────────────────
print('M2 — modality balance')
a = dict(ns)
for src in (g.M2_CFG_SRC, g.M2_MODULES_SRC, g.M2_MODEL_SRC, g.M2_HOOKS_SRC):
    exec(src, a)
AB = a['AblationCfg']
for name, ab in [('G0_probes', AB()),
                 ('G1_OGM', AB(ogm_ge=True, ogm_alpha=0.5)),
                 ('G2_AFW', AB(afw=True)),
                 ('G3_AFW_AMW', AB(afw=True, amw=True)),
                 ('G4_ModDrop', AB(mod_dropout_p=0.15)),
                 ('G5_stack', AB(ogm_ge=True, afw=True, amw=True,
                                 mod_dropout_p=0.15))]:
    check('M2', name, a, ab)

# ── M3 ────────────────────────────────────────────────────────────────────────
print('M3 — losses')
a = dict(ns)
for src in (g.M3_CFG_SRC, g.M3_LOSSES_SRC, g.M3_BUILD_SRC):
    exec(src, a)
AB = a['AblationCfg']
for name, ab in [('L0_ls0_cbOff', AB(label_smoothing=0.0, cb_weights=False)),
                 ('L3_ls01_cbOn', AB()),
                 ('L4_g0', AB(cbfc_gamma=0.0)),
                 ('L5_bcl_dia', AB(contrastive='bcl')),
                 ('L5_bcl_glob', AB(contrastive='bcl_global')),
                 ('L5_off', AB(contrastive='off')),
                 ('L6_eacl', AB(eacl=True)),
                 ('L6b_eacl_only', AB(contrastive='off', eacl=True))]:
    check('M3', name, a, ab)
# anchor telemetry works after a build
m = a['build_model'](AB(eacl=True)).to(device)
a['build_losses'](AB(eacl=True), m)
t = a['EPOCH_TELEMETRY'](m)
assert 'anchor_cos_hap_exc' in t, 'EACL telemetry missing'
print(f'  OK [M3] EACL telemetry  {t}')

# ── M4 ────────────────────────────────────────────────────────────────────────
print('M4 — implicit edges')
a = dict(ns)
for src in (g.M4_CFG_SRC, g.M4_DETECTORS_SRC, g.M4_MODEL_SRC):
    exec(src, a)
AB = a['AblationCfg']
check('M4', 'E0_off', a, AB(implicit_mode='off'))
for name, mode in [('E1_shared', 'shared'), ('E2_per_modality', 'per_modality'),
                   ('E3_gumbel', 'gumbel'), ('E4_cad', 'cad')]:
    check('M4', name, a, AB(implicit_mode=mode),
          grad_module=lambda m: m.igm.dets)
# edge-count telemetry populated
m = a['build_model'](AB(implicit_mode='gumbel')).to(device)
m.train()
batch, _ = synth_batch()
m(batch)
t = a['EPOCH_TELEMETRY'](m)
assert t.get('imp_cand', 0) > 0, 'M4 telemetry empty'
print(f'  OK [M4] edge telemetry  {t}')

# ── M6 ────────────────────────────────────────────────────────────────────────
print('M6 — fusion')
a = dict(ns)
for src in (g.M6_CFG_SRC, g.M6_FUSION_SRC, g.M6_GRAPH_SRC, g.M6_MODEL_SRC):
    exec(src, a)
AB = a['AblationCfg']
for name, ab in [('F0_mean', AB(fusion_mode='mean')),
                 ('F1_text_anchor', AB()),
                 ('F2_pairwise', AB(fusion_mode='pairwise')),
                 ('F3_gated', AB(fusion_mode='gated')),
                 ('F4a_alt', AB(igm_schedule='alternating')),
                 ('F4b_alt_rev', AB(igm_schedule='alternating_rev')),
                 ('F5_xu', AB(igm_schedule='alternating', cross_utt_inter=True)),
                 ('F5b_xu_joint', AB(cross_utt_inter=True)),
                 ('F_gated_alt', AB(fusion_mode='gated', igm_schedule='alternating'))]:
    check('M6', name, a, ab)
# gate telemetry
m = a['build_model'](AB(fusion_mode='gated')).to(device)
m.train()
batch, _ = synth_batch()
m(batch)
t = a['EPOCH_TELEMETRY'](m)
assert 'gate_w_tav' in t, 'M6 gate telemetry missing'
print(f'  OK [M6] gate telemetry  {t}')

print('\nALL SMOKE TESTS PASSED')
