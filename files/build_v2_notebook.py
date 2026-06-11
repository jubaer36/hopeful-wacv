#!/usr/bin/env python3
"""
build_v2_notebook.py — generate graphmerc_v2.ipynb from arch21.ipynb.

Strategy: REUSE the original notebook's verified data plumbing verbatim
(environment, imports, pickle parser, datasets, loaders, split) and REPLACE
the model/loss/training cells according to the two ablation campaigns.

  Study 1 (component removal, A/B/C grids):
    - REMOVE InputLN                  (+1.00 wF1 when removed)
    - REMOVE HyperGraphModule         (+0.47)
    - REMOVE MultiFrequencyModule     (+0.59)
    - REMOVE second IGM branch        (+0.45)
    - REMOVE CrossModalAttn           (M6: worst variant, -2.82 vs mean)
    - KEEP  PosEnc (-2.23 if removed), SpeakerEmb (-1.39), contrastive (-1.00),
            windowed graph (-0.45)

  Study 2 (M2/M3/M4/M6 grids):
    - M2: ADD modality dropout p=0.15; NO OGM-GE / AFW / AMW (all hurt)
    - M3: label_smoothing=0.0 + CB weights; contrastive = BCL GLOBAL
          (3-seed 0.6831±0.0077); CBFC gamma=1 fallback flag; NO EACL (-2.64)
    - M4: NO implicit edges (E0 3-seed 0.6860±0.0025 beats all detectors)
    - M6: fusion = MEAN; schedule = ALTERNATING INTER-FIRST;
          ADD cross-utterance inter-modal edges (F5: 0.6988±0.0047)

Run:  python build_v2_notebook.py [arch21.ipynb] [graphmerc_v2.ipynb]
"""
import json, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/arch21.ipynb'
DST = sys.argv[2] if len(sys.argv) > 2 else '/mnt/user-data/outputs/graphmerc_v2.ipynb'

with open(SRC) as f:
    old = json.load(f)

def oldsrc(i):
    return ''.join(old['cells'][i]['source'])

def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}

def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': text.splitlines(keepends=True)}

cells = []

# ── 0. Header ────────────────────────────────────────────────────────────────
cells.append(md("""# GraphMERC v2 — ablation-optimized MERC graph network
**Derived from HyFIN-Net (arch21) by applying the verdicts of two ablation campaigns**
(Study 1: component removal, 15 variants · Study 2: M2/M3/M4/M6 option grids, 45 variants).

| Change | Evidence |
|---|---|
| InputLN removed | Study 1 B3: +1.00 wF1 when removed |
| Hypergraph + Multi-Frequency modules removed | A2 +0.47 / A3 +0.59; C3-minimal best Study-1 variant |
| Single graph branch | B5 +0.45 |
| Implicit edges removed | M4: E0 3-seed 0.6860±0.0025 beats every detector variant |
| CrossModalAttn → parameter-free mean fusion | M6: text-anchor worst variant (−2.82 vs mean) |
| **Cross-utterance inter-modal edges added** | **M6 F5: 0.6988±0.0047 (+1.98 vs A0) — largest reliable gain** |
| Alternating inter-FIRST graph schedule | F4a +0.92 vs mean-base; intra-first −0.72 |
| label_smoothing 0.1 → 0.0 (CB weights kept) | M3 grid: smoothing dilutes minority up-weighting |
| CBFC(γ=2, dialogue) → **BCL (global scope)** | M3: BCL-global +0.51; CBFC-dialogue < no-contrastive |
| Modality dropout p=0.15 added | M2 G4: +0.49; only balancing method that helped |
| OGM-GE / AFW / AMW / EACL — excluded | all hurt when measured (−0.97…−2.64) |

Kept (essential per Study 1): **PosEnc** (−2.23 if removed), **speaker embedding** (−1.39),
**contrastive term** (−1.00), **windowed graph** (−0.45), training protocol.

> ⚠️ The three Study-2 winners were each measured on *different* bases. Section 10's
> R1 protocol verifies they compose before any number is trusted."""))

# ── 1. Environment + imports (verbatim) + set_seed ──────────────────────────
cells.append(md('## 0. Environment'))
cells.append(code(oldsrc(2)))
cells.append(code(oldsrc(3) + """

def set_seed(s):
    \"\"\"Model-training seed (the data split stays on SEED=2024 — both campaigns used it).\"\"\"
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
"""))

# ── 2. Config (v2) ───────────────────────────────────────────────────────────
cells.append(md('## 1. Config (v2)'))
cells.append(code("""# ── Local paths ── edit DATA_ROOT or set env vars when running outside Kaggle ─
_LOCAL_DATA_ROOT = os.environ.get(
    'GRAPHSMILE_DATA',
    str(Path('/mnt/Work/ML/Thesis/WACV/data/GraphSmile_PreProcessed')))
_LOCAL_SAVE_DIR  = os.environ.get('HYFIN_SAVE_DIR', './outputs')
# ─────────────────────────────────────────────────────────────────────────────

class Cfg:
    # ---- choose dataset: 'meld' or 'iemocap'
    dataset      = 'iemocap'
    data_root    = ('/kaggle/input/datasets/gilbertstrange/graphsmile-preprocessed/GraphSmile_PreProcessed'
                    if IS_KAGGLE else _LOCAL_DATA_ROOT)
    save_dir     = '/kaggle/working' if IS_KAGGLE else _LOCAL_SAVE_DIR
    @property
    def meld_path(self):    return f'{self.data_root}/meld_multi_features.pkl'
    @property
    def iemocap_path(self): return f'{self.data_root}/iemocap_multi_features.pkl'
    # ---- training (30 ep = the measured protocol of both campaigns; one 60-ep check in R1)
    batch_size_d = {'iemocap': 16, 'meld': 32}
    epochs       = 30
    lr           = 4e-4
    weight_decay = 1e-4
    grad_clip    = 1.0
    warmup_epochs = 1
    label_smoothing = 0.0          # M3 grid: ls=0.1 dilutes CB minority up-weighting
    cb_weights   = True            # M3 grid: +0.007-0.009 mF1, hap +0.05-0.09
    # ---- model
    hidden       = 256
    n_speakers   = {'iemocap': 2, 'meld': 9}
    n_classes    = {'iemocap': 6, 'meld': 7}
    # ---- graph (v2): SINGLE branch (B5), windowed + cross-utterance inter-modal (F5)
    window       = {'iemocap': (5, 3), 'meld': (7, 4)}   # (past, future)
    gnn_layers   = 2
    gnn_heads    = 4
    schedule     = 'alt_inter_first'   # F5 winner | 'joint' (F5b) | 'alt_intra_first' (loses)
    cross_utt_inter = True             # E3 — the +1.98 component; flag for attribution runs
    dropout      = {'iemocap': 0.5, 'meld': 0.5}
    # ---- losses (M3 verdicts)
    beta_cb      = 0.999
    contrastive  = 'bcl'           # 'bcl' (global, winner) | 'cbfc' (gamma=1 fallback) | 'off'
    con_mu       = 0.1
    con_temp     = 0.5
    cbfc_gamma   = 1.0             # M3 sweep: gamma=1 > gamma=0 > gamma=2
    # ---- training-side balance (M2 verdict)
    mod_dropout_p = 0.15           # only balancing method that helped (G4 +0.49)
    val_frac     = 0.1
    log_every    = 100
    @property
    def batch_size(self): return self.batch_size_d[self.dataset]

cfg = Cfg()
os.makedirs(cfg.save_dir, exist_ok=True)
print(f'data_root: {cfg.data_root}')
print(f'save_dir : {cfg.save_dir}')
"""))

# ── 3. Dataset (verbatim) ────────────────────────────────────────────────────
cells.append(md(oldsrc(6)))
cells.append(code(oldsrc(7)))
cells.append(code(oldsrc(8)))

# ── 4. Encoder (v2) — InputLN removed ────────────────────────────────────────
cells.append(md("""## 3. Unimodal Encoder (v2)
**Change vs arch21: `InputLN` removed** (Study 1 B3: removing the masked per-dialogue
LayerNorm *improved* wF1 by +1.00 and hap by +0.054 — the GraphSmile pickle features
are already utterance-normalized; per-dialogue whitening destroyed cross-dialogue
intensity information). PosEnc and the speaker embedding are the two most load-bearing
components in the whole study (−2.23 / −1.39 when removed) — kept."""))
cells.append(code("""class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d]
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class UnimodalEncoderV2(nn.Module):
    \"\"\"arch21 UnimodalEncoder minus InputLN (Study 1 B3 verdict).\"\"\"
    def __init__(self, d_t, d_a, d_v, d_h, n_speakers, dropout=0.5):
        super().__init__()
        self.t_proj = nn.Linear(d_t, d_h)
        self.pe     = PositionalEncoding(d_h)
        self.t_enc  = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_h, nhead=4, dim_feedforward=d_h,
                                       dropout=dropout, activation='gelu', batch_first=True),
            num_layers=1)
        self.a_proj = nn.Sequential(nn.Linear(d_a, d_h), nn.ReLU(), nn.Dropout(dropout))
        self.v_proj = nn.Sequential(nn.Linear(d_v, d_h), nn.ReLU(), nn.Dropout(dropout))
        self.spk    = nn.Embedding(n_speakers, d_h)

    def forward(self, text, audio, visual, spk, lengths):
        B, L, _ = text.shape
        mask = torch.arange(L, device=text.device)[None] >= lengths[:, None]  # True = pad
        ht = self.t_enc(self.pe(self.t_proj(text)), src_key_padding_mask=mask)
        ha = self.a_proj(audio)
        hv = self.v_proj(visual)
        s = self.spk(spk)
        return ht + s, ha + s, hv + s, mask
"""))

# ── 5. Graph helpers (verbatim) ──────────────────────────────────────────────
cells.append(md('## 4. Graph construction helpers (unchanged)'))
cells.append(code(oldsrc(12)))

# ── 6. Graph v2 ──────────────────────────────────────────────────────────────
cells.append(md("""## 5. Graph (v2) — windowed heterogeneous graph with cross-utterance inter-modal edges
Replaces the IGM/HM/MF triplet with ONE graph block over two edge sets:

- **E_intra** — within-modality window edges (past `p`, future `f`); directed in-edges,
  de-duplicated vs arch21 (whose past+future loops appended every pair twice).
- **E_inter** — cross-modal: same-utterance pairs (E2, as before) **plus
  cross-utterance pairs within the window (E3)** — the F5 component, the largest
  reliable gain in either campaign (3-seed 0.6988±0.0047, +1.98 vs A0). Independently
  corroborates GraphSmile's (TPAMI 2025) claim that same-utterance-only cross-modal
  graphs miss inter-utterance heterogeneous cues.
- **Schedule** — `alt_inter_first` (F4a; the reverse order loses to mean fusion).
  `joint` = F5b fallback (≈F5, simpler).
- No implicit edges (M4: E0 wins confirmation with half the variance)."""))
cells.append(code("""def build_graph_v2(lengths, offsets, p_window, f_window,
                   cross_utt_inter=True, device='cpu'):
    \"\"\"Returns (edge_index_inter, edge_index_intra) over the flattened node tensor.
    Node order per dialogue: [n text | n audio | n visual] (flatten_batch).\"\"\"
    src_i, dst_i = [], []   # inter-modal (E2 same-utt + E3 cross-utt)
    src_a, dst_a = [], []   # intra-modal window (E1)
    for b, n in enumerate(lengths.tolist()):
        o = offsets[b]
        idx = [torch.arange(n, device=device) + o,           # text block
               torch.arange(n, device=device) + o + n,       # audio block
               torch.arange(n, device=device) + o + 2 * n]   # visual block
        grid = torch.arange(n, device=device)
        # ---- E2: same-utterance cross-modal (bidirectional)
        for x in range(3):
            for y in range(x + 1, 3):
                src_i += [idx[x], idx[y]]; dst_i += [idx[y], idx[x]]
        # ---- E3: cross-utterance cross-modal within window (directed in-edges)
        if cross_utt_inter:
            for x in range(3):
                for y in range(3):
                    if x == y:
                        continue
                    for shift in range(1, p_window + 1):       # from y's past into x_i
                        m = grid >= shift
                        if m.any():
                            src_i.append(idx[y][m] - shift); dst_i.append(idx[x][m])
                    for shift in range(1, f_window + 1):       # from y's future into x_i
                        m = (grid + shift) < n
                        if m.any():
                            src_i.append(idx[y][m] + shift); dst_i.append(idx[x][m])
        # ---- E1: intra-modal window (directed in-edges, de-duplicated)
        for k in range(3):
            for shift in range(1, p_window + 1):
                m = grid >= shift
                if m.any():
                    src_a.append(idx[k][m] - shift); dst_a.append(idx[k][m])
            for shift in range(1, f_window + 1):
                m = (grid + shift) < n
                if m.any():
                    src_a.append(idx[k][m] + shift); dst_a.append(idx[k][m])
    def _stack(s, d):
        if len(s) == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=device)
        return torch.stack([torch.cat(s), torch.cat(d)], dim=0)
    return _stack(src_i, dst_i), _stack(src_a, dst_a)


class GraphBlockV2(nn.Module):
    \"\"\"Single-branch TransformerConv stack with an inter/intra layer schedule.
    Keeps arch21's relational prior: angular-similarity edge weight as 1-D edge_attr,
    gated against learned attention via beta=True.\"\"\"
    def __init__(self, d, n_layers, heads=4, dropout=0.1, schedule='alt_inter_first'):
        super().__init__()
        assert d % heads == 0, f'hidden {d} not divisible by heads {heads}'
        self.schedule = schedule
        self.convs = nn.ModuleList([
            TransformerConv(d, d // heads, heads=heads, concat=True,
                            beta=True, dropout=dropout, edge_dim=1)
            for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])

    def forward(self, x, e_inter, w_inter, e_intra, w_intra):
        if self.schedule == 'joint':
            e_all = torch.cat([e_inter, e_intra], dim=1)
            w_all = torch.cat([w_inter, w_intra], dim=0)
        h = x
        for li, (conv, ln) in enumerate(zip(self.convs, self.norms)):
            if self.schedule == 'joint':
                ei, ew = e_all, w_all
            elif self.schedule == 'alt_intra_first':
                ei, ew = (e_intra, w_intra) if li % 2 == 0 else (e_inter, w_inter)
            else:  # 'alt_inter_first' — the F5 winner
                ei, ew = (e_inter, w_inter) if li % 2 == 0 else (e_intra, w_intra)
            h = ln(F.relu(conv(h, ei, ew.unsqueeze(-1))) + h)
        return h
"""))

# ── 7. Fusion + classifier ───────────────────────────────────────────────────
cells.append(md("""## 6. Fusion (v2) — parameter-free mean + classifier
M6 verdict: the mean beat **every** learned fusion at the 30-epoch budget
(text-anchor −2.82, pairwise −1.35, gated −1.52). The classifier input shrinks
3d→d (the HM/MF zero-blocks are gone). A 60-epoch re-test of learned fusion is
queued in the R-protocol before the final lock."""))
cells.append(code("""class Classifier(nn.Module):
    def __init__(self, d_in, n_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_in // 2, n_classes))
    def forward(self, z): return self.net(z)
"""))

# ── 8. Assembly ──────────────────────────────────────────────────────────────
cells.append(md("""## 7. GraphMERC v2 assembly
encoder → (train-time modality dropout p=0.15, M2-G4) → flatten → GraphBlockV2
(alt inter-first, cross-utt inter edges) → unflatten → mean fusion → classifier.
`z` (penultimate, [N, d]) feeds the global-scope contrastive loss."""))
cells.append(code("""class GraphMERCv2(nn.Module):
    def __init__(self, cfg, d_t, d_a, d_v):
        super().__init__()
        d = cfg.hidden; ds = cfg.dataset
        self.cfg = cfg
        self.encoder = UnimodalEncoderV2(d_t, d_a, d_v, d,
                                         cfg.n_speakers[ds], cfg.dropout[ds])
        self.graph = GraphBlockV2(d, cfg.gnn_layers, heads=cfg.gnn_heads,
                                  dropout=cfg.dropout[ds], schedule=cfg.schedule)
        self.clf = Classifier(d, cfg.n_classes[ds], cfg.dropout[ds])

    def _modality_dropout(self, ht, ha, hv):
        \"\"\"M2-G4: with prob p per dialogue, zero ONE stream (train only).\"\"\"
        p = self.cfg.mod_dropout_p
        if not self.training or p <= 0:
            return ht, ha, hv
        B = ht.size(0)
        drop  = torch.rand(B, device=ht.device) < p          # which dialogues
        which = torch.randint(0, 3, (B,), device=ht.device)  # which stream
        streams = [ht.clone(), ha.clone(), hv.clone()]
        for m in range(3):
            sel = drop & (which == m)
            if sel.any():
                streams[m][sel] = 0.0
        return streams[0], streams[1], streams[2]

    def _features(self, batch):
        text   = batch['text'].to(device)
        audio  = batch['audio'].to(device)
        visual = batch['visual'].to(device)
        spk    = batch['speaker'].to(device)
        lens   = batch['lengths'].to(device)
        ht, ha, hv, key_pad_mask = self.encoder(text, audio, visual, spk, lens)
        ht, ha, hv = self._modality_dropout(ht, ha, hv)
        flat, offsets = flatten_batch(ht, ha, hv, lens)
        p_w, f_w = self.cfg.window[self.cfg.dataset]
        e_inter, e_intra = build_graph_v2(lens, offsets, p_w, f_w,
                                          cross_utt_inter=self.cfg.cross_utt_inter,
                                          device=flat.device)
        w_inter = angular_weight(flat, e_inter) if e_inter.numel() else flat.new_zeros(0)
        w_intra = angular_weight(flat, e_intra) if e_intra.numel() else flat.new_zeros(0)
        g = self.graph(flat, e_inter, w_inter, e_intra, w_intra)
        mt, ma, mv = unflatten_batch(g, lens, offsets)
        fused = (mt + ma + mv) / 3.0                      # F0/F5 mean fusion, [B, L, d]
        return fused, key_pad_mask, lens

    def forward(self, batch, return_repr=False):
        fused, key_pad_mask, lens = self._features(batch)
        z = fused[~key_pad_mask]                          # [N, d] penultimate
        logits = self.clf(z)
        if return_repr:
            B = lens.size(0)
            dialogue_ids = torch.arange(B, device=z.device).repeat_interleave(lens)
            return logits, z, dialogue_ids
        return logits
"""))

# ── 9. Losses ────────────────────────────────────────────────────────────────
cells.append(md("""## 8. Losses (v2) — CB-CE (ls=0) + BCL global
- **CB-CE**: effective-number weights kept; label smoothing removed (M3 grid: it
  dilutes minority up-weighting — hap −0.04–0.05 with ls=0.1).
- **BCL (global scope)** replaces CBFC(γ=2, dialogue): class-averaged positives +
  class-complement denominator (Zhu et al., CVPR 2022). M3: BCL-global +0.51 vs base;
  dialogue scoping was the hidden flaw (BCL-dia < no-contrastive).
- **CBFC(γ=1)** retained behind a flag — within 0.003 of BCL on the L1 base, lower
  variance; the R1 protocol re-checks both on the v2 base.
- EACL excluded (−2.64; anchor telemetry showed hap/exc anchors never separated).
  DualCL placeholder deleted."""))
cells.append(code("""def effective_class_weights(class_counts, beta=0.999):
    eff = 1.0 - np.power(beta, class_counts)
    w = (1.0 - beta) / np.maximum(eff, 1e-12)
    w = w / w.sum() * len(class_counts)
    return torch.tensor(w, dtype=torch.float32)

class CBCELoss(nn.Module):
    def __init__(self, w, label_smoothing=0.0):
        super().__init__()
        self.register_buffer('w', w); self.ls = label_smoothing
    def forward(self, logits, y):
        return F.cross_entropy(logits, y, weight=self.w.to(logits.device),
                               label_smoothing=self.ls)

class BCLLoss(nn.Module):
    \"\"\"Balanced Contrastive Learning (Zhu et al., CVPR 2022), GLOBAL scope (M3 winner).
    For each anchor i:
        L_i = -log( mean_{p: y_p=y_i, p!=i} e^{s_ip} / sum_c mean_{k: y_k=c, k!=i} e^{s_ik} )
    Class-averaging removes SupCon's head-class bias; the global (whole-batch) scope
    is what the M3 grid selected over within-dialogue scoping.\"\"\"
    def __init__(self, n_classes, temp=0.5):
        super().__init__()
        self.C = n_classes; self.t = temp
    def forward(self, z, y, dialogue_ids=None):           # dialogue_ids ignored (global)
        N = z.size(0)
        if N < 2:
            return z.new_zeros(())
        zn = F.normalize(z, dim=-1)
        sim = zn @ zn.t() / self.t
        sim = sim - sim.max(dim=-1, keepdim=True).values.detach()   # numerical stability
        eye = torch.eye(N, dtype=torch.bool, device=z.device)
        ex = sim.exp().masked_fill(eye, 0.0)
        Y = F.one_hot(y, self.C).to(ex.dtype)             # [N, C]
        S = ex @ Y                                        # per-anchor per-class sums (self excl.)
        cnt = Y.sum(0)[None, :] - Y                       # per-anchor class counts (self excl.)
        mean_c = S / cnt.clamp(min=1)
        denom = (mean_c * (cnt > 0)).sum(-1)              # class-complement denominator
        num = mean_c.gather(1, y[:, None]).squeeze(1)
        valid = cnt.gather(1, y[:, None]).squeeze(1) > 0  # anchors with >=1 positive
        if not valid.any():
            return z.new_zeros(())
        loss = -(num.clamp_min(1e-12).log() - denom.clamp_min(1e-12).log())
        return loss[valid].mean()

class CBFCLoss(nn.Module):
    \"\"\"Fallback (contrastive='cbfc'): ConxGNN-style focal supervised contrastive,
    gamma=1 per the M3 sweep (gamma=1 > gamma=0 > gamma=2), within-dialogue scope.\"\"\"
    def __init__(self, w, gamma=1.0, temp=0.5):
        super().__init__()
        self.register_buffer('w', w); self.gamma = gamma; self.temp = temp
    def forward(self, z, y, dialogue_ids):
        N = z.size(0)
        if N < 2:
            return z.new_zeros(())
        zn  = F.normalize(z, dim=-1)
        sim = (zn @ zn.t()) / self.temp
        same_dia  = dialogue_ids[:, None] == dialogue_ids[None, :]
        self_mask = torch.eye(N, dtype=torch.bool, device=z.device)
        cand = same_dia & ~self_mask
        pos  = cand & (y[:, None] == y[None, :])
        neg_inf = torch.finfo(sim.dtype).min
        logt = F.log_softmax(sim.masked_fill(~cand, neg_inf), dim=-1)
        t    = logt.exp()
        term = ((1.0 - t).pow(self.gamma) * logt).masked_fill(~pos, 0.0)
        num_pos = pos.sum(-1)
        valid   = num_pos > 0
        if not valid.any():
            return z.new_zeros(())
        per_anchor = -term.sum(-1) / num_pos.clamp(min=1)
        wj = self.w.to(z.device)[y]
        return (wj * per_anchor)[valid].sum() / valid.sum().clamp(min=1)
"""))

# ── 10. Train / evaluate ─────────────────────────────────────────────────────
cells.append(md("""## 9. Train / evaluate
arch21 protocol kept (AdamW, warmup+cosine, grad clip, best-val checkpointing;
best-test logged as upper bound only). Added: per-class F1 + hard-pair telemetry
in history (`pair_hap_exc`, `pair_ang_fru` on IEMOCAP), train−val gap logging,
`run_name` for sweep checkpoints, contrastive selected by `cfg.contrastive`."""))
cells.append(code("""IEMO_NAMES = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
MELD_NAMES = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']

@torch.no_grad()
def evaluate(model, loader, loss_fns=None):
    model.eval()
    ys, ps = [], []
    total_loss, nb = 0.0, 0
    for batch in loader:
        if loss_fns is not None:
            ce_fn, con_fn, mu = loss_fns
            logits, z, dia = model(batch, return_repr=True)
            y = batch['labels'].to(device)
            l = ce_fn(logits, y)
            if mu > 0 and con_fn is not None:
                l = l + mu * con_fn(z, y, dia)
            total_loss += l.item(); nb += 1
        else:
            logits = model(batch)
        ps.append(logits.argmax(-1).cpu().numpy())
        ys.append(batch['labels'].numpy())
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    acc = accuracy_score(ys, ps)
    wf1 = f1_score(ys, ps, average='weighted')
    mf1 = f1_score(ys, ps, average='macro')
    per_class = f1_score(ys, ps, average=None)
    mean_loss = (total_loss / max(1, nb)) if loss_fns is not None else None
    return acc, wf1, mf1, ys, ps, mean_loss, per_class

def make_scheduler(optim, cfg, steps_per_epoch):
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    total_steps  = cfg.epochs * steps_per_epoch
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

def _make_contrastive(cfg, w):
    if cfg.contrastive == 'bcl':
        return BCLLoss(cfg.n_classes[cfg.dataset], temp=cfg.con_temp)
    if cfg.contrastive == 'cbfc':
        return CBFCLoss(w, gamma=cfg.cbfc_gamma, temp=cfg.con_temp)
    return None

def train(cfg, raw, run_name=None):
    run_name = run_name or f'gmv2_{cfg.dataset}'
    model = GraphMERCv2(cfg, D_T, D_A, D_V).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[{run_name}] #params: {n_params/1e6:.2f}M  '
          f'schedule={cfg.schedule} cross_utt={cfg.cross_utt_inter} '
          f'con={cfg.contrastive} mu={cfg.con_mu} modDrop={cfg.mod_dropout_p}')
    if cfg.cb_weights:
        w = effective_class_weights(class_counts, beta=cfg.beta_cb).to(device)
    else:
        w = torch.ones(cfg.n_classes[cfg.dataset], device=device)
    ce  = CBCELoss(w, label_smoothing=cfg.label_smoothing)
    con = _make_contrastive(cfg, w)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = make_scheduler(optim, cfg, steps_per_epoch=len(train_loader))
    best_val  = {'wf1': -1, 'epoch': -1}
    best_test = {'wf1': -1, 'epoch': -1}
    history = []
    names = IEMO_NAMES if cfg.dataset == 'iemocap' else MELD_NAMES
    for ep in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time(); running = 0.0; nb = 0
        for step, batch in enumerate(train_loader):
            optim.zero_grad()
            logits, z, dia = model(batch, return_repr=True)
            y = batch['labels'].to(device)
            l_ce = ce(logits, y)
            l_co = con(z, y, dia) if (con is not None and cfg.con_mu > 0) else logits.new_zeros(())
            loss = l_ce + cfg.con_mu * l_co
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step(); sched.step()
            running += loss.item(); nb += 1
            if (step + 1) % cfg.log_every == 0:
                print(f'  ep{ep} step{step+1}/{len(train_loader)} '
                      f'ce={l_ce.item():.4f} con={l_co.item():.4f} lr={sched.get_last_lr()[0]:.2e}')
        tr_loss = running / max(1, nb)
        acc_v, wf1_v, mf1_v, _, _, loss_v, _ = evaluate(
            model, val_loader, loss_fns=(ce, con, cfg.con_mu))
        acc_t, wf1_t, mf1_t, _, _, _, pc_t = evaluate(model, test_loader)
        dt = time.time() - t0
        row = {'epoch': ep, 'loss': tr_loss, 'val_loss': loss_v,
               'val_acc': acc_v, 'val_wf1': wf1_v, 'val_mf1': mf1_v,
               'test_acc': acc_t, 'test_wf1': wf1_t, 'test_mf1': mf1_t,
               'gap': (loss_v - tr_loss) if loss_v is not None else None}
        for cname, cf1 in zip(names, pc_t):
            row[f'f1_{cname}'] = float(cf1)
        if cfg.dataset == 'iemocap':
            row['pair_hap_exc'] = (row['f1_hap'] + row['f1_exc']) / 2
            row['pair_ang_fru'] = (row['f1_ang'] + row['f1_fru']) / 2
        marker_v = marker_t = ''
        if wf1_v > best_val['wf1']:
            best_val = {'wf1': wf1_v, 'epoch': ep, 'test_acc': acc_t,
                        'test_wf1': wf1_t, 'test_mf1': mf1_t,
                        'per_class': {n: float(v) for n, v in zip(names, pc_t)}}
            torch.save({'state_dict': model.state_dict(),
                        'dims': (D_T, D_A, D_V), 'epoch': ep, 'best_val': best_val},
                       os.path.join(cfg.save_dir, f'{run_name}_bestval.pt'))
            marker_v = '  [BEST-VAL]'
        if wf1_t > best_test['wf1']:
            best_test = {'wf1': wf1_t, 'epoch': ep, 'val_wf1': wf1_v,
                         'test_acc': acc_t, 'test_mf1': mf1_t}
            marker_t = '  [BEST-TEST]'
        print(f'[ep{ep:02d}] {dt:.1f}s  train={tr_loss:.4f} val={loss_v:.4f} '
              f'gap={row["gap"]:+.4f}  val wF1={wf1_v:.4f}  '
              f'test wF1={wf1_t:.4f} mF1={mf1_t:.4f}{marker_v}{marker_t}')
        history.append(row)
    with open(os.path.join(cfg.save_dir, f'{run_name}_history.json'), 'w') as f:
        json.dump({'history': history, 'best_val': best_val, 'best_test': best_test}, f, indent=2)
    print(f'\\n[{run_name}] BEST-VAL ep{best_val["epoch"]:>3}: '
          f'test wF1={best_val.get("test_wf1", 0):.4f} mF1={best_val.get("test_mf1", 0):.4f}'
          f'   (BEST-TEST upper bound: {best_test["wf1"]:.4f} @ ep{best_test["epoch"]})')
    return model, history, best_val, best_test
"""))

# ── 11. R1 protocol runner ───────────────────────────────────────────────────
cells.append(md("""## 10. Run — R1 verification protocol
The three Study-2 winners (cross-utt edges/schedule, ModDrop, BCL-global) were each
measured on **different bases**; composition is an assumption until measured here.

| Run | Question |
|---|---|
| `V2_core` ×3 seeds | does the stack hold? (F5 alone: 0.6988±0.0047 — gate: ≥0.695) |
| `V2_noModDrop` | does ModDrop still earn its place on the F5 base? |
| `V2_noBCL` | does the contrastive still earn its place? |
| `V2_cbfc_g1` | BCL vs CBFC(γ=1) on the v2 base (Δ was 0.003 on L1) |
| `V2_joint` | F5b sanity: schedule-vs-edges attribution |
| `V2_core_60ep` | release the 30-epoch censoring once |

Set `RUN_R1 = True` to launch the sweep; default trains the single core config."""))
cells.append(code("""R1_REGISTRY = {
    'V2_core':      dict(),
    'V2_noModDrop': dict(mod_dropout_p=0.0),
    'V2_noBCL':     dict(contrastive='off'),
    'V2_cbfc_g1':   dict(contrastive='cbfc'),
    'V2_joint':     dict(schedule='joint'),
}

def run_v2(name, overrides=None, seed=42, epochs=None):
    c = copy.deepcopy(cfg)
    for k, v in (overrides or {}).items():
        setattr(c, k, v)
    if epochs is not None:
        c.epochs = epochs
    set_seed(seed)
    _, _, best_val, _ = train(c, raw, run_name=f'{name}_s{seed}')
    return best_val

RUN_R1 = False          # flip to True for the full verification sweep
SEEDS  = (42, 43, 44)

if RUN_R1:
    results = {}
    for seed in SEEDS:                                   # core, 3 seeds
        results[f'V2_core_s{seed}'] = run_v2('V2_core', seed=seed)
    for name in ('V2_noModDrop', 'V2_noBCL', 'V2_cbfc_g1', 'V2_joint'):
        results[f'{name}_s42'] = run_v2(name, R1_REGISTRY[name], seed=42)
    results['V2_core_60ep_s42'] = run_v2('V2_core_60ep', epochs=60, seed=42)
    core = [results[f'V2_core_s{s}']['test_wf1'] for s in SEEDS]
    print(f'\\nV2 core 3-seed: wF1 {np.mean(core):.4f} ± {np.std(core):.4f}')
    for k, v in results.items():
        print(f'  {k:24s} wF1={v["test_wf1"]:.4f} mF1={v["test_mf1"]:.4f} (ep{v["epoch"]})')
    with open(os.path.join(cfg.save_dir, 'r1_results.json'), 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'per_class'}
                   for k, v in results.items()}, f, indent=2)
else:
    model, history, best_val, best_test = train(cfg, raw)
    print('BEST-VAL :', {k: v for k, v in best_val.items() if k != 'per_class'})
"""))

# ── 12. Diagnostics: loss curve, report, export ──────────────────────────────
cells.append(md('## 11. Diagnostics — loss curve, report, export (single-run mode)'))
cells.append(code("""# --- Loss curve: train vs val ---
import matplotlib.pyplot as plt
if not RUN_R1:
    epochs_x = [h['epoch'] for h in history]
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_x, [h['loss'] for h in history],     marker='o', ms=3, lw=1.5, label='train loss')
    plt.plot(epochs_x, [h['val_loss'] for h in history], marker='s', ms=3, lw=1.5, label='val loss')
    if best_val.get('epoch', -1) > 0:
        plt.axvline(best_val['epoch'], color='grey', ls='--', lw=1,
                    label=f"best-val ep{best_val['epoch']}")
    plt.xlabel('epoch'); plt.ylabel('loss (CB-CE + mu*con)')
    plt.title(f'GraphMERC v2 loss curve — {cfg.dataset}')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out_png = os.path.join(cfg.save_dir, f'gmv2_{cfg.dataset}_loss_curve.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight'); plt.show()
    print('saved loss curve ->', out_png)
"""))
cells.append(code("""# --- Report with the best-val checkpoint (the proper benchmark) ---
if not RUN_R1:
    ckpt = torch.load(os.path.join(cfg.save_dir, f'gmv2_{cfg.dataset}_bestval.pt'),
                      map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    acc, wf1, mf1, y_true, y_pred, _, _ = evaluate(model, test_loader)
    target_names = IEMO_NAMES if cfg.dataset == 'iemocap' else MELD_NAMES
    print(f'[BEST-VAL ckpt]  TEST acc={acc:.4f}  weighted-F1={wf1:.4f}  macro-F1={mf1:.4f}')
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))
"""))
cells.append(code("""# ── Output export ────────────────────────────────────────────────────────────
if not RUN_R1:
    import csv, datetime
    import seaborn as sns
    run_ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(cfg.save_dir, f'gmv2_run_{cfg.dataset}_{run_ts}')
    os.makedirs(out_dir, exist_ok=True)
    report_str = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as f:
        f.write(f'GraphMERC v2 | Dataset: {cfg.dataset} | Run: {run_ts} | Epochs: {cfg.epochs}\\n')
        f.write(f'schedule={cfg.schedule} cross_utt={cfg.cross_utt_inter} '
                f'con={cfg.contrastive} modDrop={cfg.mod_dropout_p} ls={cfg.label_smoothing}\\n\\n')
        f.write(f'[BEST-VAL ckpt — test]  acc={acc:.4f}  wF1={wf1:.4f}  mF1={mf1:.4f}\\n\\n')
        f.write(report_str)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(len(target_names)*1.1 + 1, len(target_names)*1.0 + 1))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'GraphMERC v2 — {cfg.dataset} confusion (best-val ckpt)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    if history:
        with open(os.path.join(out_dir, 'training_history.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=history[0].keys())
            w.writeheader(); w.writerows(history)
    print('exported ->', out_dir)
"""))

nb = {'cells': cells,
      'metadata': old.get('metadata', {}),
      'nbformat': old.get('nbformat', 4),
      'nbformat_minor': old.get('nbformat_minor', 5)}

with open(DST, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'wrote {DST}  ({len(cells)} cells)')
