# Focused Implementation & Ablation Plan — M2 / M3 / M4 / M6

*Code-level plan against the existing arch21/ablation codebase (`UnimodalEncoder`, `InceptionGraphModule`, `ImplicitEdgeDetector`, `build_igm_graph`, `CrossModalAttn`, `CBCELoss`, `CBFCLoss`, `AblationCfg`, `run_ablation`). Each area: options → exact implementation → ablation grid → decision rules. Snippets are written to drop into the notebook with minimal renaming.*

---

## 0. Shared infrastructure (build once, used by all four areas)

### 0.1 Extended `AblationCfg`

```python
@dataclass
class AblationCfg:
    # ... existing flags ...
    # M2 — modality balance
    probe_heads:    bool  = False     # unimodal diagnostic probes (always-on telemetry once built)
    ogm_ge:         bool  = False     # gradient modulation
    ogm_alpha:      float = 0.5       # modulation strength
    afw:            bool  = False     # Ada2I feature gates
    amw:            bool  = False     # Ada2I modality-weighted logits
    mod_dropout_p:  float = 0.0       # modality dropout prob (0 = off)
    # M3 — losses
    label_smoothing: float = 0.1      # grid: {0.0, 0.1}
    cb_weights:      bool  = True     # grid: {on, off}
    contrastive:     str   = 'cbfc'   # 'off' | 'cbfc' | 'bcl'
    cbfc_gamma:      float = 2.0      # sweep {0,1,2}
    eacl:            bool  = False    # anchor loss on/off
    eacl_lambda:     float = 0.2
    # M4 — implicit edges
    implicit_mode:   str   = 'shared' # 'off' | 'shared' | 'per_modality' | 'gumbel' | 'cad'
    implicit_topk_div: int = 8        # k = max(2, n // this)
    # M6 — fusion
    fusion_mode:     str   = 'text_anchor'  # 'text_anchor' | 'mean' | 'pairwise' | 'gated'
    igm_schedule:    str   = 'joint'        # 'joint' | 'alternating'
```

### 0.2 Seed sweep + aggregation (wraps the existing `run_ablation`)

```python
SEEDS = (42, 43, 44)

def run_sweep(name, ab, epochs=30, seeds=SEEDS):
    rows = [run_ablation(f'{name}_s{s}', ab, epochs=epochs, seed=s) for s in seeds]
    import numpy as np
    agg = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
           for k in ('test_acc', 'test_wf1', 'test_mf1')}
    print(f'{name}: wF1 {agg["test_wf1"][0]:.4f}±{agg["test_wf1"][1]:.4f} '
          f'mF1 {agg["test_mf1"][0]:.4f}±{agg["test_mf1"][1]:.4f}')
    return agg
```

**Screen-then-confirm protocol** (keeps the budget sane): every grid is first screened at **1 seed / 30 epochs**; configs within 0.3 wF1 of the grid's best (or beating baseline by >0.3) are **confirmed at 3 seeds**; the area winner gets one **60-epoch** run before entering the running config. Budget per area ≈ screening (4–6 runs) + confirmation (2–3 × 3 seeds) ≈ 10–15 IEMOCAP runs ≈ 6–10 GPU-hours.

### 0.3 Per-class + pair telemetry in `run_ablation` (one-time edit)

After each eval, log `f1_score(yt, yp, average=None)` into history, and two derived numbers: `pair_hap_exc = (f1[hap]+f1[exc])/2` and `pair_ang_fru = (f1[ang]+f1[fru])/2`. M3-EACL and several M4/M6 verdicts are judged on these, not on wF1 alone.

### 0.4 McNemar significance on stored predictions (free, no GPU)

```python
from statsmodels.stats.contingency_tables import mcnemar
def mcnemar_vs(res, a='A0_Full_Model', b=None):
    yt = np.array(res[a]['y_true']); pa = np.array(res[a]['y_pred']); pb = np.array(res[b]['y_pred'])
    ca, cb = pa == yt, pb == yt
    tbl = [[(ca & cb).sum(), (ca & ~cb).sum()], [(~ca & cb).sum(), (~ca & ~cb).sum()]]
    return mcnemar(tbl, exact=False, correction=True).pvalue
```
Run on every screened config before deciding to confirm — it costs nothing and filters noise candidates early.

---

## M2 — Modality balance

### Evidence anchor
OGM-GE (Peng et al., CVPR 2022): dominant-modality gradients suppress weaker encoders under joint training; modulation recovers +2–4 pts on multimodal benchmarks. SPCL (2025) verifies the imbalance and training-side fixes on this exact graph-MERC family (MMGCN/MM-DFN/DialogueGCN), avg +2.25 wF1. Ada²I (ACM MM 2024) validates feature/modality adaptive weighting on IEMOCAP/MELD. The architecture here has *stronger* text features (RoBERTa) than those baselines → dominance is more likely, not less.

### Step 0 (mandatory diagnostic): unimodal probes

Three linear heads on the *post-encoder* per-modality streams, trained on detached features so they read but don't steer the backbone:

```python
class ModalityProbes(nn.Module):
    def __init__(self, d, n_classes):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(d, n_classes) for _ in range(3)])
    def forward(self, ht, ha, hv, mask):           # each [B,L,d]
        outs = []
        for head, h in zip(self.heads, (ht, ha, hv)):
            outs.append(head(h.detach()))           # detach: diagnostic only
        return outs                                  # 3 × [B,L,C]
```

Training: `loss += 0.1 * Σ CE(probe_logits[valid], y)` (probe params only receive gradient). Log per-epoch probe wF1 per modality. **Read-out:** if `wF1_text_probe ≈ wF1_full_model`, audio/visual contribute ~nothing and M2 is the highest-leverage area in this plan; the probe-gap is also each option's target metric (a balancing method must *close the gap*, not just move wF1 — if it moves wF1 without closing the gap, it's generic regularization and should be relabeled as such).

### Option 2A — OGM-GE (recommended first; training-side only)

Three-modality extension. Per batch, modality confidence `s_m = Σ_i softmax(probe_logits_m[i])[y_i]` (computed **with** gradient flowing to probes but detached from backbone as above). Dominance ratio per modality: `ρ_m = s_m / mean(s_{m'≠m})`. After `loss.backward()`, scale gradients of *modality-specific* parameters — in this codebase: `encoder.t_proj, encoder.t_enc, encoder.pe` (text); `encoder.a_proj` (audio); `encoder.v_proj` (visual):

```python
def ogm_ge_step(model, ratios, alpha=0.5):
    groups = {'t': [model.encoder.t_proj, model.encoder.t_enc],
              'a': [model.encoder.a_proj], 'v': [model.encoder.v_proj]}
    for m, mods in groups.items():
        rho = ratios[m]
        if rho <= 1.0:               # not dominant — leave untouched
            continue
        k = 1.0 - torch.tanh(torch.tensor(alpha * (rho - 1.0))).item()
        for mod in mods:
            for p in mod.parameters():
                if p.grad is not None:
                    p.grad.mul_(k)
                    p.grad.add_(torch.randn_like(p.grad) * p.grad.std() * 1e-1)  # GE term
```

Call between `loss.backward()` and `optim.step()`. Notes: the GE noise term is OGM-GE's generalization-enhancement; modality-shared parameters (graph, fusion, classifier) are deliberately untouched, which is the faithful adaptation of the paper's encoder-level modulation to this architecture.

### Option 2B — Ada²I AFW + AMW

**AFW** — per-feature sigmoid gates on each stream, before `flatten_batch`:

```python
class AFW(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.ModuleList([nn.Linear(d, d) for _ in range(3)])
    def forward(self, ht, ha, hv, lengths):
        outs = []
        for gate, h in zip(self.g, (ht, ha, hv)):
            pooled = h.sum(1) / lengths.clamp(min=1)[:, None]      # masked mean [B,d]
            outs.append(torch.sigmoid(gate(pooled))[:, None, :] * h)
        return outs
```

**AMW** — per-modality logits + learned weights at the head. `CrossModalAttn` already produces three d-dim pieces (`f̂_t, a_proj, v_proj`); add three linear heads and a weighting MLP:

```python
class AMWHead(nn.Module):
    def __init__(self, d, n_classes):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(d, n_classes) for _ in range(3)])
        self.w = nn.Sequential(nn.Linear(3 * d, 64), nn.GELU(), nn.Linear(64, 3))
    def forward(self, pieces):                      # list of 3 × [N,d]
        logits = [h(p) for h, p in zip(self.heads, pieces)]
        w = torch.softmax(self.w(torch.cat(pieces, -1)), -1)      # [N,3]
        return sum(w[:, i:i+1] * logits[i] for i in range(3)), w
```

**Telemetry (mandatory):** log mean `w` per epoch. Collapse to `w_text → 1` is the documented failure mode — if observed, 2B alone is insufficient and 2A becomes mandatory alongside.

### Option 2C — Modality dropout (complements either)

In `_features`, per dialogue with `p = ab.mod_dropout_p` (0.15), zero one non-text stream (or any stream, ablate both rules) *after* the encoder, before the graph. Grounding: missing-modality robustness training (GCNet-style); forces non-dropped pathways to carry signal. 3 lines.

### M2 ablation grid

| Run | Config | Judged on |
|---|---|---|
| G0 | baseline + probes | probe gap (diagnostic anchor) |
| G1 | + OGM-GE (α ∈ {0.3, 0.5} screen) | wF1 + probe-gap closure |
| G2 | + AFW | wF1 |
| G3 | + AFW + AMW | wF1 + AMW weight telemetry |
| G4 | + modality dropout 0.15 | wF1 |
| G5 | best of {G1, G3} + G4 | stacked confirmation, 3 seeds |

Interaction flag: M2 changes which fusion wins → **after the M2 winner locks, re-screen the M6 grid once** (single seed) before finalizing M6.

---

## M3 — Loss design: imbalance + similar-emotion separation

### Evidence anchor
Effective-number weighting (Cui et al., CVPR 2019; used by ConxGNN on MELD). BCL (Zhu et al., CVPR 2022): SupCon-style losses provably bias toward head classes; class-averaging + class-complement restores balanced geometry. EACL (Yu et al., 2023): anchor-based separation of the documented IEMOCAP hard pairs (hap↔exc, ang↔fru — confirmed failure modes in COGMEN/CORECT error analyses). Label smoothing redistributes target mass uniformly and partially cancels minority up-weighting — an interaction, not a free regularizer.

### Option 3A — Smoothing × CB grid (run FIRST: it re-baselines every later loss decision)

Four runs, single seed each (cheap): `{ls=0, ls=0.1} × {cb on, off}`. Both knobs already exist (`CBCELoss(weights, label_smoothing)`). Judged on **mF1 + minority-class F1** (hap on IEMOCAP), not wF1 — the interaction's symptom is minority-class dilution. The winning cell becomes the base for everything below.

### Option 3B — BCL replacing CBFC

Class-averaged positives, class-complement denominator (the two corrections over SupCon):

```python
class BCLLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__(); self.t = temp
    def forward(self, z, y, dia=None):              # z [N,d] L2-normalized, y [N]
        z = F.normalize(z, dim=-1)
        sim = z @ z.T / self.t
        classes = y.unique()
        loss, count = z.new_zeros(()), 0
        for i in range(len(y)):
            pos = (y == y[i]); pos[i] = False
            if pos.sum() == 0: continue
            # denominator: average within each class, then sum over classes (class-complement)
            denom = sum(torch.exp(sim[i][y == c]).mean() for c in classes)
            num = torch.exp(sim[i][pos]).mean()      # class-averaged positives
            loss = loss - torch.log(num / denom); count += 1
        return loss / max(count, 1)
```

(Vectorize with scatter ops for speed; the loop form documents the math.) Same call signature as `CBFCLoss` → a `contrastive` flag swap. Optionally keep the within-dialogue restriction (`dia`) for comparability with CBFC — ablate both scopes.

### Option 3C — EACL anchors

```python
class EACLLoss(nn.Module):
    def __init__(self, n_classes, d, temp=0.1, margin=0.3, lam_sep=0.1):
        super().__init__()
        self.anchors = nn.Parameter(F.normalize(torch.randn(n_classes, d), dim=-1))
        self.t, self.m, self.lam = temp, margin, lam_sep
    def forward(self, z, y):
        a = F.normalize(self.anchors, dim=-1)
        logits = F.normalize(z, dim=-1) @ a.T / self.t
        l_pull = F.cross_entropy(logits, y)                      # pull to own anchor
        d_aa = a @ a.T - 2 * torch.eye(len(a), device=a.device)  # anchor separation
        l_sep = F.relu(self.m - (1 - d_aa)).triu(1).mean()       # keep anchors apart
        return l_pull + self.lam * l_sep
```

`loss += ab.eacl_lambda * eacl(z, y)`. **Judged on `pair_hap_exc` and `pair_ang_fru` F1** (§0.3) — that is its job; a wF1-flat, pair-F1-up result is a keep. Telemetry: log the anchor cosine matrix per epoch; hap/exc anchor distance should *grow* — if it doesn't, the loss isn't doing what it claims regardless of metrics.

### Option 3D — CBFC γ sweep (hygiene)
`{0, 1, 2}` on the 3A-corrected base. γ=1 is ConxGNN's effective setting; γ=0 ≈ SupCon tests whether focality matters at all.

### M3 ablation grid

| Run | Config |
|---|---|
| L0–L3 | 3A grid (4 cells, 1 seed) → pick base |
| L4 | base + CBFC γ∈{0,1,2} (screen) |
| L5 | base + BCL (dialogue-scoped and global-scoped) |
| L6 | base + EACL (on top of L4/L5 winner) |
| L7 | winner stack, 3 seeds |

Interaction cell: **EACL × {CBFC, BCL}** — all shape the same embedding space; if L6's pair-F1 gain disappears when the contrastive term is on, they're redundant and the cheaper one stays.

---

## M4 — Implicit edge quality

### Evidence anchor
HRG-SSA (IJCAI 2025): per-modality implicit edges are a credited component of the IEMOCAP ceiling — and the detectors are **modality-specific** by design (relations differ per modality). AdaIGN (AAAI 2024): Gumbel-Softmax differentiable edge selection beats fixed thresholds. Counterfactual-attention literature (Goyal et al. 2021): similarity scores correlation, not influence — the conceptual upgrade, untested in MERC. The mandatory null row: in a graph that already has windowed + cross-modal edges, implicit edges must beat their own absence.

### Option 4A — Per-modality detectors + score-once caching

```python
# in InceptionGraphModule.__init__
self.implicit = nn.ModuleList([ImplicitEdgeDetector(d) for _ in range(3)])

# in forward — BEFORE the branch loop (scores are window-independent):
imp_edges = None
if self.ab.implicit_mode != 'off':
    imp_edges = [self.implicit[m].extra_edges(feats_m, lengths, offsets)
                 for m, feats_m in enumerate((ht, ha, hv))]
for (p, f), branch in zip(active_windows, active_branches):
    edge_index = build_igm_graph(..., implicit_edges=imp_edges)   # signature change: takes cached lists
```

`build_igm_graph` change: accept precomputed edge lists instead of the detector object. Two effects: restores HRG-SSA's modality-adaptive property; halves IGM graph-build time (no per-branch recompute).

### Option 4B — Gumbel top-k selection

Inside `extra_edges`, replace the `a > 1/n` threshold:

```python
k = max(2, n // self.topk_div)                      # per target node
if self.training:
    scores = scores - torch.log(-torch.log(torch.rand_like(scores)))  # Gumbel noise
keep = scores.topk(k, dim=-1).indices               # per-row top-k over causal candidates
```

Fixed, controlled edge budget (the threshold rule's edge count varies wildly with n); differentiable selection pressure at train via the noise. Telemetry: log kept-edge counts per modality per epoch — a detector keeping >30% of all causal pairs is a noise source regardless of metrics.

### Option 4C — CAD (counterfactual) scoring — closed-form, no second forward

For a **single-head attention scorer**, leave-one-out outputs are analytic: with unnormalized weights `e_k = exp(q_j·k_k/√d)` and `out_j = Σ e_k v_k / Σ e_k`,

```python
# cached per j: e [n_cand], v [n_cand, d], out_j [d]
def cad_scores(e, v, out_j):
    S = e.sum()
    out_wo = (out_j[None] * S - e[:, None] * v) / (S - e)[:, None]   # out_j without candidate i
    return (out_j[None] - out_wo).norm(dim=-1)                        # influence of i on j
```

So CAD costs one attention pass plus an O(n_cand·d) vector op — the "double forward" objection disappears for this scorer class. Candidates: causal window ∪ top-32 similarity pairs. Selection: Gumbel top-k on the influence scores (4B machinery reused). Honest paper framing: representation-level influence proxy, not causal inference.

### M4 ablation grid

| Run | `implicit_mode` |
|---|---|
| E0 | off *(the reference — not the shared-detector baseline)* |
| E1 | shared (current) |
| E2 | per_modality (4A) |
| E3 | per_modality + gumbel (4A+4B) |
| E4 | cad (4C, only if E2/E3 beat E0) |

Decision: every variant is judged **against E0**, not against E1 — the question is whether implicit edges earn their place at all, then which design earns it best. Confirm the top candidate at 3 seeds; carry edge-count telemetry into the paper's analysis section (it's the mechanism evidence reviewers ask for).

---

## M6 — Fusion

### Evidence anchor
GraphSmile (TPAMI 2025): fusion *ordering* matters — alternating inter/intra-modal aggregation avoids "fusion conflict" from concurrent heterogeneous aggregation; its ablation credits the alternation. Ada²I: learned modality weighting at the head. The parameter-free mean is the mandatory control: any fusion that can't beat a mean is not earning its parameters.

### Option 6A — Symmetric pairwise attention

Extend `CrossModalAttn` from {V→T, A→T} to the full set {V→T, A→T, T→A, V→A, T→V, A→V} (six heads, or three bidirectional):

```python
# per modality m: f̂_m = proj_m(x_m) + Σ_{m'≠m} attn_{m'→m}(proj_m(x_m), proj_{m'}(x_{m'}))
# output: cat([f̂_t, f̂_a, f̂_v])  — same 3d width as current, head unchanged
```

~15 lines by generalizing the existing two attention calls into a loop; +~0.5M params.

### Option 6B — Gated additive fusion (lighter; doubles as M2 telemetry)

```python
class GatedFusion(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(3 * d, 64), nn.GELU(), nn.Linear(64, 3))
    def forward(self, t, a, v):                      # each [N,d] (projected pieces)
        w = torch.softmax(self.w(torch.cat([t, a, v], -1)), -1)
        fused = w[:, 0:1] * t + w[:, 1:2] * a + w[:, 2:3] * v
        return torch.cat([fused, t, a, v], -1), w    # keep residual pieces for the head
```

Log mean `w` per epoch — the same collapse telemetry as AMW (they can share a module if both are adopted).

### Option 6C — Alternating inter/intra schedule inside IGM (GraphSmile's mechanism, imported without its scaffold)

Split the edge set built in `build_igm_graph` into `E_inter` (cross-modal edges) and `E_intra` (intra-modal window + implicit), and alternate per layer inside the branch (with `igm_layers=2`, layer 0 = inter, layer 1 = intra; screen the reverse order too):

```python
for li, layer in enumerate(branch.layers):
    ei = E_inter if (li % 2 == 0) else E_intra
    h = layer(h, ei, edge_attr[ei_slice]) + h        # residual as in current branch
```

**6C+** (the full GraphSmile claim): extend `E_inter` with *cross-utterance* cross-modal pairs within the window — current cross-modal edges are same-utterance only; GraphSmile's specific architectural argument is that same-utterance-only cross-modal graphs miss inter-utterance heterogeneous cues. ~10 extra lines in the edge builder; screen 6C and 6C+ separately so the schedule and the edge-set extension are attributed independently.

### M6 ablation grid

| Run | `fusion_mode` / `igm_schedule` |
|---|---|
| F0 | mean *(parameter-free control)* |
| F1 | text_anchor (current) |
| F2 | pairwise (6A) |
| F3 | gated (6B) |
| F4 | best of F0–F3 + alternating (6C) |
| F5 | F4 + cross-utterance inter edges (6C+) |

Decision: F1–F3 must beat F0 to keep their parameters. 6C/6C+ are judged independently (schedule vs edge-set). **Re-screen this grid once (single seed) after the M2 winner locks** — balancing changes which fusion wins, and locking M6 before M2 risks crowning a fusion that was compensating for imbalance.

---

## Cross-area run order and budget

| Phase | Runs | Purpose |
|---|---|---|
| P1 | M3-3A grid (4 × 1 seed) + M2 probes diagnostic (1 run) | re-baseline the loss; measure modality gap |
| P2 | M2 grid G1–G5 (screen 1 seed → confirm 3 seeds) | balancing winner |
| P3 | M6 grid F0–F5 on the M2 winner | fusion winner (post-balance, as required) |
| P4 | M4 grid E0–E4 | implicit-edge verdict vs the off-row |
| P5 | M3 L4–L7 (contrastive/anchors on corrected base) | loss stack winner |
| P6 | Full stack, 3 seeds × 30 ep + 1 × 60 ep + McNemar vs baseline | final attribution table |

Estimated total: ~35–45 IEMOCAP runs ≈ 25–35 GPU-hours with screen-then-confirm; MELD repeats only P6 plus any area whose verdict is plausibly dataset-dependent (M2 — MELD's class skew interacts with modality balance; M6 — shorter dialogues change the fusion picture). Every kept option enters the paper with: its grid row, its 3-seed mean±std, its McNemar p-value vs the running base, and its mechanism telemetry (probe gap / anchor distances / edge counts / modality weights) — the four pieces of evidence that make an ablation table withstand review.
