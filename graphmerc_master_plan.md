# GraphMERC — Complete Implementation Plan
## A graph-based MERC system: high accuracy, common problems addressed, overfitting controlled

*Master blueprint synthesizing: the corrected 27-paper survey (DialogueGCN → HRG-SSA), code-level analysis of GraphSmile and HRG-SSA, the arch21/HyFIN-Net build and its ablation campaign, and the focused M2/M3/M4/M6 plans. Decisions are committed (with fallbacks), not left as open menus. Every mechanism is traceable to published evidence or to a verified gap.*

---

# 1. Problem definition and targets

**Task.** Utterance-level emotion classification in multi-turn, multi-speaker conversation, multimodal (text + audio + visual). Benchmarks: IEMOCAP (6-way, sessions 1–4 train / 5 test) and MELD (7-way, official split).

**Reference ceilings (verified):** HRG-SSA (IJCAI 2025) 75.47 wF1 IEMOCAP / 66.83 MELD; GraphSmile (TPAMI 2025) 72.81 / 66.71; M3Net-RoBERTa (CVPR 2023) 72.49 / 67.05.

**Realistic targets for this system:**
| Dataset | Floor (must reach) | Target | Stretch |
|---|---|---|---|
| IEMOCAP wF1 | 69 | 71–72.5 | 73+ |
| MELD wF1 | 64 | 65.5–67 | 67.5+ |

Targets assume the GraphSmile feature pickles (the current constraint). The architecture is feature-agnostic; swapping in stronger features (WavLM audio, temporal SigLIP2⊕AU visual) later raises every number by an estimated +1–3 (M3Net's GloVe→RoBERTa ablation shows encoder swaps alone move 4–7 points on text; audio/visual swaps are smaller but real).

**The five common problems this design must address, with their evidence base:**
1. **Modality imbalance** — text dominates; verified on this exact graph family (SPCL 2025: avg +2.25 wF1 from fixing it; OGM-GE CVPR 2022 mechanism)
2. **Class imbalance** — MELD: neutral ~47%, fear/disgust ~1%; ConxGNN/DER-GCN address with CB-focal/contrastive minority losses
3. **Similar-emotion confusion** — hap↔exc, ang↔fru documented in COGMEN/CORECT error analyses; EACL anchors target it
4. **Shallow speaker modeling** — the field's most prominent under-used signal: GraphSmile ignores its own loaded speaker masks; HRG-SSA stops at binary same-speaker edges; **no published MERC method represents speaker as a graph node** (verified gap → our claim)
5. **Overfitting** — IEMOCAP train ≈ 5K utterances; deep multi-module graph stacks overfit fast (the field's hidden failure mode: single-seed reporting hides it)

---

# 2. System architecture (committed design)

```
                GraphSmile pickles (t/a/v per utterance, speaker, labels)
                                      │
        ┌─────────────────────────────┴───────────────────────────┐
        │ ENCODER                                                  │
        │  text:   Linear→PE→1-layer TransformerEncoder (masked)   │
        │  audio:  Linear+GELU+Dropout                              │
        │  visual: Linear+GELU+Dropout                              │
        │  + speaker embedding (added to all streams)               │
        │  + AFW feature gates (per-modality, sigmoid)   [M2]       │
        └─────────────────────────────┬───────────────────────────┘
                                      │  3 nodes/utterance + S speaker meta-nodes/dialogue
        ┌─────────────────────────────┴───────────────────────────┐
        │ GRAPH — single-branch windowed heterogeneous graph        │
        │  Edges:                                                   │
        │   E1 intra-modal window (past w_p, future w_f)            │
        │   E2 cross-modal same-utterance (t↔a, t↔v, a↔v)           │
        │   E3 cross-modal cross-utterance within window  [GSmile+] │
        │   E4 same-speaker (window-extended, edge-typed)           │
        │   E5 speaker meta-node hub edges (bidirectional)  [NOVEL] │
        │   E6 implicit per-modality edges (Gumbel top-k) [if >off] │
        │  Propagation: TransformerConv (4 heads, edge_dim=3:       │
        │   angular weight, is_same_speaker, is_inter_modal),       │
        │   2 layers, alternating inter/intra schedule [if >joint], │
        │   residual + LayerNorm                                    │
        └─────────────────────────────┬───────────────────────────┘
                                      │  per-utterance modality pieces
        ┌─────────────────────────────┴───────────────────────────┐
        │ FUSION — gated additive (learned per-utterance weights)   │
        │  + residual modality pieces → [fused ‖ t ‖ a ‖ v]         │
        │  (pairwise-attention variant kept as grid fallback)       │
        └─────────────────────────────┬───────────────────────────┘
                                      │
        ┌─────────────────────────────┴───────────────────────────┐
        │ HEADS                                                     │
        │  main classifier (2-layer MLP)                            │
        │  3 unimodal probes (detached) — telemetry + OGM-GE input  │
        │  SDP shift head (MELD; λ=0.3)                  [M7-lite]  │
        └───────────────────────────────────────────────────────────┘

LOSS:  CB-CE  +  μ·contrastive(BCL or CBFC, grid-decided)  +  λe·EACL  +  λs·SDP
TRAIN: AdamW, warmup+cosine, grad clip, OGM-GE gradient modulation,
       modality dropout p=0.15, early stop on val wF1, 3–5 seeds
```

### What was deliberately removed and why
The architecture is intentionally lean. The hypergraph module, the fully-connected multi-frequency module, the second IGM branch, the input dialogue-LayerNorm, and threshold-based shared implicit edges are all **excluded from the core** — each was either implemented with structural defects in the donor design (batch-position-indexed hypergraph weights), functionally redundant with the windowed graph (full-connectivity modules), or evidence-weak relative to cost. Each remains available as a grid row if a reviewer asks, but the design philosophy follows the field's own lesson: the most-cited recent MERC papers each have one clearly-articulated mechanism, not six. Fewer modules = fewer parameters = less overfitting surface = cleaner attribution.

### The novelty claims (each with a verified gap)
1. **Speaker meta-node (E5).** One learnable hub node per speaker per dialogue, bidirectionally attention-connected to all that speaker's utterance nodes. Simultaneously: a learnable speaker emotional persona (generalizing DialogueRNN's party-GRU into a bidirectional graph-native form) and a 2-hop long-range pathway that windowed graphs structurally lack. Verified unpublished in MERC: GraphSmile ignores speaker entirely (code-confirmed dead `qmask` argument); HRG-SSA stops at binary same-speaker edges; DialogueGCN/MMGCN use relation types/embeddings, never nodes.
2. **Shift-as-edge-type (MELD).** Sentiment-flip candidate edges as graph topology rather than only an auxiliary loss — unpublished placement of a published signal (GraphSmile's SDP).
3. **Reserve (conceptual, optional): closed-form counterfactual edge scoring (CAD)** — leave-one-out influence for a single-head attention scorer is analytic (no second forward), making counterfactual implicit edges tractable; pursued only if similarity-based implicit edges beat the off-row first.

---

# 3. Module-by-module specification

## 3.1 Encoder
- Projections to d=256 (IEMOCAP) / 256–384 (MELD); GELU + dropout 0.5.
- Text: sinusoidal PE + 1-layer TransformerEncoder with padding mask. (PE and the speaker embedding are the two most load-bearing encoder components — both validated across the lineage from DialogueRNN onward; never remove.)
- Speaker embedding: `nn.Embedding(n_speakers, d)` added to all three streams.
- **AFW gates** (Ada²I): per-modality per-feature sigmoid gates from masked-mean pooled stream. Cheap, composes with OGM-GE.

## 3.2 Graph construction
Per dialogue of n utterances: 3n modality nodes + S speaker meta-nodes (S = active speakers).

| Edge set | Rule | Source |
|---|---|---|
| E1 | intra-modal, |i−j| ≤ (w_p=5, w_f=3) IEMOCAP / (7,4) MELD | MM-DFN/ConxGNN window convention |
| E2 | t_i↔a_i, t_i↔v_i, a_i↔v_i | MMGCN convention |
| E3 | cross-modal, cross-utterance within window | GraphSmile's specific architectural claim (same-utterance-only cross-modal graphs miss inter-utterance heterogeneous cues) |
| E4 | same-speaker pairs, |i−j| ≤ 2×window, edge-typed | HRG-SSA explicit edge set; DialogueGCN speaker relations |
| E5 | speaker hub ↔ all that speaker's nodes (all modalities) | heterogeneous-graph virtual nodes (HGT/HAN); novel in MERC |
| E6 | per-modality implicit, Gumbel top-k (k = max(2, n//8)) | HRG-SSA detectors + AdaIGN selection; **included only if it beats the off-row** |

`edge_attr` ∈ R³: (angular similarity `1−arccos(cos)/π` [MMGCN], is_same_speaker, is_inter_modal). `TransformerConv(edge_dim=3, heads=4, beta=True)`.

**Over-smoothing guards for E5 hubs:** attention-weighted hub edges (free via TransformerConv); hubs excluded from any future fully-connected module; hub representations excluded from classification readout.

## 3.3 Propagation
2 layers, residual + LayerNorm. Two schedules, grid-decided: `joint` (all edges every layer) vs `alternating` (layer 0: inter-modal E2+E3; layer 1: intra-modal E1+E4+E6; hubs E5 in both). The alternating schedule is GraphSmile's fusion-conflict mechanism imported without its scaffold — its ablation credits the alternation, so it earns a grid row, not blind adoption.

## 3.4 Fusion
**Committed: gated additive** — w = softmax(MLP([t;a;v])) per utterance; output [fused ‖ t ‖ a ‖ v]. Chosen over text-anchored attention because (a) symmetric (no structural text bias compounding the imbalance problem), (b) the gate weights are free modality-balance telemetry, (c) parameter-light. The full pairwise-attention variant is the grid fallback; the parameter-free mean is the control both must beat.

## 3.5 Heads and losses

```
L = L_CBCE + μ·L_con + λe·L_EACL + λs·L_SDP + 0.1·L_probes
```

| Term | Spec | Evidence | Default |
|---|---|---|---|
| L_CBCE | CE with effective-number weights (β=0.999); label smoothing per the 4-cell grid (ls × CB interaction is real: smoothing dilutes minority up-weighting) | Cui et al. CVPR 2019; ConxGNN | ls grid-decided |
| L_con | BCL (class-averaged positives, class-complement denominator) vs CBFC γ∈{1,2}, grid-decided | Zhu et al. CVPR 2022 (SupCon's head-class bias is provable); ConxGNN | μ=0.1 |
| L_EACL | learnable unit-norm per-class anchors; pull-to-own + anchor-separation margin | Yu et al. 2023; targets documented hap↔exc / ang↔fru | λe=0.2 |
| L_SDP | binary sentiment-shift head on consecutive pairs; MELD real labels (already in the pickle, currently discarded); IEMOCAP derived polarity flagged as caveat | GraphSmile's credited second contribution | λs=0.3, MELD |
| L_probes | detached unimodal probe CE — telemetry + OGM-GE inputs | OGM-GE requirement | 0.1 |

## 3.6 Training-side balance and regularization (the anti-overfit layer)

| Mechanism | Spec | Why |
|---|---|---|
| **OGM-GE** | per-batch dominance ratio from probe confidences; scale dominant modality's encoder-parameter gradients by 1−tanh(α(ρ−1)), α=0.5, + GE noise | CVPR 2022; SPCL family verification. Training-side only, zero inference cost |
| **Modality dropout** | p=0.15 per dialogue, zero one stream post-encoder | forces non-text pathways; doubles as regularizer (GCNet-style robustness training) |
| **Capacity discipline** | d=256, 2 graph layers, single branch, ~2–3M params | IEMOCAP train ≈ 5K utterances; every removed module is removed overfitting surface |
| **Dropout 0.5 + weight decay 1e-4 + grad clip 1.0** | as validated in the lineage | standard, load-bearing at this data scale |
| **Early stopping** | patience 10 on val wF1; best-val checkpoint only | never select on test |
| **Label smoothing** | only if the 4-cell grid says it doesn't fight CB weights | known interaction |
| **Optional (grid row): masked-node reconstruction** | mask 15% node features pre-graph, MSE reconstruction head | DER-GCN SMGAE node branch; small-data regularizer; highest overfit-confusion risk → judged strictly by val-gap telemetry |
| **Overfit telemetry (mandatory)** | per-epoch train−val loss gap + best-val epoch drift logged in every run; mechanical rule: val loss rising 3 consecutive epochs while train falls, with earlier best-val epoch than baseline → flag | makes the "does it overfit" question answerable per-option, not impressionistic |

---

# 4. Evaluation protocol (what makes the numbers trustworthy)

1. **Splits.** IEMOCAP: sessions 1–4 train (10% carved val, frozen to disk per seed), session 5 test. MELD: official train/dev/test.
2. **Seeds.** Screen at 1 seed / 30 epochs; confirm candidates (within 0.3 wF1 of grid best) at 3 seeds; final table at 5 seeds, mean±std. Single-seed deltas <0.5 wF1 are treated as noise — by protocol, not by judgment call.
3. **Metrics.** wF1 (primary, checkpoint selection on val), mF1, per-class F1, pair-F1 (hap/exc, ang/fru), shift-utterance F1 (MELD), long-dialogue stratified accuracy (IEMOCAP >40 utterances), train−val gap.
4. **Significance.** McNemar's test on stored test predictions for every kept option vs the running base (free; predictions already persisted per run).
5. **Mechanism telemetry per claim.** Probe-gap closure for balancing; gate/AMW weight distributions for fusion; anchor cosine matrix for EACL; edge counts per modality for implicit edges; hub attention mass for the meta-node. A mechanism that improves wF1 without moving its own telemetry is relabeled as generic regularization — before a reviewer does it for you.
6. **Comparison hygiene** (from the corrected survey): report feature pipeline explicitly; never mix original-paper and re-implementation baselines without flagging; both wF1 and mF1 on MELD always (wF1 hides minority-class failure); flag any non-standard protocol (e.g., MELD-1).

---

# 5. Build phases with decision gates

**Phase 0 — Foundation (week 1).** Lean core: encoder + E1/E2/E4 edges + gated fusion + CB-CE + contrastive. Probes attached. 3-seed baseline on both datasets. *Gate: IEMOCAP wF1 ≥ 68 to proceed (else debug, don't add).*

**Phase 1 — Loss correction + balance (week 2).** The 4-cell ls×CB grid (re-baselines everything); OGM-GE + modality dropout; BCL-vs-CBFC grid; EACL on the winner. *Gate: probe gap reduced AND wF1 not degraded; EACL kept only if pair-F1 rises.*

**Phase 2 — The novelty (weeks 3–4).** E5 speaker meta-node, against the S-grid: {embed only, +E4 edges, +E5, E4+E5, E5 w/o embed}. Judged on overall wF1 **and** long-dialogue stratified accuracy and hub-attention telemetry. *Gate: E5 must beat E4 (the published level) to carry the novelty claim; expect IEMOCAP > MELD effect (2 long-dialogue speakers vs ≤9 short-dialogue speakers — a MELD-flat result is reportable, not fatal).*

**Phase 3 — Graph refinements (week 5).** E3 cross-utterance inter-modal edges; alternating-vs-joint schedule; E6 implicit edges vs the **off-row** ({off, per-modality, +Gumbel}; CAD only if implicit > off). Fusion grid re-screened once post-balance (locking fusion before balance crowns compensators).

**Phase 4 — MELD specifics (week 6).** SDP head (real labels) and shift-edges; judged on shift-utterance F1, not just wF1. CB-loss settings re-checked under MELD's skew.

**Phase 5 — Final (week 7).** Greedy stack of all gate-passing options, re-verified additively (sub-additivity is the norm — every addition must still clear its bar in combination); 5-seed final on both datasets; McNemar vs Phase-0 base; the attribution table (option → ΔwF1 ± std → p → mechanism telemetry) is the paper's Table 3.

**Contingencies.** Meta-node flat on both datasets → fall back to E4 + speaker hyperedge variant, reframe contribution around the systematic ablation + balance stack. Implicit edges lose to off → report the redundancy finding (informative in a windowed+cross-modal graph) and bank the compute. Stack sub-additivity eats the gains → the lean core + balance + losses alone is publishable as a rigorous-empiricism paper at an affective-computing venue; the survey shows the field rewards clean single-mechanism stories over kitchen sinks.

---

# 6. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Meta-node over-smooths to dialogue mean | medium | attention-weighted hub edges; hub excluded from readout; telemetry on hub attention entropy |
| OGM-GE destabilizes early training | low-medium | enable after warmup epoch; α=0.3 fallback |
| EACL redundant with BCL/CBFC | medium | explicit interaction cell; keep the cheaper |
| Implicit edges add noise (redundant with E1/E3) | medium-high | off-row is the reference; edge-count telemetry |
| MELD verdicts don't transfer from IEMOCAP | medium | Phase 4 re-checks; window/depth configs already dataset-specific |
| Sub-additive stacking erases headline gain | high (it's the norm) | greedy re-verification; honest per-option attribution; targets set assuming it |
| Single-seed mirages | high if protocol ignored | screen-confirm protocol + McNemar are mandatory, not optional |

---

# 7. What this plan delivers

A compact (~2–3M param) heterogeneous-graph MERC system with: speaker structure at three levels (embedding, typed edges, **meta-node** — the verified-gap contribution), evidence-backed modality balancing (OGM-GE + AFW + dropout, with probe-gap proof), an imbalance- and confusion-aware loss stack (CB-CE + BCL/CBFC + EACL, interaction-tested), optional implicit/inter-utterance edges that must beat their own absence, and an evaluation protocol (multi-seed, McNemar, mechanism telemetry, overfit telemetry) that makes every kept component defensible line-by-line in review. Expected landing zone: 71–72.5 wF1 IEMOCAP / 65.5–67 MELD on GraphSmile features, with a documented path (+1–3) via the stronger feature pipeline already built; every point of gain attributable to a named mechanism with a named source — which is precisely what the corrected survey shows the strongest recent MERC papers have in common.
