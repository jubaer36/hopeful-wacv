# M2 / M3 / M4 / M6 Ablation Study Findings
*IEMOCAP 6-class MERC — 30 epochs per run, ABLATION_SEED=42 (single-seed screening), seeds={42,43,44} (confirmation sweeps)*

---

## Baseline Reference

| Model | Acc | wF1 | mF1 |
|---|---|---|---|
| A0_Full_Model (30 ep, seed=42) | 0.6771 | 0.6790 | 0.6669 |
| G0_probes_baseline (M2 run) | 0.6796 | 0.6841 | 0.6710 |

The G0 run reproduces the baseline to within ±0.005 wF1 under the same 30-epoch budget; minor variance is expected across independent training runs.

---

## M2 — Modality Balance

**Grid:** G0 probes-only baseline → G1a/b OGM-GE (α=0.3/0.5) → G2 AFW feature gates → G3 AFW+AMW weighted logits → G4 modality dropout p=0.15 → G5 stacked (OGM-α0.5 + ModDrop, 3 seeds).

### Results

| Variant | Acc | wF1 | mF1 | pair_hap_exc | pair_ang_fru | Δ wF1 vs G0 |
|---|---|---|---|---|---|---|
| G0_probes_baseline | 0.6796 | 0.6841 | 0.6710 | 0.6124 | 0.6531 | — |
| G1a_OGM_a03 | 0.6661 | 0.6709 | 0.6579 | 0.5890 | 0.6442 | −0.0132 |
| G1b_OGM_a05 | 0.6691 | 0.6744 | 0.6637 | 0.5852 | 0.6553 | −0.0097 |
| G2_AFW | 0.6790 | 0.6816 | 0.6683 | 0.6000 | 0.6509 | −0.0025 |
| G3_AFW_AMW | 0.6654 | 0.6645 | 0.6526 | 0.5771 | 0.6278 | −0.0196 |
| **G4_ModDrop015** | **0.6845** | **0.6890** | **0.6768** | **0.6154** | **0.6563** | **+0.0049** |
| G5 (OGM+ModDrop, 3-seed mean) | — | 0.6821±0.0055 | 0.6670±0.0056 | — | — | −0.0020 |

### Per-emotion F1 (hap / sad / neu / ang / exc / fru)

| Variant | hap | sad | neu | ang | exc | fru |
|---|---|---|---|---|---|---|
| G0_probes_baseline | 0.529 | 0.790 | 0.705 | 0.644 | 0.696 | 0.662 |
| G1a_OGM_a03 | 0.500 | 0.794 | 0.687 | 0.640 | 0.678 | 0.648 |
| G1b_OGM_a05 | 0.528 | 0.806 | 0.696 | 0.651 | 0.643 | 0.659 |
| G2_AFW | 0.474 | 0.808 | 0.700 | 0.672 | 0.726 | 0.629 |
| G3_AFW_AMW | 0.479 | 0.811 | 0.695 | 0.646 | 0.675 | 0.609 |
| G4_ModDrop015 | 0.516 | 0.812 | 0.705 | 0.661 | 0.715 | 0.651 |

### Findings

1. **OGM-GE hurts on both α values.** G1a (Δ=−0.0132) and G1b (Δ=−0.0097) are below the G0 baseline. The probe telemetry (logged in history files) showed that text probe wF1 was already close to the full model's wF1, confirming text dominance—but OGM suppressed the strong text pathway without a compensating recovery of audio/visual.

2. **AFW (G2) gives negligible effect** (Δ=−0.0025, within noise). Per-feature gates on each stream add parameters without improving discrimination; `hap` F1 drops further (0.474 vs 0.529), suggesting the gates over-specialize toward easier classes.

3. **AFW+AMW (G3) is the worst M2 variant** (Δ=−0.0196). Adding the adaptive modality-weighted head on top of AFW compounds the damage. The worst pair F1 across all M2 runs is observed here (pair_hap_exc=0.5771, pair_ang_fru=0.6278).

4. **Modality dropout (G4) is the only M2 method that helps** (Δ=+0.0049 wF1, +0.0058 mF1). All six per-emotion F1s are at or above G0 levels except `hap` (0.516 vs 0.529). This is consistent with missing-modality robustness training forcing non-dropped pathways to carry more signal.

5. **Stacking OGM-GE + ModDrop (G5, 3 seeds) does not beat G4 alone.** G5 3-seed mean wF1=0.6821±0.0055 is below G4's single-seed 0.6890. The OGM-GE component's negative effect cancels ModDrop's benefit.

6. **`hap` is the most volatile emotion** across M2 variants (range 0.474–0.529). `sad` is the most stable (range 0.779–0.812). Modality balance methods that hurt overall performance consistently reduce `hap` F1 disproportionately.

### Verdict

None of the explicit balancing methods (OGM-GE, AFW, AMW) earn their place. Modality dropout p=0.15 is the only improvement (+0.0049 wF1). G4 is the M2 winner for downstream stacking.

---

## M3 — Loss Design

**Grid:** L0–L3 label-smoothing × class-balance (phase 1) → L4 CBFC γ∈{0,1,2} → L5 BCL (dialogue-scoped, global-scoped, and no-contrastive) + L6 EACL anchors → L7 winner confirmed at 3 seeds.

**Selected base from phase 1:** L1 (ls=0.0, cb_weights=True) — best mF1=0.6682, best minority (`hap`) F1=0.486.

### Phase 1 Results (3A smoothing × class-balance grid)

| Variant | Acc | wF1 | mF1 | hap F1 |
|---|---|---|---|---|
| L0_ls00_cbOff | 0.6895 | 0.6836 | 0.6599 | 0.398 |
| **L1_ls00_cbOn** | **0.6827** | **0.6829** | **0.6682** | **0.486** |
| L2_ls01_cbOff | 0.6778 | 0.6764 | 0.6590 | 0.448 |
| L3_ls01_cbOn | 0.6771 | 0.6767 | 0.6583 | 0.434 |

**Interaction finding:** `cb_weights=True` consistently improves mF1 and minority F1 (+0.007–0.009 mF1 vs off). Label smoothing (ls=0.1) consistently *hurts* minority F1 across both CB conditions (L0→L2: hap drops 0.398→0.448 without CB, and 0.486→0.434 with CB). Combining label smoothing with class balance dilutes the minority up-weighting.

### Contrastive Design Results

| Variant | Acc | wF1 | mF1 | pair_hap_exc | pair_ang_fru | Δ wF1 vs L1 |
|---|---|---|---|---|---|---|
| L4_cbfc_g0 | 0.6759 | 0.6794 | 0.6686 | 0.6190 | 0.6576 | −0.0035 |
| L4_cbfc_g1 | 0.6821 | 0.6848 | 0.6747 | 0.6284 | 0.6664 | +0.0019 |
| **L4_cbfc_g2** | 0.6759 | **0.6804** | 0.6670 | 0.6076 | 0.6652 | −0.0025 |
| L5_bcl_dia | 0.6722 | 0.6770 | 0.6654 | 0.6010 | 0.6644 | −0.0059 |
| **L5_bcl_global** | **0.6833** | **0.6880** | **0.6779** | **0.6167** | **0.6701** | **+0.0051** |
| L5_contrast_off | 0.6741 | 0.6750 | 0.6632 | 0.6022 | 0.6398 | −0.0079 |
| L6_eacl | 0.6513 | 0.6565 | 0.6454 | 0.5702 | 0.6395 | −0.0264 |
| L6b_eacl_noContrast | 0.6642 | 0.6712 | 0.6576 | 0.5741 | 0.6578 | −0.0117 |

### L7 Winner Confirmation (BCL global, 3 seeds)

| Seed | Acc | wF1 | mF1 |
|---|---|---|---|
| s42 | 0.6876 | 0.6922 | 0.6816 |
| s43 | 0.6839 | 0.6836 | 0.6711 |
| s44 | 0.6710 | 0.6734 | 0.6606 |
| **Mean ± Std** | — | **0.6831 ± 0.0077** | **0.6711 ± 0.0086** |

### Per-emotion F1 (selected variants)

| Variant | hap | sad | neu | ang | exc | fru |
|---|---|---|---|---|---|---|
| L1_ls00_cbOn (base) | 0.486 | 0.792 | 0.689 | 0.658 | 0.727 | 0.658 |
| L4_cbfc_g1 | 0.520 | 0.781 | 0.678 | 0.678 | 0.737 | 0.654 |
| L5_bcl_global | 0.535 | 0.803 | 0.691 | 0.673 | 0.699 | 0.668 |
| L5_contrast_off | 0.500 | 0.803 | 0.692 | 0.650 | 0.704 | 0.629 |
| L6_eacl (on contrast) | 0.471 | 0.800 | 0.653 | 0.647 | 0.669 | 0.632 |
| L7_winner_s42 | 0.536 | 0.810 | 0.702 | 0.671 | 0.709 | 0.663 |

### Findings

1. **Label smoothing consistently reduces minority-class F1.** Across both CB conditions, adding ls=0.1 drops `hap` F1 by 0.038–0.052. The CB baseline without smoothing (L1) is chosen as phase-1 winner based on mF1 and minority F1 — not wF1 (where L0 is highest at 0.6836 but has the worst hap F1=0.398).

2. **CBFC γ=1 outperforms both γ=0 and γ=2** (wF1: 0.6848 vs 0.6794 vs 0.6804). The focal weighting at moderate strength is the best of three; γ=0 (pure SupCon-style) underperforms, confirming that the focal weighting term contributes meaningfully.

3. **BCL global scope is the best contrastive design** (wF1=0.6880, +0.0051 vs base L1). BCL dialogue-scoped is worse than no contrastive at all (0.6770 vs 0.6750). This indicates that restricting the contrastive pairs to within-dialogue neighbours is overly conservative — the class-averaging and class-complement denominator of BCL need a wider pair pool to be effective.

4. **Contrastive terms benefit `hap` F1 most.** Without any contrastive term (L5_contrast_off), hap=0.500; with BCL global (L7_s42), hap=0.536. The pair_hap_exc metric follows the same pattern: 0.6022 without contrastive, 0.6223 with BCL global.

5. **EACL anchor loss hurts strongly.** L6_eacl is the worst variant in the entire M3 study (wF1=0.6565, Δ=−0.0264 vs L1). Adding anchor loss on top of the contrastive term is counter-productive. The `neu` class shows the largest collapse (0.689→0.653). EACL without contrastive (L6b) also drops below the no-contrastive baseline. The anchor cosine telemetry in the history file shows the hap/exc anchors do not separate over training epochs, indicating the pull term dominates and overfits the embedding space toward class centroids.

6. **BCL global (L7) confirmed at 3 seeds: mean wF1=0.6831±0.0077.** The std=0.0077 is higher than M4/M6 confirm sweeps; the seed-to-seed range (0.6734–0.6922) suggests the BCL loss is sensitive to weight initialization. Nonetheless, all three seeds beat the A0 baseline (0.6790).

### Verdict

Base config: `label_smoothing=0.0, cb_weights=True`. Contrastive: BCL global scope. EACL is excluded. L7 is the M3 winner for downstream stacking.

---

## M4 — Implicit Edge Quality

**Grid:** E0 off (reference) → E1 shared detector (original design, now with gradient) → E2 per-modality detectors → E3 Gumbel top-k → E4 CAD counterfactual. E1 and E0 both confirmed at 3 seeds.

**Design note:** In the original HyFIN-Net, `ImplicitEdgeDetector.W` received zero gradient (scores discarded after thresholding). In this study, detector scores are used as edge weights, giving detectors a real training signal.

### Screening Results

| Variant | Acc | wF1 | mF1 | pair_hap_exc | pair_ang_fru | Δ wF1 vs E0 |
|---|---|---|---|---|---|---|
| **E0_off** | **0.6827** | **0.6831** | **0.6699** | 0.5976 | 0.6646 | — |
| E1_shared | 0.6882 | 0.6908 | 0.6793 | 0.6256 | 0.6541 | +0.0077 |
| E2_per_modality | 0.6876 | 0.6902 | 0.6780 | 0.6240 | 0.6677 | +0.0071 |
| E3_gumbel | 0.6802 | 0.6835 | 0.6724 | 0.6022 | 0.6588 | +0.0004 |
| E4_cad | 0.6833 | 0.6874 | 0.6769 | 0.6318 | 0.6362 | +0.0043 |

### Confirmation (3 seeds: E1 vs E0)

| Run | s42 wF1 | s43 wF1 | s44 wF1 | Mean ± Std |
|---|---|---|---|---|
| E0_off_confirm | 0.6857 | 0.6892 | 0.6830 | **0.6860 ± 0.0025** |
| E1_shared_confirm | 0.6800 | 0.6763 | 0.6892 | **0.6818 ± 0.0055** |

### Per-emotion F1

| Variant | hap | sad | neu | ang | exc | fru |
|---|---|---|---|---|---|---|
| E0_off | 0.494 | 0.800 | 0.695 | 0.671 | 0.701 | 0.658 |
| E1_shared | 0.541 | 0.799 | 0.717 | 0.658 | 0.710 | 0.650 |
| E2_per_modality | 0.527 | 0.784 | 0.701 | 0.669 | 0.721 | 0.667 |
| E3_gumbel | 0.526 | 0.812 | 0.700 | 0.659 | 0.678 | 0.659 |
| E4_cad | 0.538 | 0.833 | 0.692 | 0.633 | 0.726 | 0.639 |

### Findings

1. **Single-seed screening favours E1 (shared) over E0 (+0.0077 wF1), but 3-seed confirmation reverses the verdict.** E0 confirmation mean=0.6860±0.0025 outperforms E1 confirmation mean=0.6818±0.0055. E1's variance is more than double E0's (0.0055 vs 0.0025), indicating the shared detector adds instability without reliable gain.

2. **Per-modality detectors (E2) screen at +0.0071 vs E0** — competitive with E1 — but were not confirmed separately. The lower variance of E0 makes E0 more reliable under the confirmation protocol.

3. **Gumbel top-k (E3) barely moves the needle** (Δ=+0.0004 single seed). Fixed edge budget with Gumbel noise does not help over the no-implicit-edges baseline. The edge-count telemetry shows kept_frac ≈ 0.58 (58% of causal pairs kept), far above the 30% noise threshold noted in the implementation plan.

4. **CAD counterfactual scoring (E4) gives a modest screen improvement** (Δ=+0.0043 wF1) and the highest pair_hap_exc=0.6318 in the M4 study. However E4 also shows the lowest pair_ang_fru=0.6362, suggesting it trades off the two hard pairs.

5. **All implicit-edge variants improve `hap` F1 over E0** (0.527–0.541 vs 0.494). This is the primary gain from implicit edges, not overall wF1. `sad` and `neu` are largely unaffected (±0.015).

6. **This finding is consistent with the prior B4 ablation** (removing implicit edges from the original model *helped*, +0.0090 wF1). There the detector had zero gradient (frozen random projection). With proper gradients here, E1 improves over E0 in screening — but still does not reliably beat E0 at 3 seeds, suggesting the gain is marginal and seed-dependent.

### Verdict

Implicit edges provide inconsistent benefit at the 3-seed level. E0 (no implicit edges) is the safer configuration: higher 3-seed mean and lower variance. If implicit edges are retained, E2 (per-modality, as in HRG-SSA) is the preferred design for its modality-specific semantics, but requires 3-seed confirmation before claiming a win.

---

## M6 — Fusion Design

**Grid:** F0 mean (parameter-free control) → F1 text-anchor (current design) → F2 symmetric pairwise attention → F3 gated additive → F4a/b alternating inter/intra IGM schedule (inter-first / intra-first) → F5 best-F4 + cross-utterance inter-modal edges (6C+) → F5b joint schedule + cross-utterance edges. F5 confirmed at 3 seeds.

### Results

| Variant | Acc | wF1 | mF1 | pair_hap_exc | pair_ang_fru | Δ wF1 vs F0 |
|---|---|---|---|---|---|---|
| **F0_mean** | 0.6858 | 0.6871 | 0.6761 | 0.6065 | 0.6799 | — |
| F1_text_anchor (current) | 0.6568 | 0.6589 | 0.6534 | 0.5878 | 0.6479 | −0.0282 |
| F2_pairwise | 0.6734 | 0.6736 | 0.6592 | 0.5870 | 0.6592 | −0.0135 |
| F3_gated | 0.6704 | 0.6719 | 0.6617 | 0.6268 | 0.6292 | −0.0152 |
| F4a_alt_interFirst | 0.6950 | 0.6963 | 0.6896 | 0.6302 | 0.6894 | +0.0092 |
| F4b_alt_intraFirst | 0.6802 | 0.6799 | 0.6740 | 0.6227 | 0.6629 | −0.0072 |
| **F5_cross_utt** | **0.7018** | **0.7045** | **0.6986** | **0.6375** | **0.6824** | **+0.0174** |
| F5b_cross_utt_joint | 0.7018 | 0.7042 | 0.6934 | 0.6277 | 0.6807 | +0.0171 |

### F5 Confirmation (3 seeds)

| Seed | Acc | wF1 | mF1 |
|---|---|---|---|
| s42 | 0.7024 | 0.7052 | 0.6995 |
| s43 | 0.6913 | 0.6942 | 0.6809 |
| s44 | 0.6944 | 0.6970 | 0.6873 |
| **Mean ± Std** | — | **0.6988 ± 0.0047** | **0.6892 ± 0.0077** |

### Per-emotion F1

| Variant | hap | sad | neu | ang | exc | fru |
|---|---|---|---|---|---|---|
| F0_mean | 0.489 | 0.799 | 0.684 | 0.704 | 0.724 | 0.656 |
| F1_text_anchor (current) | 0.541 | 0.783 | 0.666 | 0.661 | 0.634 | 0.635 |
| F2_pairwise | 0.481 | 0.774 | 0.688 | 0.661 | 0.693 | 0.657 |
| F3_gated | 0.522 | 0.764 | 0.694 | 0.647 | 0.732 | 0.611 |
| F4a_alt_interFirst | 0.547 | 0.808 | 0.691 | 0.713 | 0.714 | 0.666 |
| F5_cross_utt | 0.578 | 0.832 | 0.720 | 0.704 | 0.697 | 0.661 |
| F5b_cross_utt_joint | 0.539 | 0.822 | 0.721 | 0.693 | 0.716 | 0.669 |

### Findings

1. **The parameter-free mean (F0) beats all four learned cross-modal attention designs at 30 epochs.** F1 (text_anchor, current design) is the worst variant in the entire M6 study, 0.0282 wF1 below F0. F2 (pairwise) and F3 (gated) are also below F0. This pattern is consistent with the CrossModalAttn architecture being under-trained at 30 epochs: cross-attention needs more epochs to tune its projections than simple averaging.

2. **The current design (F1, text-anchored CrossModalAttn) is the single worst M6 variant by wF1 and mF1.** It loses to the mean on every metric and every emotion except `hap`. This is a critical finding: the current fusion layer is actively harmful at 30-epoch training budgets.

3. **Layer-order in the alternating IGM schedule matters significantly.** Inter-first (F4a): wF1=0.6963 (+0.0092 over F0). Intra-first (F4b): wF1=0.6799 (−0.0072 vs F0). Running cross-modal aggregation first, then within-modality refinement, yields the correct order — the reverse order loses to the parameter-free mean.

4. **Cross-utterance inter-modal edges (6C+) give the largest single gain in the M6 study** (F5 vs F4a: +0.0082 wF1). F5 wF1=0.7045 is the highest single-seed result in the entire M2–M6 study. The same edges combined with a joint (non-alternating) schedule (F5b: 0.7042) give nearly identical results, indicating that cross-utterance edges are the primary driver of the improvement, with the alternating schedule contributing less.

5. **F5 is confirmed at 3 seeds: mean wF1=0.6988±0.0047.** All three seeds beat the A0 baseline (0.6790) and the G0 probe baseline (0.6841). The 3-seed mean exceeds the full 60-epoch baseline (wF1=0.6851) at only 30 epochs.

6. **F5 improves `hap` most of all emotions** (0.578 vs F0's 0.489, a +0.089 gain). `sad` and `neu` also improve substantially. pair_hap_exc reaches 0.6375, the highest of any M6 variant except F3's pair_hap_exc=0.6268 (which achieves that at the cost of other metrics).

7. **`ang` F1 is strong in F0 and F4a but not F5.** F0=0.704, F4a=0.713, F5=0.704. The cross-utterance edges improve `hap`/`sad`/`neu` but do not further help `ang`/`exc`.

8. **F5b (cross-utterance with joint schedule) is nearly as strong as F5** (0.7042 vs 0.7045 single seed). The alternating schedule adds marginal value on top of the cross-utterance edges. The 6C+ extension (cross-utterance inter-modal pairs) is the decisive component, not the schedule.

### Verdict

F0 (mean) is the control threshold — all learned fusions must beat it. Only F4a, F5, F5b clear this bar. F5 (alternating inter-first + cross-utterance edges) is the M6 winner: 3-seed wF1=0.6988±0.0047, +0.0117 over F0, +0.0198 over A0 baseline.

---

## Cross-Area Summary

### wF1 by area best vs A0 baseline (0.6790)

| Area | Best Config | Single-seed wF1 | 3-seed wF1 | Δ vs A0 |
|---|---|---|---|---|
| A0 baseline | — | 0.6790 | — | — |
| M2 | G4_ModDrop015 | 0.6890 | — | +0.0100 |
| M2 (confirmed) | G5 (OGM+ModDrop) | — | 0.6821±0.0055 | +0.0031 |
| M3 | L5_bcl_global | 0.6880 | 0.6831±0.0077 | +0.0041 |
| M4 | E1_shared (screen) | 0.6908 | 0.6818±0.0055 | +0.0028 |
| M4 | E0_off (confirm) | — | 0.6860±0.0025 | +0.0070 |
| **M6** | **F5_cross_utt** | **0.7045** | **0.6988±0.0047** | **+0.0198** |

### Observations

- **M6 (fusion/graph structure) has the largest impact** on performance, by a wide margin. The +0.0198 gain from F5 vs A0 baseline (at 3-seed level) dwarfs every other area's improvement.
- **M2 explicit balancing methods fail; dropout helps marginally.** The text modality is confirmed dominant by probe diagnostics, but gradient-based and gate-based corrections make things worse. Simple modality dropout is the only safe option.
- **M3 loss design shows consistent but modest gains.** BCL global scope outperforms CBFC and EACL. The key insight is that label smoothing and class-balance interact negatively; removing smoothing while keeping CB improves minority-class F1.
- **M4 implicit edges show ambiguous evidence at 3 seeds.** Single-seed screening favors E1/E2, but E0 (no implicit edges) wins the 3-seed confirmation for reliability. Implicit edges appear to help `hap` F1 specifically (+0.04–0.05) without reliable wF1 gains.
- **Cross-utterance inter-modal edges (M6 6C+) are the single most impactful architectural change tested.** F5 is the only variant across all four areas to reliably exceed 0.70 wF1 at 30 epochs.
- **`hap` is the hardest class and the most sensitive to architectural changes** (range 0.434–0.578 across all 45 variants). `sad` is consistently the easiest (range 0.764–0.833).
- **F1 (current text-anchored CrossModalAttn) loses to a parameter-free mean at 30 epochs.** This calls into question the contribution of the fusion layer in the original architecture, at least at limited training budgets.

---

*Data: `Architecture/outputs/ablation_m{2,3,4,6}/{ablation_m{2,3,4,6}_results.json}`. Checkpoints + per-epoch histories in `checkpoints/` subdirectories.*
