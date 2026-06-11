# GraphMERC v2 — Update Notes (arch21 → graphmerc_v2.ipynb)

*Companion to `graphmerc_v2.ipynb` and `build_v2_notebook.py`. Every change is traced to a measured result from the two ablation campaigns: Study 1 (component removal, A/B/C grids, 15 variants) and Study 2 (M2/M3/M4/M6 option grids, 45 variants). IEMOCAP, 30 epochs, protocol-matched baseline A0 = 0.6790 wF1 unless noted.*

---

## 1. How the update was produced

`build_v2_notebook.py` generates the new notebook from `arch21.ipynb`. It **reuses verbatim** the cells both campaigns ran on — environment/PyG install, imports, the 3-layout GraphSmile pickle parser, `MERCDataset`/`pad_collate`, and the SEED=2024 train/val/test split (so every v2 number is directly comparable to every ablation number) — and **replaces** the model, loss, and training cells. Regenerate any time with:

```bash
python build_v2_notebook.py arch21.ipynb graphmerc_v2.ipynb
```

All 29 cells were validated to compile; the v2-specific logic passed functional smoke tests (edge-count correctness of `build_graph_v2` against hand-derived counts, no edges crossing dialogue/modality boundaries where prohibited, `BCLLoss` finite forward/backward incl. degenerate batches, modality dropout zeroing exactly one stream per dialogue).

---

## 2. Removed (with the measured reason)

| Component (arch21 cell) | Verdict | Evidence |
|---|---|---|
| `InputLN` (masked per-dialogue LayerNorm) | **Removed** | Study 1 B3: **+1.00 wF1, +1.34 mF1, hap +0.054** when removed. The pickle features are already utterance-normalized; per-dialogue whitening destroyed cross-dialogue intensity information. Note: this verdict is feature-conditional — re-test if the feature pipeline changes. |
| `HyperGraphModule` + `HypergraphConvM3` | **Removed** | A2: +0.47 when removed. (Its learnable weight tables were batch-position-indexed — no stable identity under shuffling — so the module was a pairwise-redundant conv in practice.) |
| `MultiFrequencyModule` + `HighFreqConv` | **Removed** | A3: +0.59 wF1 when removed. Its hap-supporting role (C4 showed −2.43 when kept in the pruned context without gating) is superseded: F5 reaches hap 0.578, the study's best. |
| Second IGM branch | **Removed** (single branch) | B5: +0.45 when removed; the (5,3)/(3,2) windows overlapped too much to differentiate. |
| `ImplicitEdgeDetector` + implicit edges | **Removed** | Study 1 B4: +0.90 when removed (ungradiented version). Study 2 M4 (gradiented, all four designs): E0-off 3-seed **0.6860±0.0025** beats best detector 0.6818±0.0055 — the screening win (+0.77) reversed at 3 seeds with double the variance. |
| `CrossModalAttn` (text-anchored V→T, A→T) | **Removed** → mean fusion | M6: worst variant in the grid, **−2.82 vs the parameter-free mean**; pairwise (−1.35) and gated (−1.52) also lost. 60-epoch re-test queued (R3) before final lock. |
| `label_smoothing=0.1` | **→ 0.0** | M3 grid: smoothing drops hap F1 by 0.038–0.052 under both CB conditions — it redistributes target mass uniformly, diluting minority up-weighting. |
| `CBFCLoss(γ=2, within-dialogue)` as the default | **→ BCL global** | M3: BCL-global +0.51 vs base, best hap (0.535) and pair metrics; CBFC-γ2 −0.25; **dialogue-scoped BCL < no-contrastive** (scope, not the loss family, was the flaw). CBFC(γ=1) kept behind `cfg.contrastive='cbfc'` as the lower-variance fallback (γ=1 beat γ=0 and γ=2 in the sweep). |
| `DualCL` placeholder (λ=0) | **Deleted** | Dead code. |
| OGM-GE / AFW / AMW / EACL | **Never added** | Measured and rejected: OGM-GE −0.97…−1.32 (both α), AFW −0.25, AFW+AMW −1.96, EACL −2.64 (anchor telemetry: hap/exc anchors never separated). M2's probe diagnostic confirmed text dominance, but with frozen pickle features the corrections had nothing to redistribute toward — re-open after the Stage-2 feature swap (R5). |

## 3. Added (with the measured reason)

| Addition | Evidence |
|---|---|
| **Cross-utterance inter-modal edges (E3)** — each node receives from the *other two modalities'* past-p/future-f window positions, not only the same utterance | M6 F5: **0.6988±0.0047 (3-seed), +1.98 vs A0** — the largest reliable gain in either campaign; F5 single-seed 0.7045 was the study's best number, exceeding the 60-epoch arch21 (0.6851) at half the budget. Independently corroborates GraphSmile's (TPAMI 2025) inter-utterance heterogeneous-cue claim in a different backbone. |
| **Alternating inter-FIRST schedule** — layer 0 propagates inter-modal edges (E2+E3), layer 1 intra-modal (E1) | F4a +0.92 vs the mean-fusion base; the reverse order (intra-first) *loses* −0.72. `'joint'` flag retained: F5b (0.7042) ≈ F5, confirming the edges, not the schedule, are the driver — joint is the simpler fallback. |
| **Modality dropout p=0.15** — train-time, one stream zeroed per selected dialogue | M2 G4: +0.49 wF1 / +0.58 mF1, the only balancing method that helped; consistent with missing-modality robustness training. |
| **Per-class + hard-pair telemetry** (`f1_hap…f1_fru`, `pair_hap_exc`, `pair_ang_fru`, train−val `gap`) in every history row | Required by the decision rules both campaigns used; hap volatility (0.42–0.58 across 60 variants) makes per-class logging non-optional. |
| `set_seed()` + `run_name` + **R1 verification registry** | The three Study-2 winners were measured on *different bases* — composition is unverified. R1 runs: `V2_core`×3 seeds, `V2_noModDrop`, `V2_noBCL`, `V2_cbfc_g1`, `V2_joint`, `V2_core_60ep`. |

## 4. Kept (essential per Study 1 — do not re-ablate)

| Component | Evidence |
|---|---|
| Positional encoding | −2.23 wF1 when removed — largest drop in the entire project |
| Speaker embedding | −1.39; also the basis for the next contribution (speaker structure, Phase R2) |
| A contrastive term | −1.00 when removed entirely |
| Windowed graph propagation (TransformerConv, angular edge prior, β-gating) | −0.45 wF1 / −1.04 mF1 when removed; hap −0.094 |
| Training protocol (AdamW, warmup+cosine, grad clip 1.0, dropout 0.5, best-val checkpointing; best-test logged as upper bound only) | unchanged from arch21 |

One deliberate cleanup: arch21's intra-modal window loop appended each (i, j) pair **twice** (once from the past loop, once from the future loop, both bidirectional). v2 builds directed in-edges once per (shift, direction) — same receptive field, no duplicate edges. This is a correctness cleanup, not an ablated change; it slightly alters effective edge weighting vs F5's measured config, which R1 will absorb.

## 5. Expected numbers and gates

| Quantity | Value |
|---|---|
| Reference: F5 alone (3-seed, 30 ep) | 0.6988 ± 0.0047 |
| v2 core gate (R1) | ≥ 0.695 3-seed mean; each component's removal costs ≥ 0.3 |
| Plausible v2 landing zone | 0.70–0.71 wF1 IEMOCAP (30 ep, pickle features) if ModDrop/BCL compose even partially |
| Param count | ~1.5–2M (down from arch21's ~4–5M) — roughly half the compute, no O(n²) modules |

## 6. What this version does NOT yet contain (deliberately)

1. **Speaker structure beyond the embedding** (same-speaker edges, speaker meta-node) — the untested novelty axis; Phase R2 adds it *on top of a verified v2 core* so its gain is attributable.
2. **MELD-specific pieces** (SDP shift head from the pickle's sentiment labels; window/imbalance re-checks) — Phase R4; the v2 verdicts are IEMOCAP-measured, and E3's gain may shrink on short multi-party dialogues.
3. **Learned fusion at 60 epochs** — R3's two runs settle the under-training hypothesis before the final lock.
4. **The Stage-2 feature swap** — R5; also the trigger to re-open the parked M2 balancing methods, whose failure is plausibly feature-conditional.

## 7. File manifest

| File | Purpose |
|---|---|
| `graphmerc_v2.ipynb` | The updated notebook (29 cells, compile-validated, smoke-tested logic) |
| `build_v2_notebook.py` | Generator script — re-run against arch21.ipynb to regenerate/modify |
| This document | Change log with per-change evidence |
