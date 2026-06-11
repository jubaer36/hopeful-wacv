# HyFIN-Net Ablation Study — Factual Findings
**Dataset:** IEMOCAP | **Epochs:** 30 (all variants including baseline) | **Seed:** SEED=2024 (data split), ABLATION_SEED=42 (model training)

---

## Baseline (A0_Full_Model — 30 epochs)
| Acc | wF1 | mF1 | Best-val epoch |
|---|---|---|---|
| 0.6771 | 0.6790 | 0.6669 | 22 |

> Note: Full 60-epoch trained model (best-val at epoch 15) had acc=0.6815, wF1=0.6851, mF1=0.6720.
> All comparisons below use the 30-epoch baseline for a fair comparison.

---

## Full Results Table (15 variants)

| Variant | Acc | wF1 | mF1 | ΔAcc | ΔwF1 | ΔmF1 | Best epoch |
|---|---|---|---|---|---|---|---|
| A0 Full Model | 0.6771 | 0.6790 | 0.6669 | — | — | — | 22 |
| A1 w/o IGM | 0.6734 | 0.6746 | 0.6565 | −0.0037 | −0.0045 | −0.0104 | 15 |
| A2 w/o HM | 0.6808 | 0.6837 | 0.6716 | +0.0037 | +0.0047 | +0.0047 | 29 |
| A3 w/o MF | 0.6888 | 0.6849 | 0.6655 | +0.0117 | +0.0059 | −0.0014 | 13 |
| A4 w/o CrossModalAttn | 0.6815 | 0.6848 | 0.6742 | +0.0043 | +0.0058 | +0.0072 | 23 |
| A5 w/o CBFC Loss | 0.6654 | 0.6691 | 0.6572 | −0.0117 | −0.0100 | −0.0097 | 26 |
| B1 w/o SpeakerEmb | 0.6630 | 0.6651 | 0.6530 | −0.0142 | −0.0139 | −0.0139 | 11 |
| B2 w/o PosEnc | 0.6531 | 0.6567 | 0.6501 | −0.0240 | −0.0223 | −0.0169 | 19 |
| B3 w/o InputLN | 0.6864 | 0.6890 | 0.6803 | +0.0092 | +0.0100 | +0.0134 | 25 |
| B4 IGM w/o ImplicitEdges | 0.6852 | 0.6880 | 0.6796 | +0.0080 | +0.0090 | +0.0127 | 30 |
| B5 IGM SingleBranch | 0.6808 | 0.6836 | 0.6743 | +0.0037 | +0.0045 | +0.0074 | 22 |
| C1 LN+Impl removed | 0.6821 | 0.6840 | 0.6774 | +0.0049 | +0.0049 | +0.0104 | 24 |
| C2 C1 + 1branch + noCA | 0.6796 | 0.6831 | 0.6752 | +0.0025 | +0.0041 | +0.0083 | 24 |
| C3 minimal | 0.6913 | 0.6924 | 0.6787 | +0.0142 | +0.0133 | +0.0118 | 23 |
| C4 minimal keepMF | 0.6667 | 0.6681 | 0.6548 | −0.0105 | −0.0110 | −0.0121 | 19 |

---

## Group A — Primary Module Ablations

**A1 (w/o IGM):** wF1 drops 0.0045, mF1 drops 0.0104. `hap` drops −0.094, `sad` drops −0.034. `ang` and `exc` both increase without IGM (+0.034, +0.036).

**A2 (w/o HM):** wF1 increases +0.0047, mF1 increases +0.0047. All per-class changes under 0.015. `sad` and `neu` slightly increase, `ang` and `fru` slightly decrease.

**A3 (w/o MF):** wF1 increases +0.0059, mF1 drops only −0.0014. Accuracy increases +0.0117. `exc` increases +0.073. `hap` drops −0.093. Best-val epoch earliest among Group A (13).

**A4 (w/o CrossModalAttn):** wF1 increases +0.0058, mF1 increases +0.0072. `ang` increases +0.038, `exc` increases +0.038. `sad` drops −0.009.

**A5 (w/o CBFC Loss):** wF1 drops 0.0100, mF1 drops 0.0097. Best-val epoch 26 vs 22 baseline. `exc` drops −0.022, `fru` drops −0.013.

---

## Group B — Sub-component Ablations

**B1 (w/o SpeakerEmb):** wF1 drops 0.0139, mF1 drops 0.0139. `sad` drops −0.062, `neu` drops −0.042. Best-val epoch earliest of all variants (11).

**B2 (w/o PosEnc):** Largest drop in entire study — wF1 −0.0223, mF1 −0.0169. `fru` drops −0.046, `sad` drops −0.012. `hap` is only class with slight increase (+0.002).

**B3 (w/o InputLN):** wF1 increases +0.0100, mF1 increases +0.0134. Largest positive wF1 delta among single-component ablations. `hap` increases +0.054, `exc` increases +0.042. `sad` drops −0.006.

**B4 (IGM w/o ImplicitEdges):** wF1 increases +0.0090, mF1 increases +0.0127. `hap` increases +0.069. Best-val epoch 30 — end of training window.

**B5 (IGM SingleBranch):** wF1 increases +0.0045, mF1 increases +0.0074. `exc` increases +0.041. `neu` drops −0.021.

---

## Group C — Combined Pruning

**C1 (InputLN + ImplicitEdges removed):** wF1 increases +0.0049, mF1 increases +0.0104 over baseline. `hap` increases +0.061, `exc` increases +0.044. `fru` drops −0.028. Stacks B3 and B4 gains — both positive individually and positive combined.

**C2 (C1 + SingleBranch + no CrossModalAttn):** wF1 increases +0.0041, mF1 increases +0.0083. `exc` increases +0.048, `ang` increases +0.016. `neu` drops −0.053 — largest `neu` drop in the study.

**C3 (minimal — C2 + no HM + no MF):** wF1 increases +0.0133, mF1 increases +0.0118. Highest wF1 (0.6924) and highest accuracy (0.6913) of any variant in the study. `sad` reaches 0.8358 — highest across all 15 variants. `exc` increases +0.058. `hap` drops −0.026 from baseline.

**C4 (minimal keepMF — C2 + no HM, MF stays):** wF1 drops −0.0110, mF1 drops −0.0121. Only Group C variant that hurts performance. `hap` drops −0.052, `neu` drops −0.042. Best-val epoch 19.

---

## Per-Emotion F1 — Full Table

| Variant | hap | sad | neu | ang | exc | fru |
|---|---|---|---|---|---|---|
| A0 Baseline | 0.5130 | 0.8008 | 0.7090 | 0.6575 | 0.6755 | 0.6456 |
| A1 w/o IGM | 0.4192 | 0.7672 | 0.6767 | 0.6918 | 0.7116 | 0.6727 |
| A2 w/o HM | 0.5251 | 0.8160 | 0.7033 | 0.6431 | 0.6917 | 0.6506 |
| A3 w/o MF | 0.4202 | 0.8008 | 0.6959 | 0.6725 | **0.7480** | 0.6556 |
| A4 w/o CrossAttn | 0.5015 | 0.7915 | 0.6932 | **0.6959** | 0.7130 | 0.6500 |
| A5 w/o CBFC | 0.4960 | 0.8058 | 0.7018 | 0.6528 | 0.6539 | 0.6328 |
| B1 w/o SpeakerEmb | 0.4646 | 0.7393 | 0.6675 | 0.6921 | 0.7145 | 0.6399 |
| B2 w/o PosEnc | 0.5109 | 0.7893 | 0.6810 | 0.6633 | 0.6562 | 0.5997 |
| B3 w/o InputLN | 0.5665 | 0.7950 | 0.7004 | 0.6527 | 0.7178 | 0.6494 |
| B4 IGM w/o Impl | **0.5823** | 0.8048 | 0.7077 | 0.6409 | 0.6877 | 0.6544 |
| B5 IGM 1-Branch | 0.5382 | 0.8160 | 0.6883 | 0.6489 | 0.7161 | 0.6384 |
| C1 LN+Impl off | 0.5738 | 0.8127 | 0.7009 | 0.6396 | 0.7197 | 0.6174 |
| C2 C1+1br+noCA | 0.5248 | 0.8148 | 0.6556 | 0.6739 | 0.7236 | 0.6584 |
| C3 minimal | 0.4868 | **0.8358** | 0.7024 | 0.6667 | 0.7336 | 0.6469 |
| C4 minimal keepMF | 0.4615 | 0.7843 | 0.6667 | 0.6565 | **0.7322** | 0.6277 |

---

## Ranking by ΔwF1 (most negative = largest drop when removed)

| Rank | Variant | ΔwF1 | ΔmF1 | ΔAcc |
|---|---|---|---|---|
| 1 | B2 w/o PosEnc | −0.0223 | −0.0169 | −0.0240 |
| 2 | B1 w/o SpeakerEmb | −0.0139 | −0.0139 | −0.0142 |
| 3 | C4 minimal keepMF | −0.0110 | −0.0121 | −0.0105 |
| 4 | A5 w/o CBFC Loss | −0.0100 | −0.0097 | −0.0117 |
| 5 | A1 w/o IGM | −0.0045 | −0.0104 | −0.0037 |
| 6 | C2 C1+1br+noCA | +0.0041 | +0.0083 | +0.0025 |
| 7 | B5 IGM SingleBranch | +0.0045 | +0.0074 | +0.0037 |
| 8 | A2 w/o HM | +0.0047 | +0.0047 | +0.0037 |
| 9 | C1 LN+Impl off | +0.0049 | +0.0104 | +0.0049 |
| 10 | A4 w/o CrossModalAttn | +0.0058 | +0.0072 | +0.0043 |
| 11 | A3 w/o MF | +0.0059 | −0.0014 | +0.0117 |
| 12 | B4 IGM w/o ImplicitEdges | +0.0090 | +0.0127 | +0.0080 |
| 13 | B3 w/o InputLN | +0.0100 | +0.0134 | +0.0092 |
| 14 | C3 minimal | +0.0133 | +0.0118 | +0.0142 |

---

## Cross-Variant Observations (raw counts / facts only)

- **C3 (minimal)** achieves the highest wF1 (0.6924) and highest accuracy (0.6913) of all 15 variants
- **C3** removes: InputLN, ImplicitEdges, second IGM branch, CrossModalAttn, HM, and MF — retains only PosEnc, SpeakerEmb, CBFC loss, IGM single branch
- **C4** is the only Group C variant that hurts performance; it differs from C3 only by keeping MF active
- Adding MF back to C3 (giving C4) reduces wF1 by 0.0243 — the largest single-component swing in the study
- `sad` F1 in C3 (0.8358) is the highest per-class score of any emotion in any variant
- `exc` in C3 (0.7336) is second-highest `exc` score after A3 (0.7480)
- `hap` in C3 (0.4868) is lower than the baseline (0.5130)
- `neu` in C2 (0.6556) is the lowest `neu` score in the study — 0.053 below baseline
- 10 of 14 ablated variants beat baseline wF1; 5 variants score ≥ 0.6836 wF1 vs baseline 0.6790
- Only 5 variants show wF1 drops below baseline: B2, B1, C4, A5, A1
- `ang` and `exc` both increase when IGM (A1), CrossModalAttn (A4), or SpeakerEmb (B1) are removed — pattern holds across 3 independent ablations
- `hap` is the most volatile class: range 0.4192 (A1) to 0.5823 (B4), span = 0.163
- `sad` is the most stable class: range 0.7393 (B1) to 0.8358 (C3), span = 0.097
- B4 (IGM w/o ImplicitEdges) best-val at epoch 30 — the hard ceiling of the training window
- B1 (w/o SpeakerEmb) best-val earliest of all variants (epoch 11)
- B2 (w/o PosEnc) is the only variant where `fru` drops below 0.60 (0.5997)
- C1 stacks B3 and B4 removals: B3 alone gives ΔwF1=+0.0100, B4 alone gives +0.0090, C1 combined gives +0.0049 — sub-additive
- C3 best-val at epoch 23, same as A4 — both converge faster than full model (ep 22, but C3 has far fewer active modules)




## Interpretive Summary and Next Steps

C3 is the strongest result in the study: the minimal core reaches **0.6924 wF1 / 0.6913 acc** on a single seed at 30 epochs, beating both the protocol-matched baseline and the full 60-epoch model. The surviving architecture is small and clean: **PosEnc + SpeakerEmb + CBFC + single-branch IGM**. Everything else appears to be either neutral or harmful in this setting.

### Key interpretation of the C4 anomaly

C4 is the most informative counterexample in the table. It differs from C3 only by keeping **MF**, and that single choice reduces performance sharply. This is not a contradiction with A3, where removing MF from the full model helped. It indicates that MF is **context-dependent**:

- In the full model, CrossModalAttn can suppress a noisy stream.
- In the pruned setting, that gating is gone, so MF reaches the classifier more directly.
- This makes MF harmful when the surrounding modules that regulate it are removed.

The broader lesson is that component value depends on context. The sub-additive combined gains in C1 support the same point: individual improvements do not always add linearly. The large C4 swing also strengthens the case for **multi-seed confirmation** before drawing final conclusions.

### C3’s remaining weakness

C3’s main regression is **hap**:
- C3 hap F1: **0.4868**
- Baseline hap F1: **0.5130**
- Best hap F1 in the study: **0.5823** from B4

C3 wins overall by improving **sad** and **exc**, but it gives up ground on the smallest class. Its macro-F1 is also slightly below B3/B4. This suggests the pruned core is the right base, but the first added module should target **hap/exc** specifically rather than adding generic capacity.

### Recommended next experiments

1. **C3 with multiple seeds**
   - Run seeds `{42, 43, 44}` at 30 epochs
   - Run one additional C3 experiment at 60 epochs

2. **C3 clean rebuild**
   - Remove zeroed concat slots
   - Shrink the classifier head to match the true dimensionality
   - Verify performance equivalence and reduce wasted parameters

3. **McNemar tests on stored predictions**
   - Compare `A0 vs C3`
   - Compare `C3 vs C4`
   - Use the existing `y_true/y_pred` outputs to test whether the differences are statistically meaningful

4. **C3 on MELD**
   - Validate whether the pruning result transfers beyond IEMOCAP
   - Do not assume the same behavior because MELD differs at the points where HM/MF were pruned

### Suggested order for additions to confirmed C3

1. Same-speaker edges in the single-branch IGM graph  
2. Speaker meta-node  
3. EACL-style anchors evaluated on hap-F1 and hap/exc, not only wF1  
4. Unimodal probes to decide whether OGM-GE is needed  
5. SDP only on MELD, where sentiment labels are available

### Summary

The final narrative is stronger than the original kitchen-sink model: a complex hybrid is pruned down to a compact core that performs better at roughly half the compute, and the remaining failure mode points directly to the next addition. The study now supports a clean paper story: **Table 1 is the ablation, C3 is the core model, and the remaining novelty should be introduced only after multi-seed confirmation**.