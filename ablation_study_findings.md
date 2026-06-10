# HyFIN-Net Ablation Study — Factual Findings
**Dataset:** IEMOCAP | **Epochs:** 30 | **Seed:** 42

---

## Baseline (A0_Full_Model)
| Acc | wF1 | mF1 | Best-val epoch |
|---|---|---|---|
| 0.6815 | 0.6851 | 0.6720 | 15 |

---

## Group A — Primary Module Ablations

| Variant | Acc | wF1 | mF1 | ΔAcc | ΔwF1 | ΔmF1 | Best epoch |
|---|---|---|---|---|---|---|---|
| A0 Full Model | 0.6815 | 0.6851 | 0.6720 | — | — | — | 15 |
| A1 w/o IGM | 0.6734 | 0.6746 | 0.6565 | −0.0080 | −0.0105 | −0.0155 | 15 |
| A2 w/o HM | 0.6808 | 0.6837 | 0.6716 | −0.0006 | −0.0014 | −0.0004 | 29 |
| A3 w/o MF | 0.6888 | 0.6849 | 0.6655 | +0.0074 | −0.0001 | −0.0065 | 13 |
| A4 w/o CrossModalAttn | 0.6815 | 0.6848 | 0.6742 | +0.0000 | −0.0003 | +0.0021 | 23 |
| A5 w/o CBFC Loss | 0.6654 | 0.6691 | 0.6572 | −0.0160 | −0.0160 | −0.0148 | 26 |

**A1 (w/o IGM):** wF1 drops 0.0105, mF1 drops 0.0155. `hap` drops −0.101, `sad` drops −0.043. `ang` and `exc` both increase without IGM (+0.046, +0.018 respectively).

**A2 (w/o HM):** wF1 drops 0.0014, mF1 drops 0.0004. All per-class changes under 0.011. `fru` drops 0.011, `sad` and `neu` slightly increase.

**A3 (w/o MF):** wF1 drops only 0.0001 (effectively flat), mF1 drops 0.0065. Accuracy increases +0.0074. `exc` increases +0.054. `hap` drops −0.100.

**A4 (w/o CrossModalAttn):** wF1 drops 0.0003, mF1 *increases* +0.0021. `ang` and `exc` both increase. Accuracy unchanged at 0.6815. Best-val epoch pushed to 23.

**A5 (w/o CBFC Loss):** wF1 drops 0.0160, mF1 drops 0.0148. Best-val epoch is 26 vs 15 baseline. `exc` drops −0.040, `fru` drops −0.029.

---

## Group B — Sub-component Ablations

| Variant | Acc | wF1 | mF1 | ΔAcc | ΔwF1 | ΔmF1 | Best epoch |
|---|---|---|---|---|---|---|---|
| B1 w/o SpeakerEmb | 0.6630 | 0.6651 | 0.6530 | −0.0185 | −0.0200 | −0.0190 | 11 |
| B2 w/o PosEnc | 0.6531 | 0.6567 | 0.6501 | −0.0283 | −0.0283 | −0.0220 | 19 |
| B3 w/o InputLN | 0.6864 | 0.6890 | 0.6803 | +0.0049 | +0.0039 | +0.0082 | 25 |
| B4 IGM w/o ImplicitEdges | 0.6852 | 0.6880 | 0.6796 | +0.0037 | +0.0030 | +0.0076 | 30 |
| B5 IGM SingleBranch | 0.6808 | 0.6836 | 0.6743 | −0.0006 | −0.0015 | +0.0023 | 22 |

**B1 (w/o SpeakerEmb):** wF1 drops 0.0200, mF1 drops 0.0190. `sad` drops −0.071, `neu` drops −0.034. Best-val epoch earliest of all variants (11).

**B2 (w/o PosEnc):** Largest drop in the entire study — wF1 −0.0283, mF1 −0.0220. `fru` drops −0.062, `sad` drops −0.021. `hap` is only class with slight increase (+0.009).

**B3 (w/o InputLN):** wF1 *increases* +0.0039, mF1 *increases* +0.0082 over baseline. `hap` increases +0.047, `exc` increases +0.024. `sad` drops −0.015. Best-val epoch 25.

**B4 (IGM w/o ImplicitEdges):** wF1 *increases* +0.0030, mF1 *increases* +0.0076. `hap` increases +0.062, `neu` increases +0.006. `ang` drops −0.005. Latest best-val epoch of all variants (30).

**B5 (IGM SingleBranch):** wF1 drops 0.0015, mF1 *increases* +0.0023. `exc` increases +0.022, `neu` drops −0.013.

---

## Per-Emotion F1 — Full Table

| Variant | hap | sad | neu | ang | exc | fru |
|---|---|---|---|---|---|---|
| A0 Baseline | 0.5200 | 0.8100 | 0.7015 | 0.6455 | 0.6938 | 0.6613 |
| A1 w/o IGM | 0.4192 | 0.7672 | 0.6767 | **0.6918** | **0.7116** | 0.6727 |
| A2 w/o HM | 0.5251 | 0.8160 | 0.7033 | 0.6431 | 0.6917 | 0.6506 |
| A3 w/o MF | 0.4202 | 0.8008 | 0.6959 | 0.6725 | **0.7480** | 0.6556 |
| A4 w/o CrossAttn | 0.5015 | 0.7915 | 0.6932 | **0.6959** | **0.7130** | 0.6500 |
| A5 w/o CBFC | 0.4960 | 0.8058 | 0.7018 | 0.6528 | 0.6539 | 0.6328 |
| B1 w/o SpeakerEmb | 0.4646 | 0.7393 | 0.6675 | **0.6921** | **0.7145** | 0.6399 |
| B2 w/o PosEnc | 0.5109 | 0.7893 | 0.6810 | 0.6633 | 0.6562 | 0.5997 |
| B3 w/o InputLN | **0.5665** | 0.7950 | 0.7004 | 0.6527 | **0.7178** | 0.6494 |
| B4 IGM w/o Impl | **0.5823** | 0.8048 | 0.7077 | 0.6409 | 0.6877 | 0.6544 |
| B5 IGM 1-Branch | 0.5382 | **0.8160** | 0.6883 | 0.6489 | **0.7161** | 0.6384 |

---

## Ranking by ΔwF1 (most negative = largest drop when removed)

| Rank | Variant | ΔwF1 | ΔmF1 | ΔAcc |
|---|---|---|---|---|
| 1 | B2 w/o PosEnc | −0.0283 | −0.0220 | −0.0283 |
| 2 | B1 w/o SpeakerEmb | −0.0200 | −0.0190 | −0.0185 |
| 3 | A5 w/o CBFC Loss | −0.0160 | −0.0148 | −0.0160 |
| 4 | A1 w/o IGM | −0.0105 | −0.0155 | −0.0080 |
| 5 | B5 IGM SingleBranch | −0.0015 | +0.0023 | −0.0006 |
| 6 | A2 w/o HM | −0.0014 | −0.0004 | −0.0006 |
| 7 | A4 w/o CrossModalAttn | −0.0003 | +0.0021 | +0.0000 |
| 8 | A3 w/o MF | −0.0001 | −0.0065 | +0.0074 |
| 9 | B4 IGM w/o ImplicitEdges | +0.0030 | +0.0076 | +0.0037 |
| 10 | B3 w/o InputLN | +0.0039 | +0.0082 | +0.0049 |

---

## Cross-Variant Observations (raw counts / facts only)

- 4 of 10 ablated variants *increase* mF1 over baseline: A4, B3, B4, B5
- 2 of 10 ablated variants *increase* wF1 over baseline: B3, B4
- `hap` is the most volatile class across variants: range 0.4192 (A1) to 0.5823 (B4), span = 0.163
- `sad` is the most stable class: range 0.7393 (B1) to 0.8160 (A2/B5), span = 0.077
- `ang` and `exc` both increase when IGM is removed (A1), when CrossModalAttn is removed (A4), and when SpeakerEmb is removed (B1)
- All 3 variants that increase overall metrics (B3, B4, A4) have later best-val epochs (23–30) vs baseline (15)
- A5 (no CBFC) has the latest best-val epoch among Group A variants (26 vs 15 baseline)
- B2 (no PosEnc) is the only variant where `fru` F1 drops below 0.60 (0.5997)
- B4 (IGM w/o ImplicitEdges) reaches best-val latest of all variants (epoch 30 = end of training window)
- B1 (no SpeakerEmb) reaches best-val earliest of all variants (epoch 11)
- A3 (no MF) reaches best-val earliest among Group A variants (epoch 13)
- `exc` F1 with A3 (0.7480) is the highest single per-class score across the entire study
- `hap` F1 with B4 (0.5823) is the highest `hap` score across the entire study
