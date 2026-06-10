# Plan: HyFIN-Net Ablation Study Notebook

## Context

HyFIN-Net is a multimodal emotion recognition model for conversations (MERC) trained on IEMOCAP. It has 4 learnable graph modules (IGM, HM, MF, CrossModalAttn) plus auxiliary components (speaker embedding, positional encoding, InputLN, CBFC loss). We need a separate notebook that systematically ablates each module to measure its individual contribution via accuracy and weighted/macro F1.

Baseline (full model): acc=0.6815, wF1=0.6851, mF1=0.6720

---

## Target File

`/mnt/Work/ML/Thesis/WACV/Architecture/ablation_study.ipynb`

---

## Ablation Variants (18 total, 3 groups)

### Group A — Primary Module Ablations (core contribution)
| ID | Name | What changes |
|----|------|-------------|
| A0 | Full Model (baseline) | No change |
| A1 | w/o IGM | Skip InceptionGraphModule; pass zeros [N,256] for p |
| A2 | w/o HM | Skip HyperGraphModule; pass zeros [N,256] for q |
| A3 | w/o MF | Skip MultiFrequencyModule; pass zeros [N,256] for f |
| A4 | w/o CrossModalAttn | Replace CrossModalAttn with `concat(ht,ha,hv)` directly (no learned attention) |
| A5 | w/o CBFC Loss | Set `cfg.cbfc_mu = 0.0`; train with CBCE only |

### Group B — Sub-component Ablations (module internals)
| ID | Name | What changes |
|----|------|-------------|
| B1 | w/o Speaker Embedding | Zero out speaker embeddings in UnimodalEncoder |
| B2 | w/o PositionalEncoding | Remove PE from text encoder path |
| B3 | w/o InputLN | Replace InputLN with nn.Identity |
| B4 | IGM w/o Implicit Edges | Skip ImplicitEdgeDetector; IGM uses only explicit edges |
| B5 | IGM Single Branch | Remove one inception branch (keep only window=(5,3)) |

~~Group C (Modality Ablations) — excluded per user request~~

---

## Implementation Strategy

### Ablation Control via Flags

Add an `AblationCfg` dataclass with boolean flags:
```python
@dataclass
class AblationCfg:
    use_igm: bool = True
    use_hm: bool = True
    use_mf: bool = True
    use_cross_attn: bool = True
    use_speaker_emb: bool = True
    use_pos_enc: bool = True
    use_input_ln: bool = True
    use_implicit_edges: bool = True
    igm_branches: int = 2  # 1 or 2
    cbfc_mu: float = 0.1   # set 0.0 to disable CBFC
    # modality zeroing excluded
```

### Modified Classes (minimal surgical changes)

**AblatedUnimodalEncoder** — wraps UnimodalEncoder:
- `use_speaker_emb=False` → skip `self.spk_emb(speakers)` addition
- `use_pos_enc=False` → skip `self.pe(xt)` in text path
- `use_input_ln=False` → replace `self.ln_t/a/v` with identity

**AblatedHyFINNet** — wraps HyFINNet forward:
- `use_igm=False` → `p = torch.zeros_like(flat)` (no graph processing)
- `use_hm=False` → `q = torch.zeros_like(flat)`
- `use_mf=False` → `f = torch.zeros_like(flat)`
- `use_cross_attn=False` → `z = torch.cat([ht, ha, hv], dim=-1)` (simple concat, skip CrossModalAttn)
- `use_implicit_edges=False` → pass `implicit_edge_index=None` to IGM (use only explicit)
- `igm_branches=1` → only run one IGMBranch
- `zero_text/audio/visual=True` → zero input tensors before UnimodalEncoder

**AblatedCBFCLoss**:
- When `cbfc_mu=0.0`, skip CBFC computation entirely

### Notebook Structure (cells)

```
1. Setup & Imports
2. Paste all classes from arch21.ipynb verbatim (InputLN, PositionalEncoding,
   UnimodalEncoder, IGM classes, HM classes, MF classes, CrossModalAttn,
   Classifier, HyFINNet, loss functions, evaluate(), train())
3. AblationCfg dataclass
4. AblatedUnimodalEncoder (subclass of UnimodalEncoder)
5. AblatedHyFINNet (subclass of HyFINNet, overrides __init__ and forward)
6. build_ablated_model(ablation_cfg) → instantiates AblatedHyFINNet
7. ABLATION_REGISTRY dict — maps variant name → AblationCfg instance
8. run_ablation(name, ablation_cfg, epochs=60, seed=42) → dict with results
9. Main loop: iterate ABLATION_REGISTRY, collect results
10. Results table (pandas DataFrame)
11. Bar charts: accuracy, wF1, mF1 per variant grouped by ablation group
12. Delta table: each variant's drop from full model baseline
13. Per-emotion F1 heatmap comparing all ablations
14. Summary markdown cell with interpretation
```

### Key Implementation Notes

1. **Epochs**: Use `cfg.epochs = 30` for ablation runs (half of full training) unless user wants full 60. Make it a top-level constant `ABLATION_EPOCHS = 30`.

2. **Seed**: Fix `torch.manual_seed(42)` + `numpy.random.seed(42)` per run for reproducibility.

3. **Zero-module approach**: When `use_igm=False`, output zeros rather than removing the module entirely. This keeps the concat dimension `[p|q|f]` at `[N, 768]` always. Then CrossModalAttn sees correct input shape.

4. **Results saving**: Save each ablation result to `outputs/ablation_results.json` incrementally (after each variant completes) so partial results survive crashes.

6. **Reuse `evaluate()`**: Use the exact `evaluate()` function from arch21.ipynb — no modifications needed.

7. **Data loading**: Reuse all data loading code from arch21.ipynb verbatim. Load data once, reuse across all ablations.

---

## Files to Reuse (verbatim copy from arch21.ipynb)

- `_load_pickle()` — arch21.ipynb line 171
- `parse_graphsmile_pickle()` — arch21.ipynb line 178
- `_speaker_to_idx()` — arch21.ipynb line 225
- `MERCDataset` — arch21.ipynb line 240
- `pad_collate()` — arch21.ipynb line 260
- All graph builder functions: `build_igm_graph()`, `build_hyperedge_index()`, `build_mf_edges()`
- All nn.Module classes listed in the map above
- `effective_class_weights()`, `CBCELoss`, `CBFCLoss`
- `evaluate()`, `make_scheduler()`, `train()`

---

## Output Artifacts

All outputs saved to `Architecture/outputs/ablation/` (created if not exists).

```
Architecture/outputs/ablation/
├── ablation_results.json               # Raw metrics per variant (written incrementally)
├── ablation_summary_table.csv          # Pandas CSV with acc, wF1, mF1 per variant
├── ablation_delta_table.csv            # Drop from baseline per variant
├── plots/
│   ├── ablation_accuracy_bar.png       # Bar chart: accuracy per variant
│   ├── ablation_wf1_bar.png            # Bar chart: weighted F1 per variant
│   ├── ablation_mf1_bar.png            # Bar chart: macro F1 per variant
│   ├── ablation_grouped_bar.png        # Combined grouped bar (acc+wF1+mF1)
│   ├── ablation_delta_bar.png          # Delta from baseline (drop = importance)
│   └── ablation_per_emotion_heatmap.png  # Per-class F1 heatmap across variants
└── checkpoints/
    ├── {variant_name}_bestval.pt       # Best-val checkpoint per variant
    └── {variant_name}_history.json     # Per-epoch train/val/test history per variant
```

**Incremental saving**: After each variant completes, immediately:
1. Append result to `ablation_results.json`
2. Save model checkpoint to `checkpoints/`
3. Save per-epoch history to `checkpoints/`

This ensures partial results survive crashes during the long ablation run.

---

## Verification

1. Run Cell 1-7 (setup + class definitions) — no errors
2. Run Group A0 (full model) — should match or closely reproduce baseline (acc≈0.68, wF1≈0.685)
3. Run one ablation (A1 w/o IGM) — training completes, metrics printed
4. Run full loop — all 18 variants finish, results JSON populated
5. Plot cells render 4 figures without error
6. Delta table shows which modules hurt most when removed (largest F1 drop = most important module)
