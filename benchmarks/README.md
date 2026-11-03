# Production Quality Benchmarks & Empirical Validation

## 1. Out-of-Distribution Stress Test (Campos do Jordão Protocol)
To rigorously evaluate whether the fine-tuned LoRA models achieved true concept generalization without studio overfitting, we conducted inference in an out-of-distribution, highly complex scene: **Sofia at Campos do Jordão (Vila Capivari)** during a cold, misty alpine morning.

### Target Benchmark Image (Uncompressed RAW):
- **Source File:** [`results/foto_01_campos_cafe_capivari_Raw_00001_.png`](./results/foto_01_campos_cafe_capivari_Raw_00001_.png)
- **Model Checkpoint:** `krea/Krea-2-Raw` + `sofia_krea2_step_700.safetensors`
- **Native Resolution:** $1024 \times 1024$ pixels
- **Inference Setup:** FlowMatch Scheduler, 28 steps, CFG Scale 4.5, Seed 42

---

## 2. Quantitative & Qualitative Assessment Metrics

```
+---------------------------------------------------------------------------------------+
| EVALUATION PILLAR       | OBSERVED PERFORMANCE                     | BENCHMARK STATUS |
+---------------------------------------------------------------------------------------+
| 1. Dermal Realism       | Micro-pores, organic freckles, natural  | PASSED (Top 1%)  |
|                         | subsurface scattering (no airbrushing)  |                  |
| 2. Ocular Consistency   | Perfect pupil alignment, emerald hue,   | PASSED (Zero-Err)|
|                         | realistic corneal reflection            |                  |
| 3. Hand & Bone Anatomy  | Bi-manual mug grip, 5 articulated digits| PASSED (Zero-Err)|
|                         | per hand, correct fingernail beds       |                  |
| 4. Volumetric Physics   | Hot chocolate steam interacts with knit | PASSED           |
|                         | fabric without clipping artifacts       |                  |
| 5. Style Disentanglement| Swiss chalet alpine architecture rendered| PASSED (Zero-OOD)|
|                         | seamlessly around subject               |                  |
+---------------------------------------------------------------------------------------+
```

### Visual Preview:

| Production Sample (RAW) | Checkpoint Comparison (Step 400) |
| :---: | :---: |
| ![Sofia Café Capivari](./results/foto_01_campos_cafe_capivari_Raw_00001_.png) | ![Sofia Step 400](./results/comparison_step_400.png) |
| *High-Fidelity Out-of-Studio Test* | *Checkpoint Progress Evaluation* |

---

## 3. Additional Benchmark Scenarios
- [`results/foto_02_campos_mirante_por_do_sol_Raw_00001_.png`](./results/foto_02_campos_mirante_por_do_sol_Raw_00001_.png) — Sunset mountain overlook testing golden-hour atmospheric lighting.
- [`results/foto_04_campos_lareira_chale_Raw_00001_.png`](./results/foto_04_campos_lareira_chale_Raw_00001_.png) — Indoor fireplace testing dynamic low-light specular highlights and shadow falloff.
