# Dataset Curation & Concept Disentanglement Methodology

## 1. The Core Problem in Identity Fine-Tuning
A major vulnerability in LoRA adaptations for generative vision models is **concept entanglement**:
If a character is consistently depicted in the same setting (e.g., photo studio) or wearing identical clothing, the neural network conflates the person's identity with those peripheral tokens. When prompted to generate the subject in a mountain village or casual beachwear, the model either hallucina the studio background or suffers facial distortion.

---

## 2. Concept Disentanglement Strategy
To teach the transformer that **Sofia** corresponds strictly to biometrical facial geometry and skin texture, we engineered a dissociative captioning protocol:

```
[Trigger Token] + [Core Biometrics] + [Dynamic Lighting/Framing] + [Contextual Environment & Wardrobe]
```

### Prompt Separation Example:
- **Trigger Token:** `sofia_woman`
- **Mandatory Facial Anchors:** `28-year-old Scandinavian woman, sharp emerald green eyes, warm smile, natural blonde hair parted in middle, defined cheekbones, fair skin with subtle freckles`
- **Dynamic Context (Dissociated):** `wearing oversized wool sweater, sitting in outdoor alpine cafe, cold morning mist, shallow depth of field, 85mm f/1.4 lens`

By explicitly describing clothing, lens focal length, lighting, and environmental attributes in the training caption, the gradient updates for those visual features are absorbed by generic semantic tokens rather than bound to `sofia_woman`.

---

## 3. Dataset Distribution & Multi-Scale Stratification
The training set comprises 13 pristine, uncompressed $1024 \times 1024$ image-text pairs:

| Framing Category | Count | Primary Objective | Key Samples |
| :--- | :--- | :--- | :--- |
| **Tight Face / Macro** | 2 | Iris specular reflection, microdermal pores, freckles | `sofia_05_tight_face.png` |
| **Studio Headshot** | 3 | Neutral facial geometry, standard lighting calibration | `sofia_01_portrait_studio_white.png` |
| **Upper Body / Bust** | 5 | Natural neck-to-shoulder transitions, diverse garments | `sofia_08_bust_studio_white.png`, `sofia_09_upper_cafe_mustard.png` |
| **Full Body / Lifestyle** | 3 | Anatomical proportions, natural walking poses | `sofia_15_full_street_casual.png`, `sofia_16_full_summer_dress.png` |

---

## 4. Sample Verification Pairs
All 13 training pairs are preserved in [`samples/`](./samples/) for full auditability and reproducibility.
