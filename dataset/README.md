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

## 4. Limitações de Escala Observadas (Dataset V1)

A análise empírica dos resultados revelou que **13 pares de treinamento representam um gargalo de generalização**:
1. **Sub-representação de Corpo Inteiro:** Com apenas 3 imagens de corpo inteiro, o modelo sofreu deriva de identidade fisionômica ao tentar sintetizar a personagem em planos abertos e dinâmicos.
2. **Homogeneidade de Iluminação Inicial:** As imagens de treino possuíam iluminação predominantemente difusa e controlada de estúdio, o que dificultou a adaptação para luz solar direta ou sombras duras de montanha sem introduzir o efeito de pele cerosa/plástica.
3. **Plano de Expansão (Próxima Iteração):** Curadoria de mais 12 pares focados em variações angulares (perfil e 3/4), poses dinâmicas e texturas de pele não retocadas para restaurar a microderme natural.

---

## 5. Sample Verification Pairs
All 13 training pairs are preserved in [`samples/`](./samples/) and mirrored on [Hugging Face](https://huggingface.co/datasets/gustavogrecov002/sofia-consistency-engine/tree/main/dataset/samples) for full auditability and reproducibility.

### Visual Preview of Dataset Stratification:

| Macro / Tight Face | Studio Headshot | Upper Body / Bust | Full Body / Lifestyle |
| :---: | :---: | :---: | :---: |
| <img src="https://huggingface.co/datasets/gustavogrecov002/sofia-consistency-engine/resolve/main/dataset/samples/sofia_05_tight_face.png" width="180" alt="Tight Face" /> | <img src="https://huggingface.co/datasets/gustavogrecov002/sofia-consistency-engine/resolve/main/dataset/samples/sofia_01_portrait_studio_white.png" width="180" alt="Studio Headshot" /> | <img src="https://huggingface.co/datasets/gustavogrecov002/sofia-consistency-engine/resolve/main/dataset/samples/sofia_08_bust_studio_white.png" width="180" alt="Upper Body Bust" /> | <img src="https://huggingface.co/datasets/gustavogrecov002/sofia-consistency-engine/resolve/main/dataset/samples/sofia_15_full_street_casual.png" width="180" alt="Full Body Street" /> |
| `sofia_05_tight_face.png` | `sofia_01_portrait_studio_white.png` | `sofia_08_bust_studio_white.png` | `sofia_15_full_street_casual.png` |
