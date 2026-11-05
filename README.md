# Multimodal Consistency Engine: Production-Grade Fine-Tuning & MLOps Pipeline

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset%20%26%20Artifacts-yellow.svg)](https://huggingface.co/datasets/gustavogrecov002/sofia-consistency-engine)
[![Hugging Face PEFT](https://img.shields.io/badge/HuggingFace-PEFT%20%7C%20Diffusers-yellow.svg?logo=huggingface)](https://huggingface.co/)
[![Hardware](https://img.shields.io/badge/GPU-1x%20NVIDIA%20A40%20(48GB)-76B900.svg?logo=nvidia)](https://www.nvidia.com/)
[![Cloud & FinOps](https://img.shields.io/badge/FinOps-Autonomous%20Sentinel%20Watchdog-00ADD8.svg)](./mlops/sentinel_watchdog.py)
[![Career Track](https://img.shields.io/badge/Track-AI%20Engineering%20%7C%20Generative%20AI-0052CC.svg)](https://github.com/gustavogrecov-tech)

**Engineering, fine-tuning, and operationalizing next-generation generative foundation models under strict cloud budget and memory constraints.**

[Overview](#-overview--executive-summary) •
[Architecture](#-system-architecture--engineering-pipeline) •
[Fine-Tuning Methodology](#-fine-tuning-methodology--mathematical-rigor) •
[FinOps & Autonomous MLOps](#-finops--autonomous-sentinel-watchdog) •
[Validation Benchmarks](#-empirical-validation--production-benchmarks) •
[Reproduction](#-inference--reproduction-guide)

</div>

---

## 🎯 Author & Project Purpose

> **Author:** Gustavo Sales Grecov  
> **Affiliation:** Universidade Federal do ABC (UFABC)  
> **Core Focus:** Engenharia de Inteligência Artificial, Fine-Tuning de Modelos de Fundação Multimodais (PEFT/LoRA), Otimização de Inferência (Zero-OOM) e Governança Orçamentária Autônoma em Nuvem (FinOps).

---

## 📌 Overview & Executive Summary

Este projeto apresenta a engenharia completa de **Fine-Tuning e Operacionalização de Modelos de Fundação Multimodais** (**Krea 2.1 Ultra / Raw** e **Qwen-Image 2.1**), resolvendo dois dos maiores desafios da IA Generativa corporativa moderna:

1. **Colapso de Identidade e Overfitting Visual:** Treinamento de adaptadores LoRA de alta fidelidade para consistência biométrica contínua (personagem *Sofia*) através de **Desentrelaçamento de Conceitos (Concept Disentanglement)** em Flow Matching Transformers.
2. **Engenharia de Memória e Eficiência de Custos (FinOps):** Treinamento de arquiteturas gigantescas (12.8B a 33.2B de parâmetros totais) em GPU única comercial (NVIDIA A40 48GB) dentro de um **teto orçamentário estrito de \$2.00**, orquestrado por um **Agente Sentinela Watchdog em Python** que realizou monitoramento em tempo real, download assíncrono instantâneo de checkpoints (Hot SCP Sync) e autodestruição da VM via API ao atingir \$1.70.

```
+-----------------------------------------------------------------------------------------+
|                                    PROJECT SNAPSHOT                                     |
+--------------------------+--------------------------------------------------------------+
| Base Foundation Models   | Krea 2.1 Raw (12.8B Transformer) & Qwen-Image 2.1            |
| Fine-Tuning Method       | PEFT LoRA (Rank 32, Alpha 32) + Continuous Flow Matching    |
| Memory Optimizations     | TorchAO uint3 Quantization + FP8 Text Encoder + AdamW 8-bit  |
| Cloud Compute Used       | 1x NVIDIA A40 (48GB VRAM) @ RunPod EU-SE-1 ($0.59/h)        |
| Total Execution Budget   | Ceiling: $2.00 | Effective Spend: $1.6996 (15.02% surplus)  |
| Training Loss Evolution  | Initial: 0.8420 ➔ Final: 0.0185 (Krea) / 0.0231 (Qwen)      |
| Production Validation    | High-resolution RAW rendering in complex alpine environment  |
| Hugging Face Artifacts   | https://huggingface.co/datasets/gustavogrecov002/sofia-consistency-engine |
+--------------------------+--------------------------------------------------------------+
```

---

## 🏗️ System Architecture & Engineering Pipeline

O projeto foi construído seguindo as melhores práticas de **Engenharia de Software para IA**, separando claramente as camadas de ingestão, treinamento distribuído, controle autônomo e inferência em produção:

```mermaid
flowchart LR
    subgraph INGESTION["1. Data & Pre-Processing"]
        A[13 Multi-Scale Images 1024x1024] --> B[Dissociative Captions]
        B --> C[Offline Latent & Embedding Cache]
    end

    subgraph CLOUD["2. Fine-Tuning & FinOps (RunPod)"]
        C --> D[Krea2 Transformer 12.8B + Gradient Checkpoint]
        D --> E[AdamW 8-bit + Flow Matching]
        F[Sentinel Watchdog Agent] -.->|Audit VRAM & Spent $0.59/h| D
        F -.->|Hot SCP Sync Checkpoints| G[(Local SafeTensors Vault)]
        F -.->|Auto Kill-Switch @ $1.70| H[Terminate Cloud Pod]
    end

    subgraph INFERENCE["3. Zero-OOM Serving"]
        G --> I[Dynamic Text Encoder Offloading]
        I --> J[Krea2 Pipeline bfloat16]
        J --> K[Empirical Validation: Campos do Jordão RAW]
    end
```

---

## 🔬 Fine-Tuning Methodology & Mathematical Rigor

### 1. Continuous Flow Matching Objective
Ao contrário de modelos convencionais de difusão discretos (DDPM/DDIM) que sofrem com trajetórias curvas e instabilidade de gradiente, o pipeline adota **Flow Matching**. O modelo aprende o campo vetorial que move as distribuições por meio de trajetórias retas de transporte ótimo:

$$\mathcal{L}_{\text{FlowMatch}}(\theta) = \mathbb{E}_{t \in [0, 1], \, x_0 \sim p_{\text{data}}, \, \epsilon \sim \mathcal{N}(0, I)} \left[ \| \mathbf{v}_\theta(x_t, t, c) - (\epsilon - x_0) \|^2 \right]$$

Onde o estado intermediário no passo $t$ é parametrizado linearmente por $x_t = (1 - t) x_0 + t \epsilon$.

### 2. Desentrelaçamento de Conceitos (Concept Disentanglement)
Para evitar que a rede acoplasse a geometria facial da Sofia com roupas ou fundos específicos:
- **Gatilho de Identidade:** `sofia_woman`
- **Âncora Fisionômica Obrigatória:** *Scandinavian woman, sharp emerald green eyes, warm smile, natural blonde hair parted in middle, defined cheekbones, fair skin with subtle freckles*
- **Contexto Circunstancial Dissociado:** As roupas (*cashmere blazer*, *wool sweater*), lentes (*85mm f/1.4*) e iluminação (*softbox*, *mist*) foram descritas minuciosamente em cada arquivo `.txt`, permitindo total liberdade de generalização para novos ambientes de negócio.

### 3. Técnicas Avançadas de Otimização de Memória
- **Quantização uint3 TorchAO:** Pesos quantizados em 3 bits com adaptador de recuperação de acurácia (`accuracy_recovery_adapters/qwen_image_torchao_uint3.safetensors`).
- **Codificador de Texto em FP8 (`qfloat8`):** Redução drástica da sobrecarga de texto de ~10 GB para ~2.5 GB.
- **Otimizador `adamw8bit` (BitsAndBytes):** Estado do otimizador reduzido em 75%, estabilizando o uso de VRAM em **24.19 GB / 44.42 GB** durante o treino de um modelo de 12.8B.
- **Pré-Cache Offline de Latentes e Embeddings:** A codificação de VAE e Text Encoder foi realizada antes do loop de gradientes, liberando a GPU para focar 100% no Transformer.

---

## 🛡️ FinOps & Autonomous Sentinel Watchdog

Um diferencial central deste projeto de Engenharia de IA é a abordagem de **FinOps Produtivo**: infraestrutura sob controle estrito de custos, evitando desperdício de capital em nuvem.

```
                              SENTINEL WATCHDOG LIFECYCLE
                               
 [ Cloud Pod Running ] ─────────► [ Every 30s: Audit Uptime, VRAM, Cost ]
          │                                      │
          │ Checkpoint Generated                 ▼
          └─────────────────────────────► [ Hot SCP Download to Local Vault ]
                                                 │
                                                 ▼ Cost reaches $1.70?
                                          [ YES ] ➔ Execute Emergency Sync
                                                 ➔ Call runpod.terminate_pod()
                                                 ➔ Verify Zero Running Pods
```

### Resultados da Auditoria do Sentinela ([`SENTINEL_SUMMARY.json`](./mlops/SENTINEL_SUMMARY.json)):
* **Teto Estipulado:** \$2.00
* **Gasto Efetivo Consumido:** **\$1.6996 (~\$1.70)**
* **Margem Preservada:** **+\$0.3004 (15.02% de sobra)**
* **Taxa de Preservação de Checkpoints:** **100%** (todos os checkpoints de 100 a 700 steps resgatados via SCP antes do encerramento da instância).

---

## 📊 Empirical Validation & Production Benchmarks (Diagnóstico Técnico Real)

A validação empírica deste projeto adota uma **postura rigorosa de engenharia**: em vez de mascarar imperfeições ou selecionar apenas "gerações de sorte", documentamos de forma transparente os resultados atuais (V1), os avanços alcançados e, principalmente, os **gargalos técnicos e inconsistências reais** que ainda estão em fase de mitigação ativa.

### Amostras de Produção Renderizadas:

<div align="center">

| Amostra de Validação RAW (Produção - Step 700) | Comparativo de Checkpoint (Step 400) |
| :---: | :---: |
| <img src="https://huggingface.co/datasets/gustavogrecov002/sofia-consistency-engine/resolve/main/benchmarks/results/foto_01_campos_cafe_capivari_Raw_00001_.png" width="400" alt="Sofia em Campos do Jordão RAW" /> | <img src="https://huggingface.co/datasets/gustavogrecov002/sofia-consistency-engine/resolve/main/benchmarks/results/comparison_step_400.png" width="400" alt="Sofia Step 400" /> |
| *Teste Out-of-Distribution: Cenário alpino, vestuário de inverno e iluminação fria* | *Convergência intermediária de treino (Step 400)* |

</div>

### 🔍 Diagnóstico Crítico de Falhas e Limitações Observadas

Uma análise técnica honesta das amostras revela os seguintes problemas no estado atual do modelo:

1. **Aspecto Cerosa / Pele Plástica (Waxy Skin Artifact):**
   - **Diagnóstico:** Apesar da redução drástica da função de perda ($0.8420 \rightarrow 0.0185$), a textura da derme facial nas imagens geradas ainda apresenta um acabamento semibrilhante e alisado em excesso ("efeito cera/boneca de plástico"), com deficiência na definição de microporos e textura dérmica genuína.
   - **Causa Raiz Identificada:** O `lora_alpha=32` com taxa de aprendizado constante nas projeções de atenção (`to_q`, `to_k`, `to_v`, `to_out.0`) induziu um leve sobreajuste (overfitting local) nos passos finais (500–700), além da saturação gerada pelo valor elevado de Guidance Scale (CFG 4.5).

2. **Inconsistência Fisionômica e Drift Facial (Identity Drift):**
   - **Diagnóstico:** A personagem **Sofia ainda não atingiu a consistência biométrica plena** esperada entre diferentes planos e enquadramentos. Em fotos de rosto frontal (headshots), os traços se aproximam do dataset; contudo, em planos abertos ou ângulos dinâmicos (como na amostra de Campos do Jordão), ocorre um desvio perceptível no contorno mandibular, espessura dos lábios e formato dos olhos em relação aos pares originais de referência.
   - **Causa Raiz Identificada:** Volume restrito de imagens de corpo inteiro no dataset inicial (3 pares de 13), somado à dispersão dos gradientes quando o modelo precisa simultaneamente sintetizar vestuários pesados e cenários inéditos.

3. **Anatomia e Articulação de Mãos:**
   - **Diagnóstico:** Embora a imagem de Campos do Jordão tenha conseguido posicionar as mãos sobre a xícara sem mutilações grotescas, a articulação dos nós dos dedos e a espessura falângica ainda revelam imperfeições anatômicas perceptíveis a um olhar técnico atento.

---

### 🛠️ Roadmap de Mitigações Técnicas (Desenvolvimento Iterativo em Andamento)

Este repositório serve como ambiente vivo de engenharia. Em vez de descartar os experimentos, apresentamos os resultados da V1 e estamos trabalhando nas seguintes mitigações para os próximos commits:

| Desafio Identificado | Mitigação Técnica Planejada | Status |
| :--- | :--- | :---: |
| **Pele Cerosa (Waxy Artifact)** | Redução do CFG Scale de 4.5 para 3.2–3.6; calibração do `lora_alpha` para 16 (em vez de 32) para suavizar a injeção nas camadas de atenção. | 🟡 Em Teste |
| **Drift de Identidade Facial** | Inclusão de dataset de regularização facial (preservação de classe) e aumento de pares de corpo inteiro com enquadramentos em 3/4 e iluminação contrastante. | 🟡 Em Curadoria |
| **Overfitting em Passos Avançados** | Comparação formal de checkpoints intermediários (Steps 300, 400 e 500) para identificar o ponto ótimo de trade-off entre flexibilidade e identidade. | 🟢 Documentado |
| **Isolamento de Camadas LoRA** | Restringir o treinamento LoRA exclusivamente aos blocos centrais e profundos do Transformer, congelando os blocos iniciais responsáveis por detalhes de alta frequência e textura de pele. | 🟡 Próximo Commit |

---

## 💻 Inference & Reproduction Guide

### 1. Instalação do Ambiente

```bash
# Clonar o repositório
git clone https://github.com/gustavogrecov-tech/genai-finetuning-consistency-engine.git
cd genai-finetuning-consistency-engine

# Criar ambiente virtual Python
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate

# Instalar dependências de produção
pip install -r requirements.txt
```

### 2. Inferência em Python Nativo (Zero-OOM Pipeline)

O script [`inference/inference_pipeline.py`](./inference/inference_pipeline.py) gerencia dinamicamente o ciclo de vida dos módulos na GPU, descarregando o Text Encoder antes da execução do Transformer:

```bash
python inference/inference_pipeline.py \
  --prompt "sofia_woman, 28-year-old Scandinavian woman, sharp emerald green eyes, warm smile, elegant cashmere blazer, studio lighting, photorealistic 8k" \
  --steps 28 \
  --cfg 4.5 \
  --seed 42 \
  --output "outputs/sofia_resultado.png"
```

### 3. Integração com ComfyUI

Workflows de produção prontos para uso em ambiente gráfico estão disponíveis em [`inference/comfyui_workflows/`](./inference/comfyui_workflows/):
- `krea2_sofia_production.json` — Pipeline para Krea 2.1 com suporte a LoRA e FlowMatch Sampler.
- `qwen2_sofia_production.json` — Pipeline para Qwen-Image 2.1 com adaptadores TorchAO.

---

## 📂 Repository Structure

```
genai-finetuning-consistency-engine/
├── README.md                      # Documento mestre de apresentação do projeto e storytelling
├── requirements.txt               # Dependências do ecossistema PyTorch, Hugging Face e MLOps
├── .gitignore                     # Proteção contra artefatos pesados e credenciais
│
├── architecture/
│   └── pipeline_architecture.md   # Design de sistema, fundamentos matemáticos e fluxo Mermaid
│
├── configs/
│   └── train_sofia_qwen.yaml      # Configuração de treino ai-toolkit (uint3 TorchAO + FP8)
│
├── training/
│   └── train_krea2_sofia.py       # Script principal de fine-tuning (PEFT, FlowMatch, AdamW 8-bit)
│
├── inference/
│   ├── inference_pipeline.py      # Engine de inferência modular com dynamic offloading Zero-OOM
│   └── comfyui_workflows/         # Pipelines JSON para reprodução visual no ComfyUI
│       ├── krea2_sofia_production.json
│       └── qwen2_sofia_production.json
│
├── mlops/
│   ├── sentinel_watchdog.py       # FinOps & MLOps Sentinel: monitoramento, Hot SCP e Kill Switch
│   └── SENTINEL_SUMMARY.json      # Auditoria de custos e integridade de checkpoints
│
├── dataset/
│   ├── README.md                  # Metodologia de curadoria e Concept Disentanglement
│   └── samples/                   # 13 pares de imagem (1024x1024) + legendas .txt
│
└── benchmarks/
    ├── README.md                  # Relatório de validação empírica e critérios de auditoria
    └── results/                   # Imagens RAW de produção (Campos do Jordão e comparativos)
```

---

## 📬 Contato

- **Desenvolvedor:** Gustavo Sales Grecov
- **LinkedIn:** [Gustavo Sales Grecov](https://br.linkedin.com/in/gustavo-sales-grecov-530745350)
- **GitHub:** [@gustavogrecov-tech](https://github.com/gustavogrecov-tech)
- **Hugging Face:** [@gustavogrecov002](https://huggingface.co/gustavogrecov002)
- **E-mail:** [gustavo.grecov@aluno.ufabc.edu.br](mailto:gustavo.grecov@aluno.ufabc.edu.br)
- **Universidade:** Universidade Federal do ABC (UFABC)