# Hybrid Multimodal Architecture for bitnet.cpp

> Implementação de referência de uma arquitetura híbrida multimodal em torno do framework `bitnet.cpp` (Microsoft), combinando encoders especialistas em alta precisão com um núcleo de linguagem ternário `{-1, 0, +1}`. Estruturada como pipeline de pesquisa reproduzível para Google Colab.

---

## Visão Geral

O projeto organiza-se em dois componentes que se complementam:

| Componente | Localização | Papel |
|---|---|---|
| **Biblioteca de módulos** | `utils/` | Código Python reutilizável, importável, documentado com docstrings NumPy e type hints |
| **Notebooks de pipeline** | `notebooks/` | Artefatos executáveis por fase, autocontidos, prontos para Colab |

A separação é intencional: os `utils/` são a **camada de abstração estável**, os notebooks são o **protocolo de experimento executável**. Os notebooks podem operar de forma autocontida (inline) ou importar os `utils/` quando o projeto estiver montado no Drive.

---

## Estrutura do Repositório

```
multimodal-ternary-llm/
│
├── notebooks/                       # Pipeline de implementação — 1 notebook por fase
│   ├── 00_baseline_teacher.ipynb   # Fase 0 — Teacher FP16/BF16, interfaces, critérios de aceite
│   ├── 01_connector_pretraining.ipynb  # Fase 1 — Pré-treinamento isolado do conector MLP
│   ├── 02_multimodal_alignment.ipynb   # Fase 2 — Distilação KL + alignment loss
│   ├── 03_ternary_transition.ipynb     # Fase 3 — QAT contínuo, conversão ternária
│   └── 04_export_deploy.ipynb          # Fase 4 — Exportação .safetensors, bitnet.cpp, benchmarks
│
└── utils/                           # Módulos Python reutilizáveis (biblioteca interna)
    ├── __init__.py
    ├── bit_layers.py                # BitLinear, quantização ternária, ReLU², replace_linear
    ├── colab_utils.py               # Drive, seeds, logging, checkpoints, device
    ├── connector.py                 # ModalityConnector — MLP 2 camadas GeLU, BF16
    ├── heads.py                     # ClassificationHead, RegressionHead, ActionHead
    ├── losses.py                    # HybridLoss, distillation KL, alignment, stability
    └── training.py                  # QATScheduler, train_one_epoch, build_optimiser
```

---

## utils/ — Biblioteca de Módulos

Os módulos em `utils/` são código de biblioteca: reutilizáveis, totalmente documentados (docstrings NumPy/SciPy), com type hints em todas as assinaturas. São independentes dos notebooks e podem ser importados diretamente.

### `bit_layers.py`

Implementação do `BitLinear` — substituto direto de `nn.Linear` — com quantização ternária de pesos via absmean scaling e estimador straight-through para retropropagação. Inclui:

- `BitLinear` — camada linear ternária treinável
- `quantise_weights_ternary` — quantização `{-1, 0, +1}`
- `quantise_activations_8bit` — quantização de ativações absmax
- `replace_linear_with_bitlinear` — conversão recursiva de um módulo inteiro
- `ReLUSquared` — ativação ReLU² conforme especificado no BitNet

```python
from utils.bit_layers import BitLinear, replace_linear_with_bitlinear

# Substituir todas as projeções lineares de um backbone existente
model = replace_linear_with_bitlinear(model, skip_names=["lm_head"])
```

### `connector.py`

`ModalityConnector` — MLP de 2 camadas com ativação GeLU, mantido em BF16. Projeta a saída de encoders especializados (visão, séries temporais) para o espaço de embedding do backbone BitNet.

```python
from utils.connector import ModalityConnector

connector = ModalityConnector(d_enc=1024, d_model=2048)
projected = connector(encoder_output)   # (B, seq, 2048)
```

### `losses.py`

`HybridLoss` — agregador configurável das 5 funções de perda da especificação (Seção 6):

| Componente | Função |
|---|---|
| `language` | Cross-entropy autoregressiva |
| `distill` | KL divergence logits teacher→student |
| `align` | Cosine similarity embeddings encoder/connector |
| `task` | Classificação, regressão ou ação (plug-in) |
| `stability` | Regularização de outliers de ativação (BitNet a4.8) |

```python
from utils.losses import HybridLoss

loss_fn = HybridLoss(w_language=1.0, w_distill=0.5, w_align=0.1, temperature=4.0)
losses = loss_fn(student_logits, labels, teacher_logits, student_embeds, teacher_embeds)
# losses["total"], losses["language"], losses["distill"], losses["align"], ...
```

### `heads.py`

Cabeças de saída por tarefa, mantidas em FP16/BF16:

- `ClassificationHead` — pooling (mean ou cls) + linear, com dropout
- `RegressionHead` — regressão contínua com MSE ou Huber loss
- `ActionHead` — controle/ação com tanh bounds, float32 por padrão

### `training.py`

Utilitários de treinamento:

- `build_optimiser` — AdamW com grupos de parâmetros diferenciados por weight decay
- `build_scheduler` — warmup linear + cosine annealing
- `train_one_epoch` — loop de treinamento com AMP, grad clip e logging
- `QATScheduler` — ramp gradual de força de quantização 0.0 → 1.0
- `freeze_module` / `unfreeze_module` — controle granular de gradientes
- `get_trainable_param_count` — diagnóstico de parâmetros

```python
from utils.training import QATScheduler, build_optimiser

optimiser = build_optimiser(model, lr=1e-4)
qat = QATScheduler(total_steps=3000, warmup_fraction=0.2)

for step in range(3000):
    strength = qat.get_strength()       # 0.0 → 1.0
    set_quant_strength(model, strength)
    qat.step()
```

### `colab_utils.py`

Infraestrutura de sessão para Google Colab:

- `mount_drive` — monta o Drive (no-op fora do Colab)
- `set_global_seed` — seeds determinísticas (`random`, `numpy`, `torch`, CUDA)
- `resolve_device` — seleciona GPU/CPU e loga VRAM disponível
- `save_checkpoint` / `load_checkpoint` — persistência em Drive
- `configure_logging` — logging estruturado com file handler opcional
- `ensure_drive_dirs` — cria a árvore de diretórios do projeto no Drive

---

## notebooks/ — Pipeline de Experimento

Cada notebook corresponde a uma fase do pipeline e é **autocontido**: instala dependências, fixa seeds, monta o Drive e documenta cada decisão com células Markdown antes de cada bloco de código. O tom é formal e acadêmico, adequado para uso como artefato de pesquisa reproduzível.

### Uso dos `utils/` nos notebooks

Os notebooks definem os módulos principais inline por padrão (portabilidade no Colab Free). Para usar os `utils/`, adicionar no início de qualquer notebook:

```python
import sys
sys.path.insert(0, "/content/drive/MyDrive/multimodal-ternary-llm")

from utils.bit_layers  import BitLinear, replace_linear_with_bitlinear
from utils.connector   import ModalityConnector
from utils.losses      import HybridLoss
from utils.training    import QATScheduler, build_optimiser
from utils.heads       import ClassificationHead, ActionHead
from utils.colab_utils import set_global_seed, mount_drive, save_checkpoint
```

### Estrutura obrigatória de cada notebook

Conforme a Seção 0.4 da especificação:

```
Célula 1   — Markdown: Título, Resumo, Fase
Célula 2   — Markdown: Índice
Célula 3   — Code:     !pip install (versões fixadas)
Célula 4   — Code:     Imports + seeds + device + constantes
Célula 5   — Code:     Google Drive mount + criação de diretórios
Célula 6   — Markdown: Fundamentação teórica
Células N  — Code/Markdown alternados (cada bloco precedido por Markdown)
Célula N-1 — Code:     Validação e métricas
Célula N   — Code:     Salvamento de artefatos no Drive
Célula fim — Markdown: Conclusões e próximos passos
```

| Notebook | Fase | Componentes treináveis | Loss ativa |
|---|---|---|---|
| `00_baseline_teacher` | 0 | Nenhum (só avaliação) | — |
| `01_connector_pretraining` | 1 | Conector MLP | Language CE |
| `02_multimodal_alignment` | 2 | Conector + top-N blocos do backbone | Language + Distillation KL + Alignment |
| `03_ternary_transition` | 3 | Backbone completo (QAT) | Language CE |
| `04_export_deploy` | 4 | Nenhum (congelado) | — (benchmarks) |

---

## Política de Precisão

| Componente | Precisão | Justificativa |
|---|---|---|
| Encoders especialistas | FP16/BF16 | Preservar fidelidade perceptual |
| Connector MLP | BF16 | Alta precisão no alinhamento cross-modal |
| Backbone (BitNet) | Ternário `{-1,0,+1}` | Eficiência de inferência extrema |
| Ativações do backbone | INT8 (baseline), INT4 (avançado) | Balancear custo e outlier sensitivity |
| Cabeças de tarefa | FP16/BF16 | Sensibilidade numérica na saída |

---

## Fluxo de Persistência (Google Drive)

```
/content/drive/MyDrive/multimodal-ternary-llm/
├── checkpoints/
│   ├── phase0_baseline/tokenizer/
│   ├── phase1_connector/connector_phase1.pt
│   ├── phase2_alignment/phase2_aligned.pt
│   ├── phase3_ternary/phase3_ternary_backbone.pt
│   └── phase4_deploy/
├── export/
│   ├── ternary_backbone.safetensors
│   ├── tokenizer/
│   └── deployment_report.json
├── logs/
└── metrics/
    ├── phase0_baseline_metrics.json
    ├── modality_interfaces.json
    ├── acceptance_criteria.json
    ├── phase1_metrics.json
    ├── phase2_metrics.json
    ├── phase3_metrics.json
    └── phase4_final_report.json
```

---

## Requisitos

| Recurso | Mínimo | Recomendado |
|---|---|---|
| GPU | T4 (Colab Free) | L4 / A100 80 GB |
| Python | 3.10+ | 3.11 |
| PyTorch | 2.x | 2.3+ |
| RAM (CPU) | 12 GB | 32 GB |
| Armazenamento Drive | 10 GB | 50 GB |

> **Fases 0 e 1** executam em T4. **Fases 2 e 3** (distilação + QAT de backbone completo) requerem L4 ou A100 para escala de produção. **Fase 4** (benchmarks) é limitada pelo hardware disponível.

---

## Referências Arquiteturais

- **BitNet / BitNet b1.58** — Wang et al., 2023–2024. [arXiv:2310.11453](https://arxiv.org/abs/2310.11453) / [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
- **BitNet a4.8** — Wang et al., 2024. [arXiv:2411.04965](https://arxiv.org/abs/2411.04965)
- **BitVLA** — Padrão multimodal encoder full-precision + backbone ternário
- **Qwen2.5-VL** — Dynamic resolution, window attention, OCR e vídeo longo
- **VL-Mamba / Mamba** — State space models para sequências longas com escala linear
- **bitnet.cpp** — [github.com/microsoft/BitNet](https://github.com/microsoft/BitNet)
- **T-MAC** — Kernels low-bit gerais recomendados pelo repositório bitnet.cpp