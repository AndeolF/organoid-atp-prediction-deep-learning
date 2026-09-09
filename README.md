# Organoid ATP Concentration Prediction

This repository contains the code for the **organoid ATP concentration prediction project** based on 48-hour videos (8 frames per video).
This project was completed as part of the **M2 Deep Learning course at CentraleSupélec** in the form of a **Kaggle competition**.
The project statement and full presentation can be found in the PDF [`projet_presentation.pdf`](projet_presentation.pdf).

---

## Repository Organization

- **`pipeline_w_ResNet_final`**: The final version of the code, integrating all iterative improvements (ABMIL, robust normalization, custom sampler, strong augmentation) and a **comprehensive monitoring dashboard** for diagnosing training quality (prediction distribution, MIL attention entropy, MAPE per ATP range, gradient norms per component, etc.).

- **`pyproject.toml`** / **`poetry.lock`**: Configuration and virtual environment lock files.

- **`.csv` files**: Submission files and test data for the project.

The notebook is organized into three main sections:

### **1. Dataset & DataLoader**
All data preparation code:
- Dataset parsing and grouping by well
- Train/val split isolated by patient
- Data augmentation (flips, rotations, Gaussian noise, crop/zoom)
- Random sampling of cavities per time window
- Logarithmic transformation of labels
- Exploratory analysis functions (label distribution statistics, image size statistics, split balance verification)

### **2. Model Architecture**
Definition of each model component (spatial encoder, temporal encoder, MIL aggregator, regressor) and assembly into the final model `OrganoidATPModel`.
**Dimensional sanity checks** are included to verify shape consistency at each step of the pipeline before training.

### **3. Training**
- Model definition with hyperparameters
- Optimizer (component-wise AdamW)
- Scheduler (cosine decay with warmup)
- Loss function and early stopping criteria
- Training loop includes progressive unfreezing of ResNet, gradient accumulation management, and automatic saving of the best checkpoint.

---

## Model Architecture

The pipeline consists of four cascading modules:


```
Images [N_cavités, 8, 1, 224, 224]
        │
        ▼
┌─────────────────────────┐
│  Spatial Encoder        │ Pre-trained ResNet (MicroNet)  
│  (per frame)            │  Fine-tuning of the last blocks
└─────────────────────────┘
        │  [N_cavities, 8, d_model]
        ▼
┌─────────────────────────┐
│  Temporal Encoder       │ Transformer (1 layer, 4 heads)
│  (per cavity)           │  Learned positional encoding over 8 timesteps
└─────────────────────────┘
        │ [N_cavities, d_model]
        ▼
┌─────────────────────────┐
│   MIL Aggregator        │ ABMIL: Gated Attention Pooling
│  (per well)             │  Aggregates N cavities → 1 well embedding
└─────────────────────────┘
        │ [d_model]
        ▼
┌─────────────────────────┐
│  MLP Regressor          │ Predicts log1p(ATP) → denormalized via expm1
└─────────────────────────┘
```



### Key Design Choices
- **ABMIL (Gated Attention Pooling) and Transformer MIL**
- **Log Space**: All targets are transformed to `log1p(ATP)` for training (initial distribution was heavy-tailed).
- **Huber Loss (delta=1.0)** in log space: Direct approximation of MAPE, robust to outliers.

---
## Handling Distribution Imbalance
Wells with low ATP (< 12.7 in log1p) are underrepresented and heavily penalize MAPE. Two combined strategies:
- **WeightedRandomSampler**: Low-ATP wells are sampled ~1.2-1.6× more frequently depending on the number of available cavities.
- **Strong Augmentation** (`strong=True`) for rare wells: More intense Gaussian noise, in addition to standard augmentations.

---
**Grade obtained:**  **A+**