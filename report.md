# collaborative_cnn_team05 — Project Report

## 1. Title
**Cross-Dataset CNN Collaboration — Leaf Disease Classification**  
Team: collaborative_cnn_team05  
Repository: `https://github.com/kshitizrajpatel/collaborative_cnn_team05.git`

## 2. Abstract
This project implements and compares two CNN models trained independently on two leaf-disease datasets: **PlantVillage** (Model v1) and **New Plant Diseases** (Model v2). We evaluate in-domain performance and cross-dataset generalization to study domain shift and model robustness. The dataset used for this assignment is the PlantVillage / New Plant Diseases combined experimental setup (Leaf Disease Classification).

## 3. Datasets
### 3.1 PlantVillage (used to train Model v1)
- Task: multi-class leaf disease classification across several crops (apple, potato, tomato, etc.).
- Notes: standard preprocessing (resize, normalize) and common augmentations used in training.
- Metrics files used: `metrics_v1.json` and cross-test `test_v2_user1.json`.

### 3.2 New Plant Diseases Dataset (used to train Model v2)
- Task: similar leaf disease classification but collected with different imaging conditions and class distributions, producing a domain shift.
- Metrics files used: `metric_v2.json` and cross-test `test_v1_user2.json`.

## 4. Models (versions & descriptions)

### 4.1 Model v1 (User 1)
- **Filename**: `models/model_v1.py` 
- **Weights**: `models/model_v1.pth` 
- **Architecture**: **MobileNetV2** (transfer learning backbone, final classifier adjusted to number of classes)
- **Loss**: CrossEntropyLoss
- **Notes**: Training produced very low learning progress (see Section 6).

### 4.2 Model v2 (User 2)
- **Filename**: `models/model_v2.py` 
- **Weights**: `trained_model2.keras` / `my_cnn_model2.h5`
- **Architecture**: Transfer-learning CNN (trained with stronger augmentation and longer schedule)
- **Notes**: Achieved strong in-domain performance on New Plant Diseases dataset.

## 5. Training setup (common)
- Framework: PyTorch / Keras (models present in both formats). 
- Hardware: Google Colab GPU (e.g., T4) used in training runs (noted in notebooks).
- Checkpointing: best checkpoints saved to `models/` (uploaded model files included).

## 6. Metrics & Evaluation (populated)

### 6.1 Model v1 — In-domain (PlantVillage)
Source: `metrics_v1.json`.
- Final validation loss: **0.6953**
- Final validation accuracy: **0.50** (50%)
- Epochs trained: **5**
  
### 6.2 Model v2 — In-domain (New Plant Diseases)
Source: `metric_v2.json`.
- Training accuracy: **0.9766**
- Validation accuracy: **0.9584**
- Evaluation / Test accuracy: **0.9548**
- Test precision: **0.9503**
- Test recall: **0.9610**
- Test loss: **0.1462**

These numbers indicate Model v2 learned well and generalizes strongly within its domain.

### 6.3 Cross-dataset testing (domain shift)

#### Model v1 (trained on PlantVillage) → tested on New Plant Diseases
Source: `test_v1_user2.json` (this file lists per-sample predictions).
- Observed behavior: Predictions are almost constant with `predicted_index` = 1 for many samples and top probabilities around ~0.53 vs ~0.47 for two top classes. Ground-truth labels are not present in the JSON, so per-class accuracy cannot be computed directly from that file alone.
- Practical takeaway: Model v1 is effectively guessing (or stuck predicting a single class) when evaluated on dataset B, consistent with the poor in-domain training performance (50% val acc).

#### Model v2 (trained on New Plant Diseases) → tested on PlantVillage
Source: `test_v2_user1.json` 
Key aggregated metrics (from the provided JSON):
- **Macro avg** precision: **0.94850**
- **Macro avg** recall: **0.94237**
- **Macro avg** f1-score: **0.94386**
- **Weighted avg** f1-score: **0.95702**
- **Micro avg** f1-score: **0.95705**

These results show Model v2 transfers quite well to PlantVillage (the cross-test), achieving high precision/recall across many classes — this suggests Model v2 learned robust features that generalize across datasets better than Model v1 did.

## 7. Results analysis & observations
- **Model v1 (MobileNetV2) failed to converge** properly on PlantVillage (validation stuck at 50%). Possible causes: bug in data loader/label mapping, learning rate/scheduler issue, incorrect loss computation, accidental freezing of the classifier, or insufficient training steps. This needs debugging.
- **Model v2 performed strongly** on its own dataset and also produced high cross-dataset metrics when tested on PlantVillage (`macro F1 ≈ 0.944`). This asymmetry (v2→A works well while v1→B fails) implies:
  - Model v2 likely used stronger augmentation, better regularization, or a longer/more correct training schedule.
  - Model v1's 50% validation suggests the model did not learn class-discriminative features in its training run and therefore cannot generalize.
- **Domain shift** is present but not equally damaging: a well-trained model (v2) handles cross-dataset variation well; a poorly trained model (v1) fails both in-domain and cross-domain.

## 8. Reproducibility checklist
-  Metrics files included: `metrics_v1.json`, `metric_v2.json`, `test_v1_user2.json`, `test_v2_user1.json`
-  Model files included (uploaded): `model_v1.pth`, `model_v1.py`, `model_v2.py`, `my_cnn_model2.h5`, `trained_model2.keras`

## 9. Conclusion
- **Model v2** is a successful, well-trained model with strong in-domain and cross-dataset performance.
- **Model v1** (MobileNetV2) requires debugging — its current run did not converge and produced effectively random predictions.


