# 🍃 collaborative_cnn_team05: Leaf Disease Classification Collaboration

This repository documents the completion of the **Fork-Based Collaborative CNN Development** project, which focused on designing, training, and evaluating Convolutional Neural Network (CNN) models for **Leaf Disease Classification** across two private, non-shared datasets.

This project utilized the full GitHub collaboration workflow (forks, branches, pull requests, and issues) and emphasized cross-domain testing and model comparison.

---

## 🎯 Project Objective

The primary objective was to collaboratively design, train, and evaluate CNN models for the same image classification task, focusing on deep learning implementation, the full GitHub collaboration workflow, and cross-domain testing and model comparison.

### Scenario and Dataset Allocation

[cite_start]Two users investigated the Leaf Disease Classification problem, but each user trained their model on a **separate, proprietary dataset** that could not be shared.

| Team Member | Dataset Name | Role |
| :--- | :--- | :--- |
| **User 1 (Base Repo Owner)** |Plant Village Dataset  | Developed Model V1, Cross-tested Model V2 |
| **User 2 (Fork Owner)** | New Plant Diseases Dataset | Tested Model V1, Developed Model V2 |

---

## ⚙️ Repository Structure and Contents

[cite_start]The following structure, as required by the assignment, holds all the project files and deliverables [cite: 70-88]:

| Path | Description |
| :--- | :--- |
| `README.md` | This repository overview. |
| `requirements.txt` | Lists all necessary Python dependencies. |
| `models/` | Contains model definitions (`model_v1.py`, `model_v2.py`) and saved weights (`.pth` or `.h5`). |
| `notebooks/` | Contains Jupyter Notebooks used for training and testing (e.g., `train_v1.ipynb`, `test_v2.ipynb`). |
| `results/` | Contains performance metrics (`metrics_v1.json`, `metrics_v2.json`) and cross-test results (`test_v1_user2.json`, `test_v2_user1.json`). |
| `report.md` | **Final Deliverable:** Contains the required summary comparison of model v1 vs model v2, metrics on both datasets, and observations on generalization and domain shift |

---

