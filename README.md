# Inhibitor Predictor for sEH (IPred-S 1.0)

### Overview
IPred-S 1.0 is a Streamlit web application implementing a highly stringent, consensus-based machine learning pipeline to predict soluble epoxide hydrolase (sEH) inhibitors. It employs 23 elite topological Mordred molecular descriptors alongside Random Forest (RF) and Support Vector Machine (SVM) models. The pipeline is rigorously validated on external test sets, decoy datasets, and PAINS datasets to enforce ultra-high precision and minimize false positives during virtual screening.

### Features
* **Flexible Input:** Input SMILES strings manually or via CSV upload.
* **Automated Feature Extraction:** Automatically embeds 3D coordinates and computes 23 highly predictive topological descriptors.
* **Consensus Prediction:** Evaluates molecules simultaneously using RF and SVM classifiers.
* **Strict Consensus Logic:** Implements independent Applicability Domain (AD) constraints and a hierarchical consensus voting rule to actively prevent structural decoys and reactive artifacts (PAINS) from being misclassified.
* **Exportable Data:** Download standard predictions and detailed raw/scaled descriptor tables for further analysis.

---

### Access the Web Tool
You can access and use the IPred-S 1.0 virtual screening pipeline directly through your web browser without any installation required:

🔗 **[Launch IPred-S 1.0 Web Tool Here](https://ipred-s-1-single-stage-screening.streamlit.app/)**

---

### Citation
If you utilize the IPred-S 1.0 webtool or concepts in your research, please cite:

> **IPred-S 1.0 Webtool** | Dileep Kumar et al. | Version 1.0 (2026).  
> **Webtool URL:** *(https://ipred-s-1-single-stage-screening.streamlit.app/)*

> **IPred-S 1.0: Consensus Machine Learning Framework for Predicting Soluble Epoxide Hydrolase (sEH) Inhibitors** | A. J. Martin, D. Kumar. | *Manuscript in preparation* (2026).

*(Final journal citation and DOI will be updated here once published and archived.)*

---

### Copyright & Intellectual Property

**© 2026 Manipal Academy of Higher Education (MAHE). All rights reserved.**

**Authors/Creators:** Dileep Kumar and Ajwin Joseph Martin

The source code, algorithms, consensus logic, and trained models associated with IPred-S 1.0 are the exclusive intellectual property of Manipal Academy of Higher Education (MAHE). This repository is hosted on the creators' personal account and made public for the sole purpose of deploying the Streamlit web application and facilitating transparency for academic peer review.

**Permissions:**
* You are permitted to view the source code for educational and peer-review purposes.
* You are permitted to use the deployed web tool via the provided Streamlit URL for your own virtual screening tasks, provided proper citation is given.

**Restrictions:**
* You may **NOT** copy, reproduce, distribute, modify, or create derivative works from this codebase.
* You may **NOT** use the code or models for any commercial or private non-commercial deployment without explicit written permission from the copyright owner (MAHE) and the authors.

For licensing inquiries or permission requests, please contact the authors directly.
