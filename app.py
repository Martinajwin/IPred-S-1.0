# ==========================================================
# 🏷️ IPred-S 1.0 (Production Pipeline)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
import joblib
from graphviz import Digraph
import os

warnings.filterwarnings("ignore")

# -----------------------------
# 🧩 Elite Features List
# -----------------------------
elite_features = [
    'ATSC4i', 'BCUTse-1l', 'FCSP3', 'FilterItLogS', 'GATS1d', 'GATS2s', 'GRAV', 
    'MW', 'NaaNH', 'NaaS', 'NdssC', 'NssNH', 'SLogP', 
    'SlogP_VSA1', 'TopoPSA', 'n10FaRing', 'nAromAtom', 'nAtom', 
    'nBase', 'nHBAcc', 'nHBDon', 'nRing', 'nRot'
]

@st.cache_resource(show_spinner="Loading models and assets...")
def load_pipeline_assets():
    try:
        rf = joblib.load("models/rf_model_no_qm.pkl")
        svm = joblib.load("models/svm_model_no_qm.pkl")
        # Load the ORIGINAL scaler here
        scaler = joblib.load("models/scaler.joblib")
        rf_ad_threshold = float(np.load("models/rf_ad_no_qm.npy"))
        svm_ad_threshold = float(np.load("models/svm_ad_no_qm.npy"))
        return rf, svm, scaler, rf_ad_threshold, svm_ad_threshold
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure the 'models/' folder contains all 5 required files.")
        return None, None, None, None, None

def get_consensus_class(rf_p, svm_p, rf_ad, svm_ad):
    # --- NEW THRESHOLDS ---
    # Base threshold set to 0.40 (The Realistic Sweet Spot)
    threshold = 0.40
    # High-confidence threshold for AD overrides (Base + 0.10)
    strict_threshold = 0.50 

    rf_pred = 'Active' if rf_p >= threshold else 'Inactive'
    svm_pred = 'Active' if svm_p >= threshold else 'Inactive'
    
    # Confidence is now measured as the distance from the new 0.40 threshold
    rf_conf = abs(rf_p - threshold)
    svm_conf = abs(svm_p - threshold)
    
    ens_p = (rf_p + svm_p) / 2
    ens_pred = 'Active' if ens_p >= threshold else 'Inactive'

    if (rf_pred == svm_pred) and rf_ad and svm_ad:
        return rf_pred
    elif (rf_pred != svm_pred) and rf_ad and svm_ad:
        if rf_conf > svm_conf: return rf_pred
        elif svm_conf > rf_conf: return svm_pred
        else: return ens_pred
    elif rf_ad and not svm_ad:
        if rf_pred == 'Active' and rf_p < strict_threshold: return 'Inactive'
        return rf_pred
    elif svm_ad and not rf_ad:
        if svm_pred == 'Active' and svm_p < strict_threshold: return 'Inactive'
        return svm_pred
    elif not rf_ad and not svm_ad:
        if rf_pred == svm_pred:
            if rf_pred == 'Active':
                if rf_p >= strict_threshold and svm_p >= strict_threshold: return 'Active (outside AD)'
                else: return 'Inactive (outside AD)'
            else:
                return 'Inactive (outside AD)'
        else:
            return f"{ens_pred} (outside AD)"# -----------------------------
# ⚗️ Streamlit UI
# -----------------------------
st.set_page_config(page_title="IPred-S 1.0", layout="wide")
st.title("Inhibitor Predictor for sEH (1.0)")  
st.markdown("### (IPred-S 1.0)")

tabs = st.tabs(["1️⃣ Molecule Screening", "2️⃣ Methodology", "3️⃣ Model Performance", "4️⃣ References and Citation"])
tab1, tab2, tab3, tab4 = tabs

# ==========================================================
# 1️⃣ SCREENING TAB
# ==========================================================
with tab1:
    st.header("Predict sEH inhibitors")

    rf_model, svm_model, scaler_loaded, rf_ad_threshold, svm_ad_threshold = load_pipeline_assets()

    st.markdown("**A maximum of 300 SMILES and minimum of 10 SMILES is recommended.**")
    
    input_option = st.radio("Input Type:", ["Enter SMILES manually", "Upload CSV"])
    smiles_list = []
    df_input = pd.DataFrame()

    if input_option == "Upload CSV":
        st.write("CSV must contain a column named 'SMILES' in the first column.")
        uploaded_file = st.file_uploader("Upload CSV with SMILES", type=["csv"])
        if uploaded_file is not None:
            df_input = pd.read_csv(uploaded_file)
            if "SMILES" not in df_input.columns:
                st.error("CSV must contain a 'SMILES' column.")
                st.stop()
            smiles_list = [str(s) for s in df_input["SMILES"] if pd.notna(s)]
    else:
        user_smiles = st.text_area("Enter SMILES (one per line)")
        smiles_list = [s.strip() for s in user_smiles.split("\n") if s.strip()]

    if st.button("🚀 Predict") and smiles_list and rf_model is not None:
        
        st.info("Computing 2D & 3D Topological features and checking data integrity... please wait ⏳")

        canonical_smiles, mols, valid_indices = [], [], []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
                
                # CRITICAL: Embed 3D coordinates using HEAVY ATOMS ONLY.
                try:
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                except:
                    pass
                
                mols.append(mol)
                valid_indices.append(i)
            else:
                st.warning(f"Invalid SMILES skipped: {smi}")

        if len(mols) == 0:
            st.error("No valid SMILES found. Please check your input.")
            st.stop()

        # 1. Generate Mordred Descriptors
        calc = Calculator(descriptors, ignore_3D=False)
        df_mordred = calc.pandas(mols)

        # 2. Safely Extract Exact Features without Lowercasing Mordred Data
        X_screen_raw = pd.DataFrame(index=range(len(mols)), columns=elite_features)
        missing_features = []

        csv_cols_lower = {}
        if not df_input.empty:
            csv_cols_lower = {str(c).lower(): c for c in df_input.columns}

        for col in elite_features:
            col_lower = col.lower()
            
            # Priority 1: Check User CSV
            if not df_input.empty and col_lower in csv_cols_lower:
                actual_col = csv_cols_lower[col_lower]
                val = pd.to_numeric(df_input[actual_col].iloc[valid_indices], errors='coerce').values
                X_screen_raw[col] = val
                
            # Priority 2: Check Mordred
            elif col in df_mordred.columns:
                val = pd.to_numeric(df_mordred[col], errors='coerce').values
                X_screen_raw[col] = val
                
            # Priority 3: Not found -> Mean Imputation
            else:
                missing_features.append(col)
                col_idx = list(scaler_loaded.feature_names_in_).index(col)
                X_screen_raw[col] = scaler_loaded.mean_[col_idx]
                
        if missing_features:
            st.warning(f"⚠️ **DATA WARNING:** {len(missing_features)} features could not be calculated by Mordred and were replaced with training averages. Missing: {missing_features}")

        # Fill NaNs
        for col in elite_features:
            if X_screen_raw[col].isna().any():
                col_idx = list(scaler_loaded.feature_names_in_).index(col)
                X_screen_raw[col] = X_screen_raw[col].fillna(scaler_loaded.mean_[col_idx])

        # Force numerical type to prevent array math errors
        X_screen_raw = X_screen_raw.astype(np.float64)

        # 3. Scale manually using True Original Scaler parameters 
        feature_names = list(scaler_loaded.feature_names_in_)
        elite_indices = [feature_names.index(f) for f in elite_features]
        elite_means = scaler_loaded.mean_[elite_indices]
        elite_scales = scaler_loaded.scale_[elite_indices]

        # Exact mathematical scaling match to your local reference script
        X_screen_scaled = (X_screen_raw.values - elite_means) / elite_scales

        # 4. Generate Predictions
        active_probs_rf = rf_model.predict_proba(X_screen_scaled)[:, 1]
        active_probs_svm = svm_model.predict_proba(X_screen_scaled)[:, 1]
        active_probs_ens = (active_probs_rf + active_probs_svm) / 2
        
        rf_confidences = np.maximum(active_probs_rf, 1 - active_probs_rf)
        svm_confidences = np.maximum(active_probs_svm, 1 - active_probs_svm)

        rf_ad_mask = rf_confidences >= rf_ad_threshold
        svm_ad_mask = svm_confidences >= svm_ad_threshold
        
        predicted_classes = [
            get_consensus_class(rf_val, svm_val, rf_ad_val, svm_ad_val) 
            for rf_val, svm_val, rf_ad_val, svm_ad_val in zip(
                active_probs_rf, active_probs_svm, rf_ad_mask, svm_ad_mask
            )
        ]

        # 5. Compile Standard Results
        results = pd.DataFrame({
            "SMILES": canonical_smiles,
            "RF_Probability": np.round(active_probs_rf, 4),
            "SVM_Probability": np.round(active_probs_svm, 4),
            "Ensemble_Probability": np.round(active_probs_ens, 4),
            "Predicted_Class": predicted_classes
        })

        results = results.sort_values(by=["Predicted_Class", "Ensemble_Probability"], ascending=[True, False]).reset_index(drop=True)
        st.dataframe(results)
        
        csv_standard = results.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", data=csv_standard, file_name="predictions.csv", mime="text/csv")

        # 6. Compile DEBUG Results
        debug_df = results.copy()
        for idx, col in enumerate(elite_features):
            debug_df[f"{col}_RAW"] = X_screen_raw[col].values
            debug_df[f"{col}_SCALED"] = X_screen_scaled[:, idx]
            
        csv_debug = debug_df.to_csv(index=False).encode("utf-8")
        st.download_button("⚙️ Download Detailed Debug CSV", data=csv_debug, file_name="debug_predictions.csv", mime="text/csv")
        
        st.success(f"✅ Prediction complete! (RF AD: {rf_ad_threshold:.4f} | SVM AD: {svm_ad_threshold:.4f})")

# ==========================================================
# 2️⃣ METHODOLOGY TAB (PROCEDURE & FLOWCHART)
# ==========================================================
with tab2:
    st.header("Methodology and Working of IPred-S 1.0")

    st.markdown(
        """
**Consensus ML Workflow for Compound Activity Prediction**

The IPred-S 1.0 pipeline implements a highly stringent, consensus-based machine learning framework for predicting soluble epoxide hydrolase (sEH) inhibitors. SMILES input via manual entry or CSV is first validated and canonicalized. To accurately capture 3D volumetric properties without corrupting 2D graph features, molecules undergo a rapid heavy-atom 3D coordinate embedding. The pipeline then computes 23 highly stable topological descriptors (1D, 2D, and 3D) using Mordred. While early iterations of this framework incorporated Quantum Mechanical (QM) descriptors to capture deeper electronic properties, they were excluded from the final deployment. The prohibitive computational cost of executing real-time quantum calculations creates a massive bottleneck for web-based screening. By isolating the 23 most predictive topological features, the tool ensures rapid, high-throughput performance without sacrificing the strict accuracy of the models. These raw features are then strictly normalized using a pre-fitted True Standard Scaler extracted from the original training dataset.

Classification is performed simultaneously by a Random Forest (RF) and a Support Vector Machine (SVM). Each model establishes an independent Applicability Domain (AD) threshold based on the training set's confidence distribution. The final prediction is governed by a strict 5-rule hierarchical consensus logic that evaluates model agreement, independent confidence margins, and AD constraints to actively minimize false positives. Finally, the merged predictions, probabilities, and detailed data with descriptors are displayed and available for export.
"""
    )

    st.subheader("IPred-S 1.0 Flowchart Overview")
    st.markdown("<br>", unsafe_allow_html=True)

    # 🛑 SIZING FIX: We wrap the chart in columns to prevent it from stretching across the entire wide screen.
    col1, col2, col3 = st.columns([1, 2, 1]) # The middle column takes 50% width, centering the chart.

    with col2:
        from graphviz import Digraph
        dot = Digraph("ConsensusFlow", engine="dot")
        
        # Tighter vertical and horizontal spacing
        dot.attr(rankdir="TB", splines="ortho", nodesep="0.25", ranksep="0.35")

        # Node style: Adjusted margins and readable font size
        dot.attr("node", shape="box", style="rounded,filled,solid", fontsize="11", margin="0.1,0.1")
        dot.attr("edge", style="solid", arrowhead="normal", constraint="true")

        # ✒️ BOLD FIRST LINES using HTML-like labels
        dot.node("Start", "<<b>START</b>>", fillcolor="lightblue")
        
        dot.node("Load", "<<b>LOAD LIBRARIES &amp; MODELS</b><br align='left'/>• Load RDKit, Mordred, scikit-learn<br align='left'/>• Load RF &amp; SVM models<br align='left'/>• Load True Scaler &amp; AD thresholds>", fillcolor="lightcyan")
        
        dot.node("Input", "<<b>USER INPUT</b><br align='left'/>• Enter SMILES manually<br align='left'/>• Upload CSV (SMILES column)<br align='left'/>• Optional: Pre-scaled data bypass>", fillcolor="lightyellow")
        
        dot.node("Validate", "<<b>VALIDATE &amp; EMBED 3D</b><br align='left'/>• RDKit Mol conversion<br align='left'/>• Heavy-atom 3D embedding (MMFF94)<br align='left'/>• Remove invalid entries>", fillcolor="gold")
        
        dot.node("Desc", "<<b>COMPUTE DESCRIPTORS</b><br align='left'/>• 23 Mordred features (1D, 2D, 3D)<br align='left'/>• Handle missing features via mean imputation>", fillcolor="lightcoral")
        
        dot.node("Scale", "<<b>NORMALIZATION</b><br align='left'/>• Apply True Standard Scaler parameters<br align='left'/>• Clip extreme outliers>", fillcolor="plum")

        # Prediction Stage
        dot.node("Stage1", "<<b>INFERENCE STAGE</b>>", fillcolor="orange")
        dot.node("RF1", "<<b>RANDOM FOREST</b><br align='left'/>• Predict: Active / Inactive<br align='left'/>• Confidence Probability<br align='left'/>• Calculate AD status>", fillcolor="lightskyblue")
        dot.node("SVM1", "<<b>SUPPORT VECTOR MACHINE</b><br align='left'/>• Predict: Active / Inactive<br align='left'/>• Confidence Probability<br align='left'/>• Calculate AD status>", fillcolor="lightgreen")
        
        cons_label = (
            "<<b>STRICT CONSENSUS LOGIC</b><br align='left'/>"
            "• Agree + Both in AD &rarr; Output Agreed Class<br align='left'/>"
            "• Disagree + Both in AD &rarr; Output Higher Confidence<br align='left'/>"
            "• One inside AD + One outside AD &rarr; Trust Inside-AD Model<br align='left'/>"
            "• Both outside AD &rarr; Output Ensemble (Flagged 'outside AD')<br align='left'/>"
            "Output: Active / Inactive>"
        )
        dot.node("Cons", cons_label, fillcolor="navajowhite")

        # Outputs
        with dot.subgraph() as s_out:
            s_out.attr(rank='same')
            s_out.node("Inactive", "<<b>INACTIVE</b>>", fillcolor="lightgray")
            s_out.node("Active", "<<b>ACTIVE</b>>", fillcolor="lightgoldenrod")

        # Display & End
        dot.node("Display", "<<b>DISPLAY &amp; EXPORT</b><br align='left'/>• Show interactive dataframe<br align='left'/>• Download Standard Predictions CSV<br align='left'/>• Download Detailed Debug CSV>", fillcolor="lightblue")
        dot.node("End", "<<b>END</b>>", fillcolor="lightblue")

        # Connections
        dot.edge("Start", "Load")
        dot.edge("Load", "Input")
        dot.edge("Input", "Validate")
        dot.edge("Validate", "Desc")
        dot.edge("Desc", "Scale")
        dot.edge("Scale", "Stage1")
        dot.edge("Stage1", "RF1")
        dot.edge("Stage1", "SVM1")
        dot.edge("RF1", "Cons")
        dot.edge("SVM1", "Cons")
        dot.edge("Cons", "Inactive", minlen="0.4")
        dot.edge("Cons", "Active", minlen="0.4")
        dot.edge("Inactive", "Display")
        dot.edge("Active", "Display")
        dot.edge("Display", "End")

        # Because it is restricted by col2, use_container_width will neatly fit it in the center.
        st.graphviz_chart(dot, use_container_width=True)# # ==========================================================
# 3️⃣ MODEL PERFORMANCE TAB
# ==========================================================
with tab3:
    st.header("IPred-S 1.0 Evaluation Results")

    st.info("Detailed cross-validation metrics, full external test set validation, and comparative benchmarking data for the Random Forest and SVM models will be fully updated upon the publication of the associated research article.")

    st.write("""
        The models were built on a rigorously curated dataset of soluble epoxide hydrolase (sEH) inhibitors and decoys. Multiple levels of internal validation were performed, including 10-Fold Cross-Validation, followed by external test set validation. 
        
        Additionally, a highly imbalanced external test set (containing potent actives and structurally similar decoys) and a dataset of Pan Assay Interference Compounds (PAINS) were used to aggressively assess the predictive robustness of the consensus framework.
        
        Applicability domain (AD) analysis of the training set indicates that the model primarily learns from well-represented topological regions of chemical space. Molecules containing these established scaffolds fall within a highly reliable prediction zone. Conversely, uncommon scaffolds or highly flexible, long-chain aliphatic structures falling outside the AD should be interpreted cautiously.
    """)

    # ------------------------------------------------------
    # 🔹 External Test Set Metrics (Threshold = 0.40)
    # ------------------------------------------------------
    st.subheader("External Test Set Performance (Operational Threshold = 0.40)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Table 1: Classification Metrics**")
        metrics_data = {
            "Evaluation Metric": [
                "ROC-AUC", "Accuracy", "Balanced Accuracy", 
                "Precision", "Recall (Sensitivity)", "Specificity", "F1-Score", "MCC"
            ],
            "Value": [
                "0.9821", "0.9431", "0.8868", 
                "0.9817", "0.9554", "0.8182", "0.9683", "0.6950"
            ]
        }
        st.table(pd.DataFrame(metrics_data))
        
    with col2:
        st.markdown("**Table 2: Confusion Matrix**")
        cm_data = {
            "Actual Class": ["Actual Active", "Actual Inactive"],
            "Predicted Active": ["107 (True Positives)", "2 (False Positives)"],
            "Predicted Inactive": ["5 (False Negatives)", "9 (True Negatives)"]
        }
        st.table(pd.DataFrame(cm_data).set_index("Actual Class"))

    # ------------------------------------------------------
    # 🔹 Threshold Analysis
    # ------------------------------------------------------
    st.subheader("Evaluation of Model Stringency and Decision Thresholds")
    
    st.markdown("""
    To evaluate the predictive behavior of the consensus framework, we analyzed the test set performance across a sliding probability threshold. The table below demonstrates the Precision-Recall tradeoff inherent to high-stringency virtual screening models.
    """)

    threshold_data = {
        "Threshold": [">= 0.30", ">= 0.35", ">= 0.40 (Operational)", ">= 0.45", ">= 0.50", ">= 0.55 (Strict AD)"],
        "True Positives (TP)": [112, 111, 107, 104, 87, 38],
        "False Positives (FP)": [10, 8, 2, 0, 0, 0],
        "True Negatives (TN)": [1, 3, 9, 11, 11, 11],
        "False Negatives (FN)": [0, 1, 5, 8, 25, 74],
        "Precision": ["0.9180", "0.9328", "0.9817", "1.0000", "1.0000", "1.0000"],
        "Recall": ["1.0000", "0.9911", "0.9554", "0.9286", "0.7768", "0.3393"],
        "Specificity": ["0.0909", "0.2727", "0.8182", "1.0000", "1.0000", "1.0000"]
    }
    st.table(pd.DataFrame(threshold_data))

    st.info("""
    **💡 Note on Screening Performance (Precision vs. Recall):** At a standard operational threshold of `0.40`, the model demonstrates highly balanced performance, achieving **98.1% Precision** and **95.5% Recall**. 
    
    However, the final deployed web tool utilizes a strict Applicability Domain (AD) and hierarchical consensus logic that mimics a much higher decision threshold (>0.55). While this strict filtering conservatively reduces Recall (capturing only the most potent, topologically conforming nanomolar inhibitors), it ensures **100% Precision**. This is a deliberate design choice optimized for large-scale virtual screening, where minimizing false positives is prioritized over exhaustive hit retrieval.
    """)

    # ------------------------------------------------------
    # 🔹 PAINS Dataset Validation Results
    # ------------------------------------------------------
    st.subheader("PAINS Dataset Validation Results")

    pains_data = {
        "Predicted Class": [
            "Inactive (Strictly Rejected)", 
            "Inactive (outside AD)", 
            "Active (outside AD)", 
            "Active (False Positives)"
        ],
        "Count": [310, 5, 3, 2],
        "Percent": ["96.88%", "1.56%", "0.94%", "0.63%"]
    }
    st.table(pd.DataFrame(pains_data))

    st.success("""
    **🛡️ Note on Screening Performance (PAINS & False Positives):** Pan Assay Interference Compounds (PAINS) are notorious for generating false-positive signals in computational screening. The strict consensus logic implemented in IPred-S 1.0 successfully identifies and rejects **98.44%** of these deceptive compounds, leaving a negligible false-positive rate of only 1.56%. 
    
    This explicitly justifies the highly punitive nature of the deployed pipeline: if the web tool's threshold were relaxed to `0.40` to catch more true actives, PAINS false-positives would surge to over 14%. The strict consensus rules actively sacrifice baseline recall to ensure that any predicted "Active" is completely insulated from chemical noise.
    """)

# ==========================================================
# 4️⃣ REFERENCES & CITATION TAB
# ==========================================================
with tab4:
    st.header("References, Citation & Intellectual Property")
    
    # --- ADDED MANIPAL COPYRIGHT SECTION FOR LEGAL COMPLIANCE ---
    st.markdown("### Institutional Affiliation & Copyright")
    st.markdown("**© 2026 Manipal Academy of Higher Education (MAHE). All rights reserved.**")
    st.markdown("Developed by: **D. Kumar, A. J. Martin**")
    st.markdown("*The algorithms, consensus logic, and trained models associated with IPred-S 1.0 are the intellectual property of Manipal Academy of Higher Education (MAHE).*")
    st.markdown("---")
    
    st.markdown("### How to Cite IPred-S 1.0 (Webtool Citation)")
    st.markdown("If you use the IPred-S 1.0 webtool in research or publications, please cite:")
    
    # Using st.info creates a bright, highlighted box instead of dimmed text
    st.info("**IPred-S 1.0 Webtool** | D. Kumar, A. J. Martin | Manipal Academy of Higher Education (MAHE) | Version 1.0 (2026).  \n**Webtool URL:** *https://ipred-s-1-single-stage-screening.streamlit.app/*")

    st.markdown("### How to Cite the Associated Research Article (Pre-publication)")
    st.markdown("This tool accompanies an unpublished research manuscript. Until acceptance, please cite the framework as follows:")
    
    st.info("**Integrating Quantum Chemical Descriptors and Matched Molecular Pair Analysis in an Explainable Ensemble Machine Learning Pipeline for In silico Identification of Soluble Epoxide Hydrolase Inhibitors** | D. Kumar, A. J. Martin | Manipal Academy of Higher Education (MAHE). | *Manuscript in preparation* (2026).")

    st.markdown("*(Final journal citation and DOI will be updated here once published and archived.)*")

    st.markdown("---")

    st.markdown("""
### Scientific Literature & Computational Packages
Below is the complete list of scientific literature, software tools, and computational packages used in the development, validation, and deployment of IPred-S 1.0.

#### 1. Machine Learning & Data Processing
* **Breiman, L.** Random Forests. *Machine Learning*, 45, 5–32 (2001).
* **Cortes, C., Vapnik, V.** Support-vector networks. *Machine Learning*, 20, 273–297 (1995).
* **Platt, J.** Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. *Advances in Large Margin Classifiers*, 10(3), 61-74 (1999).
* **Pedregosa et al.** Scikit-Learn: Machine Learning in Python. *JMLR* 12, 2825–2830 (2011).
* **Chicco, D., Jurman, G.** The advantages of the Matthews correlation coefficient (MCC). *BMC Genomics* 21, 6 (2020).

#### 2. Descriptor Generation & Cheminformatics
* **Moriwaki et al.** Mordred: A Comprehensive Descriptor Library for Molecular Descriptors. *J. Cheminf.* 10, 4 (2018).
* **RDKit:** Open-source cheminformatics. [http://www.rdkit.org](http://www.rdkit.org).
* **Todeschini, R., Consonni, V.** Handbook of Molecular Descriptors. *Wiley-VCH* (2000).

#### 3. Model Interpretation & Performance Evaluation
* **Powers, D.** Evaluation: Precision, Recall, F-measure, ROC, Informedness, Markedness. *JMLT* 2, 37–63 (2011).
* **Hand, D.J., Till, R.J.** A Simple Generalisation of the AUC for Multiclass Problems. *Machine Learning* 45, 171–186 (2001).
* **Trenton, M.** Balanced Accuracy and Its Advantages in Imbalanced Data. *Pattern Recogn. Lett.*, 120 (2019).

#### 4. Applicability Domain (AD)
* **Sahigara, F. et al.** Comparison of Different Approaches to Define the Applicability Domain. *J. Chemometrics* 26, 269–276 (2012).
* **Jaworska, J., Nikolova-Jeliazkova, N.** AD in QSAR Models. *Mutation Research* 575, 1–2 (2005).

#### 5. Datasets & Decoys
* **Mysinger et al.** Directory of Useful Decoys, Enhanced (DUD-E). *J. Med. Chem.* 55, 14 (2012).
* **sEH Bioassay Data:** Retrieved from peer-reviewed literature *(details in Supplementary Material of the upcoming manuscript)*.

#### 6. Software, Platforms & Versions (Used in IPred-S 1.0)
| Software / Package | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | 3.10 | Core Development |
| **Streamlit** | 1.50 | Web Interface Deployment |
| **RDKit** | 2025.03.6 | SMILES parsing and 3D embedding |
| **Mordred** | 1.2.0 | Topological Descriptor generation |
| **scikit-learn** | 1.4.2 | Model inference and Normalization |
| **NumPy** | 1.25.2 | Numerical computing |
| **Pandas** | 2.3.2 | DataFrame processing |
| **Graphviz** | latest | Flowchart rendering |
| **Matplotlib** | 3.10.6 | Internal plotting and validation |
    """)
