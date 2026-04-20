
## Algorithmic Bias in AI Detection Tools

This project provides a reproducible Responsible AI pipeline to demonstrate and mitigate algorithmic bias in AI content identification systems towards non-native English speakers (ESL authors). 

### Prerequisites

**Core Runtime:** Python 3.10+
**Required Libraries:** `pandas`, `numpy`, `scikit-learn`, `NLTK`, `textstat`, `SHAP`, `LIME`, `matplotlib`, `seaborn`, `hashlib`

You can install the primary dependencies using pip:
```bash
pip install pandas numpy scikit-learn nltk textstat shap lime matplotlib seaborn
```
*Note: `hashlib` is included in the Python Standard Library.*

### Dataset and Code

The complete source code and the anonymized 300-essay benchmark dataset used in this study are publicly available. 

**Download from:** Google Drive Repository
**Link:** `https://drive.google.com/drive/folders/1z4D7QpbEVVz7-68shq6rl-F3BcaVxHp3?usp=sharing`

### System Pipeline & Execution Steps

The project follows a four-stage processing pipeline. Once you have downloaded the repository, you will run the code to execute the following stages:

1.**PII Redaction:** Process the raw essays through the `PIIDetector` class to scan and redact personal identifiable information using regex and SHA-256 fingerprinting.


2.**Bias Injection:** The system will simulate real-world ESL tool usage by replacing specific non-native essays with QuillBot-style paraphrased versions.


3.**Feature Engineering:** The code will extract six core linguistic features from the text, including Flesch Reading Ease, Burstiness, and Booster Word Frequency.


4.**Model Training & Fairness Audit:** A Logistic Regression baseline model is trained, and fairness metrics (like Demographic Parity and Equal Opportunity) are computed.


5.**Bias Mitigation:** The pipeline applies two strategies: pre-processing using TF-IDF feature diversification with 3x class re-weighting, and post-processing threshold calibration.


6.**Explainability Analysis:** Finally, run the modules for SHAP and LIME to generate global and local interpretations of the model's decisions. 

### Key Results

Applying the bias mitigation strategies reduces the False Positive Rate (FPR) gap between native and non-native writers from 58.0% to 0.0%.Overall model accuracy improves to 100.00%.
