# Impact of Model Quantization on Anomaly Detection in High-Dimensional NLP Embedding Spaces

## Project Overview

This research project investigates the **impact of model quantization on anomaly detection reliability** in high-dimensional NLP embedding spaces. The study focuses on phishing email detection using precomputed text embeddings under a one-class anomaly detection setting, where normal (ham) samples are used for training and phishing samples are treated as anomalies evaluated only at test time.

The central research question addresses whether reducing numeric precision (FP32 → FP16 / INT8 / INT4 / INT2) affects the reliability, calibration stability, ranking robustness, and interpretability of anomaly detection systems under deployment constraints.

## Motivation / Research Goal

Model quantization is widely adopted in industrial systems to reduce memory footprint, accelerate inference, and enable deployment on edge or resource-constrained hardware. However, anomaly detection systems are particularly sensitive to small numerical perturbations because they rely on:

- Precise reconstruction error magnitudes (Autoencoder)
- Accurate likelihood estimation (GMM)
- Threshold-based decision rules

Even minor perturbations in score distributions may shift anomaly rankings, break threshold calibration, change alert rates, and alter feature attribution patterns (explainability).

This project aims to answer the following research questions:

1. To what extent does quantization (FP32 → FP16 / INT8 / INT4 / INT2) degrade anomaly detection performance?
2. Does quantization alter score distributions and break threshold calibration?
3. How stable are anomaly rankings under reduced precision?
4. Which model layers are most sensitive to quantization?
5. Does quantization affect SHAP-based feature importance and interpretability stability?
6. Can Quantization-Aware Training (QAT) mitigate performance degradation compared to Post-Training Quantization (PTQ), particularly at INT8 and INT4 precision levels?
7. Does embedding dimensionality (1024 vs 384) influence quantization robustness in anomaly detection systems?

## Methodology

### Anomaly Detection Protocol

- **Training**: Uses only normal (ham) samples
- **Testing**: Uses both normal and phishing samples
- **Models**: Learn the distribution of normal data
- **Anomaly Detection**: Phishing emails are detected as deviations from the learned distribution

This reflects a realistic anomaly detection setting where anomaly labels are not available during training.

### Deployment Constraint: Fixed FPR = 5%

To simulate a realistic industrial deployment scenario, we impose a Fixed False Positive Rate (FPR) of **5%** on normal (ham) training data.

The anomaly detection threshold is defined as the **95th percentile of anomaly scores** computed on ham samples:

$$
\tau = quantile(s_{ham}, 0.95)
$$

where:

- `s_ham` = anomaly scores for normal emails  
- `tau` = decision threshold  

Emails with scores greater than `tau` are flagged as anomalies.

## Models Implemented

### 1. Autoencoder (AE)

A fully connected symmetric encoder-decoder architecture that learns to reconstruct normal embeddings.

**Architecture:**
- **Encoder**: Linear(D → h₁) → ReLU → Linear(h₁ → h₂) → ReLU → Linear(h₂ → z)
- **Decoder**: Linear(z → h₂) → ReLU → Linear(h₂ → h₁) → ReLU → Linear(h₁ → D)
- **Latent dimension**: z = 64

**Configuration:**
- **BGE-1024**: h₁ = 512, h₂ = 256
- **MiniLM-384**: h₁ = 256, h₂ = 128

**Anomaly Score**: Reconstruction Mean Squared Error (MSE)
$$\text{score}(x) = \|x - \hat{x}\|_2^2$$

### 2. Gaussian Mixture Model (GMM)

A diagonal-covariance Gaussian Mixture Model that learns the distribution of normal embeddings.

**Configuration:**
- **Number of components**: K = 8
- **Covariance type**: Diagonal
- **Regularization**: `reg_covar = 1e-6`
- **Maximum iterations**: 300

**Anomaly Score**: Negative Log-Likelihood (NLL)
$$\text{score}(x) = -\log p(x)$$

## Dataset Description

### Source
The project uses a phishing email detection dataset containing:
- **Label 0**: Normal (ham) emails
- **Label 1**: Phishing/Spam emails (anomalies)

**Dataset Statistics:**
- **Training set (ham only)**: 37,808 samples
- **Test set (mixed)**: 79,978 samples
- **Phishing ratio (test)**: ~52.7%

### Embedding Representations

**Note**: Precomputed embedding files are **not included** in this repository due to their large size (~500MB+). Users must generate embeddings using the provided `convert_embeddings.ipynb` notebook.

Two types of precomputed, L2-normalized embeddings are used:

1. **BGE-large-en-v1.5** (dimension = 1024)
   - Generated using `BAAI/bge-large-en-v1.5`
   - Mean L2 norm ≈ 1.0

2. **all-MiniLM-L6-v2** (dimension = 384)
   - Generated using `sentence-transformers/all-MiniLM-L6-v2`
   - Mean L2 norm ≈ 1.0

All embeddings are:
- Precomputed and stored as NumPy arrays
- L2-normalized (mean L2 norm ≈ 1)
- Stored as `float32`

This allows isolation of numerical precision effects from text preprocessing variability.

## Experiments and Evaluation

### Quantization Variants

The project evaluates the following precision variants:

- **FP32** (reference baseline)
- **FP16** (half precision)
- **INT8** (dynamic quantization)
- **INT4** (simulated symmetric quantization)
- **INT2** (simulated symmetric quantization)

### Quantization Methods

#### FP16 (Half Precision)
- Applied directly to Autoencoder weights
- Tests mild precision reduction
- Commonly used in GPU acceleration

#### INT8 (Dynamic Quantization)
- Applied to `nn.Linear` layers
- Weights stored as INT8
- Activations quantized dynamically at runtime
- Simulates realistic production deployment

#### INT4 and INT2 (Simulated Quantization)
- Symmetric weight-only quantization
- Quantize weights to low-bit integers, then dequantize back to float
- Performs standard forward pass
- Isolates numerical precision loss while preserving computational stability

### Evaluation Metrics

For each precision variant, the following metrics are computed:

- **ROC-AUC**: Area under the ROC curve
- **PR-AUC**: Area under the Precision-Recall curve
- **F1 Score**: Harmonic mean of precision and recall
- **Spearman Correlation**: Rank correlation vs FP32 scores
- **Decision Flip Rate**: Fraction of samples with changed binary decisions
- **Top-5% Overlap**: Fraction of top anomalies preserved under quantization
- **Score Drift**: Mean absolute difference in anomaly scores

### Additional Analyses

1. **Layer Sensitivity Analysis**: Quantizes one Autoencoder layer at a time to identify critical layers
2. **Ranking Stability Analysis**: Measures preservation of anomaly ordering
3. **SHAP Explainability Analysis**: Evaluates feature attribution stability under quantization
4. **QAT vs PTQ Comparison**: Compares Quantization-Aware Training with Post-Training Quantization

## Results Summary

### Performance Degradation

- **FP16 and INT8**: Preserve anomaly detection performance almost perfectly for both AE and GMM models
- **INT4**: Introduces structured degradation but remains partially usable after recalibration
- **INT2**: Leads to structural collapse of anomaly ordering and detection reliability

### Calibration Stability

- Quantization can shift score distributions and break fixed-FPR calibration when FP32 threshold is reused
- Recalibration largely restores performance for INT4, indicating degradation is primarily due to calibration drift
- INT2 remains unstable even after recalibration

### Ranking Robustness

- **FP16 and INT8**: Preserve anomaly ordering (Spearman > 0.99)
- **INT4**: Introduces noticeable but structured ranking distortion (Spearman ~0.97)
- **INT2**: Results in near-random ordering (Spearman < 0.7)

### Layer Sensitivity

- Quantization sensitivity is not uniformly distributed
- Deeper encoder and early decoder layers exhibit slightly higher sensitivity under INT4
- Overall degradation remains limited at 4-bit precision

### Interpretability Stability

- **INT8**: Preserves feature attribution ranking (Top-10 overlap: 80%)
- **INT4**: Significantly disrupts top feature consistency (Top-10 overlap: 10%)
- Explanation stability degrades earlier than detection performance

### PTQ vs QAT

- **INT8**: QAT provides no meaningful benefit; PTQ already preserves ranking stability
- **INT4**: QAT substantially improves ranking consistency and reduces decision instability compared to PTQ

### Embedding Dimensionality Effects

- Higher-dimensional embeddings (BGE-1024) demonstrate greater robustness to quantization noise
- Increased representational capacity provides structural resistance to precision-induced perturbations

### Compression vs Performance Trade-Off

- **FP16**: 2× compression with no performance loss
- **INT8**: 4× compression with negligible ROC-AUC degradation
- **INT4**: 8× compression with noticeable but manageable degradation
- **INT2**: 16× compression with performance collapse

## Project Structure

```
.
├── main_notebook.ipynb          # Main experimental notebook
├── convert_embeddings.ipynb      # Embedding generation notebook
├── slides explanation.pdf        # Presentation slides
├── phishing_bge_1024_train.npy   # BGE-1024 training embeddings
├── phishing_bge_1024_test.npy    # BGE-1024 test embeddings
├── phishing_bge_1024_y_test.npy # BGE-1024 test labels
├── phishing_minilm_384_train.npy # MiniLM-384 training embeddings
├── phishing_minilm_384_test.npy  # MiniLM-384 test embeddings
├── phishing_minilm_384_y_test.npy # MiniLM-384 test labels
└── README.md                      # This file
```

## Installation Instructions

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- Jupyter Notebook or JupyterLab

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Impact-of-Model-Quantization-on-Anomaly-Detection-in-High-Dimensional-NLP-Embedding-Spaces-main
```

2. Create a virtual environment (recommended):
```bash
conda create -n quantization_env python=3.9
conda activate quantization_env
```

3. Install required packages:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn scipy matplotlib shap sentence-transformers ipython jupyter
```

## How to Run the Notebook

1. **Start Jupyter Notebook**:
```bash
jupyter notebook
```

2. **Open `main_notebook.ipynb`** in your browser

3. **Ensure data files are present**:
   - The notebook expects embedding files (`.npy`) in the same directory
   - If embeddings are not precomputed, run `convert_embeddings.ipynb` first

4. **Run cells sequentially**:
   - The notebook is organized into sections that should be executed in order
   - Some cells may take significant time (especially training and SHAP computation)
   - Results are cached where possible to avoid recomputation

5. **Key sections**:
   - **Section 1**: Environment setup and imports
   - **Section 2**: Dataset loading and visualization
   - **Section 3**: FP32 baseline model training
   - **Section 4**: Quantization experiments
   - **Section 5**: Layer sensitivity analysis
   - **Section 6**: Ranking stability analysis
   - **Section 7**: SHAP explainability analysis
   - **Section 8**: QAT experiments
   - **Section 9**: Results summary and conclusions

## Dependencies

### Core Libraries
- **PyTorch** (>=1.11.0): Model implementation and quantization
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **scikit-learn**: GMM implementation and evaluation metrics
- **SciPy**: Statistical functions

### Visualization
- **Matplotlib**: Plotting and visualization

### Explainability
- **SHAP**: Feature attribution analysis

### Embedding Generation (if needed)
- **sentence-transformers**: Text embedding generation

### Development
- **Jupyter/IPython**: Interactive notebook environment

## Key Findings

1. **INT8 provides optimal balance**: 4× compression with negligible performance loss
2. **INT4 approaches structural limits**: Requires recalibration or QAT for stability
3. **INT2 is unsuitable**: Leads to structural collapse of anomaly detection
4. **Higher dimensionality improves robustness**: BGE-1024 more robust than MiniLM-384
5. **Interpretability degrades earlier**: SHAP stability more sensitive than detection performance
6. **QAT beneficial for INT4**: Training-time adaptation significantly improves INT4 stability

## Future Work

1. **Extended embedding models**: Evaluate quantization robustness across more embedding architectures
2. **Mixed-precision strategies**: Investigate layer-specific precision allocation
3. **Alternative quantization methods**: Explore per-channel quantization, non-uniform quantization
4. **Real hardware deployment**: Benchmark quantized models on edge devices
5. **Other anomaly detection methods**: Extend analysis to isolation forests, one-class SVMs
6. **Multi-class anomaly detection**: Investigate quantization effects in multi-class settings
7. **Online learning scenarios**: Evaluate quantization stability under distribution shift
8. **Interpretability methods**: Compare SHAP with other explanation techniques under quantization

## Author

This research project was conducted as part of an investigation into the robustness of quantized anomaly detection systems in high-dimensional NLP embedding spaces.

## References

### Dataset
- Al-Subaiey, A., et al. (2024). Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection. ArXiv.org. https://arxiv.org/abs/2405.11619
- Phishing Email Dataset: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

### Key Papers
- Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey. ACM Computing Surveys.
- Ruff, L., et al. (2021). A Unifying Review of Deep and Shallow Anomaly Detection.
- Jacob, B., et al. (2018). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.
- Nagel, M., et al. (2021). A White Paper on Neural Network Quantization.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.

### Documentation
- PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
- SHAP Documentation: https://shap.readthedocs.io/
- BGE Model: https://huggingface.co/BAAI/bge-large-en-v1.5
- MiniLM Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

---

**Note**: This project focuses on numerical precision effects rather than hardware benchmarking. Simulated quantization (INT4/INT2) isolates precision loss while maintaining computational stability for controlled analysis.




