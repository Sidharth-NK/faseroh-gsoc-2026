# faseroh-gsoc-2026

# Math to Taylor Series Translation (Seq2Seq)

This project formulates the mathematical operation of calculating Taylor series expansions as a machine translation problem. Given a source mathematical function (e.g., `sin(x) + x**2`), the models are trained to predict its 4th-order Taylor series expansion polynomial (e.g., `x**2 + x - x**3/6`).

The codebase implements, trains, and compares three distinct Sequence-to-Sequence (Seq2Seq) architectures in PyTorch to demonstrate the evolution of sequence modelling techniques and how they handle information bottlenecks.

## Dataset Generation
The dataset is dynamically generated using the `sympy` library to calculate exact polynomials with no floating-point errors. 
* **Total Samples:** 3,000 (Train: 2400 | Val: 300 | Test: 300)
* **Vocabulary:** Expressions are tokenized into atomic string units (e.g., `['sin', '(', 'x', ')']`).
* **Composition:** Base functions include `sin`, `cos`, `exp`, `log`, `tan`, `sinh`, `cosh`, `atan`, `inv`, and `sqrt`. These are randomly combined with scalars, polynomials, or other functions to create compound inputs.

---

## Models Implemented

### 1. Vanilla LSTM Seq2Seq (Baseline)
A classic Encoder-Decoder architecture. The encoder compresses the entire mathematical sequence into a single, fixed-size context vector. 
* **Limitation:** Struggles with the "information bottleneck," as seen in the validation loss diverging early (Epoch 4). It fails to predict exact matches for complex sequences.

### 2. LSTM with Bahdanau Attention
Introduces additive attention to the baseline model. Instead of a single context vector, the decoder computes a dynamic weighted sum over *all* encoder hidden states at each decoding step.
* **Advantage:** Directly eliminates the information bottleneck, allowing the model to "look back" at specific parts of the source equation when generating the Taylor polynomial.

### 3. Transformer Seq2Seq
Utilizes multi-head self-attention, processing all tokens in parallel rather than sequentially. 
* **Advantage:** Achieves the lowest validation loss, the fastest convergence, and the highest token-level accuracy due to superior long-range dependency modeling.

---

## Results & Performance

*Models were evaluated on a test set of 300 held-out pairs.*

| Metric | Vanilla LSTM | LSTM + Attention | Transformer |
| :--- | :--- | :--- | :--- |
| **Token Accuracy (%)** | 28.01 | 93.82 | **96.70** |
| **Exact Match (%)** | 0.0 | **78.67** | 76.00 |
| **BLEU-1 Score** | 48.44 | **96.78** | 96.24 |
| **Best Val Loss** | 2.4720 | 0.3832 | **0.0999** |
| **Best Epoch** | 4 | 63 | 42 |
| **Parameters** | 1,877,200 | 2,291,152 | 4,002,128 |

> **Note on Exact Match:** While the Transformer achieved better token accuracy and lower validation loss, the LSTM with Attention slightly outperformed it on the strict Exact Match metric for this specific test distribution.

---

## Visualizations Output
Running the script automatically generates six core evaluation plots:

* **`01_dataset_analysis.png`**: Distributions of base function types, token lengths, and sample pairs.
* **`02_training_curves.png`**: Train vs. Validation loss across epochs, illustrating the divergence of the Vanilla LSTM and the convergence of the Attention/Transformer models.
* **`03_metrics_comparison.png`**: Grouped bar charts comparing Token Accuracy, Exact Match, and BLEU-1 scores.
* **`04_sample_predictions.png`**: Side-by-side strings of actual model outputs vs. the target Taylor series for qualitative analysis.
* **`05_attention_heatmap.png`**: Transformer Layer 0 self-attention heatmaps, showing how the model learns to associate specific mathematical tokens.
* **`06_per_functype_accuracy.png`**: Exact-match performance broken down by the root function (e.g., `sin`, `sqrt`, `log`).

---

## Requirements & Usage

Install the required dependencies via `pip install -r requirements.txt`.

**Execution:**
Run the script directly to generate the dataset, train all three models, and output the visualizations and `results_summary.json`.

```bash
python main.py
