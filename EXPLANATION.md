# Task-Circuit Quantization (TCQ) Demonstration: Detailed Explanation

## Paper Summary

The paper "Task-Circuit Quantization: Leveraging Knowledge Localization and Interpretability for Compression" introduces a novel approach to model compression called Task-Circuit Quantization (TCQ). This technique identifies and preserves critical weights in a neural network while quantizing less important ones to lower bit-widths, resulting in significant memory savings with minimal performance degradation.

### Key Findings from the Paper:

1. **Not All Weights Are Equal**: The paper demonstrates that not all weights in a large language model (LLM) are equally important for a given task. Some weights form "circuits" that are critical for specific tasks.

2. **Saliency Metric**: TCQ uses a novel saliency metric composed of two components:
   - **Quantization-aware Localization (QAL)**: Estimates how quantization will affect each weight and its impact on model performance
   - **Magnitude-sharpened Gradient (MSG)**: Measures the general importance of each weight

3. **Performance Improvements**: TCQ significantly outperforms existing quantization methods, especially in ultra-low bit settings (2-3 bits):
   - In 2-bit settings: Up to 21.9% improvement over the strongest baseline
   - In 3-bit settings: Preserves up to 96% of unquantized accuracy

4. **Task-Specific Benefits**: Conditioning on task-specific data further improves performance compared to using general datasets.

## How This Demonstration Works

This application uses two AI agents to demonstrate the TCQ process:

### TCQAG-1 (Task-Circuit Quantization Agent 1)

TCQAG-1 is the primary agent that:
- Explains the TCQ process and its benefits
- Delegates tasks to TCQAG-2
- Provides context and guidance for the demonstration

### TCQAG-2 (Task-Circuit Quantization Agent 2)

TCQAG-2 is the secondary agent that:
- Performs the actual TCQ demonstration
- Shows step-by-step how TCQ identifies important weights
- Compares TCQ performance with other quantization methods
- Provides task-specific examples

## The TCQ Process Explained

1. **Weight Importance Calculation**:
   - Calculate the saliency score for each weight using the formula:
     `TACQ(Wij) = |Wij| · ∂L/∂Wij · |Q(Wij) - Wij|`
   - Where:
     - `|Wij|` is the magnitude of the weight
     - `∂L/∂Wij` is the gradient of the loss with respect to the weight
     - `|Q(Wij) - Wij|` is the expected change in the weight due to quantization

2. **Outlier Identification**:
   - Identify the top p% (typically 0.35%) of weights with the highest saliency scores
   - These weights are considered "outliers" and are critical for model performance

3. **Mixed-Precision Quantization**:
   - Preserve the outlier weights in 16-bit precision
   - Quantize the remaining weights to the target bit-width (2-3 bits)
   - This results in a model with an average bit-width slightly higher than the target

4. **Memory Savings**:
   - The resulting model requires significantly less memory than the original 16-bit model
   - For example, a 2-bit model with 0.35% outliers requires approximately 1/8 of the original memory

## Task-Specific Examples

### Question Answering (MMLU)

For question-answering tasks like MMLU (Massive Multitask Language Understanding), TCQ preserves weights that are critical for reasoning and knowledge retrieval. In 2-bit settings, TCQ achieves 49.19% accuracy compared to 34.84% for SliM-LLM (the strongest baseline).

### Math Reasoning (GSM8k)

For math reasoning tasks like GSM8k, TCQ preserves weights that are important for numerical operations and step-by-step reasoning. In 2-bit settings, TCQ achieves 36.11% accuracy compared to 20.12% for SliM-LLM.

### Text-to-SQL (Spider)

For text-to-SQL tasks like Spider, TCQ preserves weights that are critical for understanding database schemas and generating SQL queries. In 2-bit settings, TCQ achieves 21.92% accuracy while other methods degrade to near-zero performance.

## Why TCQ Works

TCQ works because it:

1. **Identifies Task-Specific Circuits**: By using gradient information and task-specific data, TCQ identifies weights that form circuits critical for specific tasks.

2. **Accounts for Quantization Effects**: Unlike other methods, TCQ explicitly considers how quantization will affect each weight and its impact on model performance.

3. **Combines Global and Local Information**: TCQ uses both global gradient information and local weight magnitude to identify important weights.

4. **Preserves Critical Weights**: By keeping a small percentage of critical weights in 16-bit precision, TCQ maintains model performance while significantly reducing memory requirements.

## Practical Applications

TCQ has several practical applications:

1. **Edge Deployment**: Enables deployment of large language models on resource-constrained devices like smartphones and IoT devices.

2. **Reduced Memory Footprint**: Reduces the memory requirements for serving large language models, allowing more models to be served on the same hardware.

3. **Task-Specific Optimization**: Allows models to be optimized for specific tasks, improving performance for targeted applications.

4. **Privacy-Sensitive Applications**: Enables local processing of sensitive data without sending it to the cloud.

## Try It Yourself

Experiment with different configurations in the sidebar:

1. **Target Bit Width**: Adjust the target bit-width for quantization (2-8 bits)
2. **Outlier Percentage**: Change the percentage of weights to preserve in 16-bit precision (0.1-2.0%)
3. **Task Type**: Select different task types to see how TCQ performs on various tasks

Then explore the different tabs to see the TCQ process in action and compare its performance with other quantization methods.
