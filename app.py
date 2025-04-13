import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tcq_agents import TCQAG1, TCQAG2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Task-Circuit Quantization Demo",
    page_icon="🧠",
    layout="wide"
)

# App title and description
st.title("Task-Circuit Quantization (TCQ) Real-Time Demonstration")
st.markdown("""
This application demonstrates the Task-Circuit Quantization technique described in the paper
"Task-Circuit Quantization: Leveraging Knowledge Localization and Interpretability for Compression".

Two AI agents will work together to demonstrate how TCQ identifies and preserves important weights
while quantizing less important ones to achieve efficient model compression.

- **TCQAG-1**: The primary agent that delegates tasks and explains the TCQ process
- **TCQAG-2**: The secondary agent that performs the actual TCQ demonstration
""")

# Initialize agents
tcqag1 = TCQAG1()
tcqag2 = TCQAG2()

# Sidebar for configuration
st.sidebar.title("Configuration")
bit_width = st.sidebar.slider("Target Bit Width", min_value=2, max_value=8, value=3, step=1)
outlier_percentage = st.sidebar.slider("Outlier Percentage", min_value=0.1, max_value=2.0, value=0.35, step=0.05)
task_type = st.sidebar.selectbox(
    "Task Type",
    ["Question Answering", "Math Reasoning", "Text-to-SQL", "General Purpose"]
)

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Paper Summary", "TCQ Explanation", "Live Demonstration", "Results Comparison"])

with tab1:
    st.header("Task-Circuit Quantization Paper Summary")

    st.markdown("""
    ## Paper: Task-Circuit Quantization: Leveraging Knowledge Localization and Interpretability for Compression

    **Authors**: Hanqi Xiao, Yi-Lin Sung, Elias Stengel-Eskin, Mohit Bansal (UNC Chapel Hill)

    ### Abstract

    Post-training quantization (PTQ) reduces a model's memory footprint by mapping full precision weights into low bit weights without costly retraining, but can degrade its downstream performance especially in low 2- to 3-bit settings. We develop a new mixed-precision PTQ approach, Task-Circuit Quantization (TACQ), that draws parallels to automated circuit discovery, directly conditioning the quantization process on specific weight circuits – which we define as sets of weights associated with downstream task performance. These weights are kept as 16-bit weights, while others are quantized, maintaining performance while only adding a marginal memory cost.

    ### Key Findings

    1. **Not All Weights Are Equal**: The paper demonstrates that not all weights in a large language model (LLM) are equally important for a given task. Some weights form "circuits" that are critical for specific tasks.

    2. **Saliency Metric**: TCQ uses a novel saliency metric composed of two components:
       - **Quantization-aware Localization (QAL)**: Estimates how quantization will affect each weight and its impact on model performance
       - **Magnitude-sharpened Gradient (MSG)**: Measures the general importance of each weight

    3. **Performance Improvements**: TCQ significantly outperforms existing quantization methods, especially in ultra-low bit settings (2-3 bits):
       - In 2-bit settings: Up to 21.9% improvement over the strongest baseline
       - In 3-bit settings: Preserves up to 96% of unquantized accuracy

    4. **Task-Specific Benefits**: Conditioning on task-specific data further improves performance compared to using general datasets.

    ### Methodology

    The TCQ process involves:

    1. **Weight Importance Calculation**: Calculate the saliency score for each weight using the formula: `TACQ(Wij) = |Wij| · ∂L/∂Wij · |Q(Wij) - Wij|`

    2. **Outlier Identification**: Identify the top p% (typically 0.35%) of weights with the highest saliency scores

    3. **Mixed-Precision Quantization**: Preserve the outlier weights in 16-bit precision while quantizing the remaining weights to the target bit-width (2-3 bits)

    ### Results

    TCQ shows major improvements over existing approaches, especially in ultra-low bit settings (2-3 bits):

    - In 2-bit settings: 16% improvement on GSM8k, 14.1% on MMLU, and 21.9% on Spider over SliM-LLM
    - In 3-bit settings: Preserves approximately 91%, 96%, and 89% of unquantized accuracy on GSM8k, MMLU, and Spider

    ### Practical Applications

    TCQ has several practical applications:

    1. **Edge Deployment**: Enables deployment of large language models on resource-constrained devices
    2. **Reduced Memory Footprint**: Reduces the memory requirements for serving large language models
    3. **Task-Specific Optimization**: Allows models to be optimized for specific tasks
    4. **Privacy-Sensitive Applications**: Enables local processing of sensitive data without sending it to the cloud
    """)

with tab2:
    st.header("Understanding Task-Circuit Quantization")

    # TCQAG-1 explains TCQ
    explanation = tcqag1.explain_tcq(bit_width, outlier_percentage)
    st.markdown(explanation)

    # Visualization of the TCQ process
    st.subheader("TCQ Process Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Weight Distribution Before Quantization")
        # Generate sample data for visualization
        np.random.seed(42)
        weights = np.random.normal(0, 1, 1000)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(weights, bins=50, alpha=0.7)
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.markdown("### Weight Distribution After TCQ")
        # Simulate TCQ effect
        quantized_weights = np.copy(weights)

        # Identify top outlier_percentage% of weights by absolute value
        threshold = np.percentile(np.abs(weights), 100 - outlier_percentage)
        outlier_mask = np.abs(weights) >= threshold

        # Quantize non-outlier weights to 2^bit_width levels
        non_outliers = ~outlier_mask
        levels = 2**bit_width
        min_val, max_val = weights[non_outliers].min(), weights[non_outliers].max()
        step = (max_val - min_val) / (levels - 1)

        for i in range(levels):
            level_val = min_val + i * step
            mask = non_outliers & (weights >= (level_val - step/2)) & (weights <= (level_val + step/2))
            quantized_weights[mask] = level_val

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(quantized_weights, bins=50, alpha=0.7)
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

with tab3:
    st.header("Live TCQ Demonstration")

    if st.button("Start Demonstration"):
        with st.spinner("TCQAG-1 is delegating the task to TCQAG-2..."):
            delegation = tcqag1.delegate_task(bit_width, outlier_percentage, task_type)
            st.markdown("### TCQAG-1 (Delegator)")
            st.markdown(delegation)

        with st.spinner("TCQAG-2 is performing the TCQ demonstration..."):
            demo_result = tcqag2.perform_demonstration(bit_width, outlier_percentage, task_type)
            st.markdown("### TCQAG-2 (Executor)")
            st.markdown(demo_result)

            # Visualize the demonstration results
            st.subheader("Demonstration Results")

            # Sample data for visualization
            original_accuracy = 100
            if task_type == "Question Answering":
                baseline_accuracies = {
                    "Uniform Quantization": original_accuracy * (0.4 + 0.05 * bit_width),
                    "GPTQ": original_accuracy * (0.45 + 0.07 * bit_width),
                    "SqueezeLLM": original_accuracy * (0.5 + 0.07 * bit_width),
                    "SPQR": original_accuracy * (0.48 + 0.08 * bit_width),
                    "SliM-LLM": original_accuracy * (0.53 + 0.08 * bit_width),
                }
            elif task_type == "Math Reasoning":
                baseline_accuracies = {
                    "Uniform Quantization": original_accuracy * (0.35 + 0.05 * bit_width),
                    "GPTQ": original_accuracy * (0.4 + 0.07 * bit_width),
                    "SqueezeLLM": original_accuracy * (0.45 + 0.07 * bit_width),
                    "SPQR": original_accuracy * (0.43 + 0.08 * bit_width),
                    "SliM-LLM": original_accuracy * (0.48 + 0.08 * bit_width),
                }
            elif task_type == "Text-to-SQL":
                baseline_accuracies = {
                    "Uniform Quantization": original_accuracy * (0.3 + 0.05 * bit_width),
                    "GPTQ": original_accuracy * (0.35 + 0.07 * bit_width),
                    "SqueezeLLM": original_accuracy * (0.4 + 0.07 * bit_width),
                    "SPQR": original_accuracy * (0.38 + 0.08 * bit_width),
                    "SliM-LLM": original_accuracy * (0.43 + 0.08 * bit_width),
                }
            else:  # General Purpose
                baseline_accuracies = {
                    "Uniform Quantization": original_accuracy * (0.38 + 0.05 * bit_width),
                    "GPTQ": original_accuracy * (0.43 + 0.07 * bit_width),
                    "SqueezeLLM": original_accuracy * (0.48 + 0.07 * bit_width),
                    "SPQR": original_accuracy * (0.46 + 0.08 * bit_width),
                    "SliM-LLM": original_accuracy * (0.51 + 0.08 * bit_width),
                }

            # Add TCQ with a boost
            tacq_boost = 0.15 if bit_width <= 3 else 0.08
            baseline_accuracies["TCQ (Ours)"] = min(original_accuracy,
                                                  baseline_accuracies["SliM-LLM"] +
                                                  original_accuracy * tacq_boost)

            # Create bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            methods = list(baseline_accuracies.keys())
            accuracies = list(baseline_accuracies.values())

            # Color the bars
            colors = ['#A9A9A9', '#A9A9A9', '#A9A9A9', '#A9A9A9', '#A9A9A9', '#1E88E5']

            bars = ax.bar(methods, accuracies, color=colors)
            ax.set_xlabel('Method')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'Model Accuracy at {bit_width}-bit Quantization on {task_type} Task')
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom')

            st.pyplot(fig)

with tab4:
    st.header("Results Comparison")

    # Create comparison table
    st.subheader("Performance Comparison at Different Bit Widths")

    # Generate sample data for the table
    bit_widths = [2, 3, 4, 8]
    methods = ["Uniform Quantization", "GPTQ", "SqueezeLLM", "SPQR", "SliM-LLM", "TCQ (Ours)"]

    # Create sample data
    data = []
    for bit in bit_widths:
        row = {"Bit Width": bit}
        for method in methods:
            if method == "TCQ (Ours)":
                # TCQ performs better, especially at lower bit widths
                if bit == 2:
                    row[method] = f"{min(95, 40 + bit * 10 + 15)}%"
                elif bit == 3:
                    row[method] = f"{min(95, 40 + bit * 10 + 10)}%"
                else:
                    row[method] = f"{min(98, 40 + bit * 10 + 5)}%"
            else:
                # Other methods perform worse at lower bit widths
                base = 30 if method == "Uniform Quantization" else 40
                row[method] = f"{min(95, base + bit * 10)}%"
        data.append(row)

    st.table(data)

    # Memory savings visualization
    st.subheader("Memory Savings")

    # Calculate memory usage for different methods
    model_size_gb = 16  # Original model size in GB (16-bit)

    memory_usage = {
        "Original (16-bit)": model_size_gb,
        "8-bit": model_size_gb / 2,
        "4-bit": model_size_gb / 4,
        "3-bit": model_size_gb / 5.33,
        "2-bit": model_size_gb / 8,
        f"TCQ ({bit_width}-bit)": model_size_gb / (16 / (bit_width - outlier_percentage/100 * (16-bit_width)))
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    methods = list(memory_usage.keys())
    sizes = list(memory_usage.values())

    # Color the bars
    colors = ['#A9A9A9', '#A9A9A9', '#A9A9A9', '#A9A9A9', '#A9A9A9', '#1E88E5']

    bars = ax.bar(methods, sizes, color=colors)
    ax.set_xlabel('Method')
    ax.set_ylabel('Memory Usage (GB)')
    ax.set_title('Memory Usage Comparison')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f} GB', ha='center', va='bottom')

    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("**Task-Circuit Quantization Demo** | Based on research by Hanqi Xiao, Yi-Lin Sung, Elias Stengel-Eskin, Mohit Bansal")
