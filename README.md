# Task-Circuit Quantization (TCQ) Demonstration

This application demonstrates the Task-Circuit Quantization technique described in the paper "Task-Circuit Quantization: Leveraging Knowledge Localization and Interpretability for Compression" by Hanqi Xiao, Yi-Lin Sung, Elias Stengel-Eskin, and Mohit Bansal.

![TCQ-Interface Gen-Purpose](https://github.com/user-attachments/assets/1f87b39d-4dce-4022-a9a5-578aed4cf870)
![TCQ-Interface Text-to-SQL](https://github.com/user-attachments/assets/cbbdcb5e-217c-453d-ba79-f45e73a213c7)
![TCQ-Interface Math Reasoning](https://github.com/user-attachments/assets/59211a6c-1e39-440e-a67f-3619d4949c70)

# Overview

Task-Circuit Quantization (TCQ) is a novel mixed-precision post-training quantization approach that identifies and preserves critical weights in a neural network while quantizing less important ones to lower bit-widths. This demonstration uses two AI agents to explain and showcase the TCQ process:

- **TCQAG-1**: The primary agent that delegates tasks and explains the TCQ process
- **TCQAG-2**: The secondary agent that performs the actual TCQ demonstration

## Features

- Interactive explanation of the TCQ process
- Live demonstration of how TCQ identifies and preserves important weights
- Visualization of weight distributions before and after quantization
- Performance comparison between TCQ and other quantization methods
- Memory savings analysis

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/tcq-demonstration.git
cd tcq-demonstration
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Set up your API keys:
   - Create a `.env` file in the root directory
   - Add your OpenAI API key: `OPENAI_API_KEY=your_openai_api_key_here`

## Usage

Run the Streamlit application:
```
streamlit run app.py
```

The application will open in your default web browser. You can:

1. Adjust the configuration parameters in the sidebar:
   - Target bit width
   - Outlier percentage
   - Task type

2. Explore the different tabs:
   - **TCQ Explanation**: Learn about the TCQ process and see visualizations of weight distributions
   - **Live Demonstration**: Watch TCQAG-1 delegate tasks to TCQAG-2 and see the demonstration results
   - **Results Comparison**: Compare the performance of TCQ with other quantization methods

## Key Concepts

- **Saliency Metric**: TCQ uses a saliency metric composed of two components:
  - **Quantization-aware Localization (QAL)**: Traces how model performance is impacted by estimating the expected change in weights due to quantization
  - **Magnitude-sharpened Gradient (MSG)**: A generalized metric for the absolute importance of each weight

- **Mixed-Precision Quantization**: TCQ preserves a small percentage of critical weights in 16-bit precision while quantizing the rest to lower bit-widths (2-3 bits)

- **Task-Specific Conditioning**: TCQ can be conditioned on specific tasks to further improve performance

## Paper Abstract

Post-training quantization (PTQ) reduces a model's memory footprint by mapping full precision weights into low bit weights without costly retraining, but can degrade its downstream performance especially in low 2- to 3-bit settings. We develop a new mixed-precision PTQ approach, Task-Circuit Quantization (TACQ), that draws parallels to automated circuit discovery, directly conditioning the quantization process on specific weight circuits – which we define as sets of weights associated with downstream task performance. These weights are kept as 16-bit weights, while others are quantized, maintaining performance while only adding a marginal memory cost.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The original TCQ paper authors: Hanqi Xiao, Yi-Lin Sung, Elias Stengel-Eskin, Mohit Bansal
- UNC Chapel Hill for supporting this research
