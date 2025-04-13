# Task-Circuit Quantization (TCQ) Demonstration

## Overview

I've created an interactive application that demonstrates the Task-Circuit Quantization (TCQ) technique described in the paper "Task-Circuit Quantization: Leveraging Knowledge Localization and Interpretability for Compression" by Hanqi Xiao, Yi-Lin Sung, Elias Stengel-Eskin, and Mohit Bansal from UNC Chapel Hill.

## What I've Built

1. **Interactive Web Application**: A Streamlit-based web application that demonstrates TCQ in real-time with two AI agents:
   - **TCQAG-1**: The primary agent that delegates tasks and explains the TCQ process
   - **TCQAG-2**: The secondary agent that performs the actual TCQ demonstration

2. **Paper Analysis**: A comprehensive summary of the TCQ paper, including:
   - Key findings and methodology
   - Performance improvements over baseline methods
   - Practical applications of TCQ

3. **Visual Demonstrations**: Interactive visualizations showing:
   - Weight distributions before and after quantization
   - Performance comparisons between TCQ and other methods
   - Memory savings achieved through TCQ

## Key Features of the Application

- **Paper Summary Tab**: Provides a detailed overview of the TCQ paper
- **TCQ Explanation Tab**: TCQAG-1 explains the TCQ process with visualizations
- **Live Demonstration Tab**: TCQAG-1 delegates tasks to TCQAG-2, who performs the demonstration
- **Results Comparison Tab**: Compares TCQ performance with other quantization methods

## How to Run the Application

1. Make sure you have Python installed on your system
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py` or use the provided `run_app.bat` file
4. The application will open in your default web browser at http://localhost:8501

## Key Takeaways from the TCQ Paper

1. **Task-Specific Weight Importance**: Not all weights in an LLM are equally important for a given task. TCQ identifies and preserves task-specific weights that are critical for performance.

2. **Saliency Metric Components**: TCQ uses two key components:
   - **Quantization-aware Localization (QAL)**: Traces how model performance is impacted by estimating the expected change in weights due to quantization
   - **Magnitude-sharpened Gradient (MSG)**: A generalized metric for the absolute importance of each weight

3. **Significant Performance Improvements**: TCQ shows major improvements over existing approaches, especially in ultra-low bit settings (2-3 bits):
   - In 2-bit settings: 16% improvement on GSM8k, 14.1% on MMLU, and 21.9% on Spider over SliM-LLM
   - In 3-bit settings: Preserves approximately 91%, 96%, and 89% of unquantized accuracy on GSM8k, MMLU, and Spider

4. **Task-Conditioning Benefits**: Conditioning on task-specific data significantly improves performance compared to using general datasets like WikiText2.

## Practical Applications

TCQ has several practical applications:

1. **Edge Deployment**: Enables deployment of large language models on resource-constrained devices like smartphones and IoT devices.

2. **Reduced Memory Footprint**: Reduces the memory requirements for serving large language models, allowing more models to be served on the same hardware.

3. **Task-Specific Optimization**: Allows models to be optimized for specific tasks, improving performance for targeted applications.

4. **Privacy-Sensitive Applications**: Enables local processing of sensitive data without sending it to the cloud.

## Additional Resources

- **README.md**: Contains detailed instructions on how to run the application
- **EXPLANATION.md**: Provides a more detailed explanation of the TCQ process
- **requirements.txt**: Lists all the dependencies required to run the application

## Next Steps

You can experiment with different configurations in the application:

1. **Target Bit Width**: Adjust the target bit-width for quantization (2-8 bits)
2. **Outlier Percentage**: Change the percentage of weights to preserve in 16-bit precision (0.1-2.0%)
3. **Task Type**: Select different task types to see how TCQ performs on various tasks

This demonstration provides a clear understanding of how TCQ works and its benefits for model compression while maintaining performance.
