# Time series forecasting of U.S. unemployment
Jay Lau, 12/2025

## Introduction

Unemployment is a key indicator of the health of the U.S. economy, so understanding and forecasting its trends is crucial. For example, predictions inform the federal government as to whether it needs to implement policies to take corrective action, such as adjusting interest rates.

Because large language models (LLMs) are popular for sequence prediction, this project investigated whether their architecture can be successfully applied to time series analysis. The project evaluated the performance of a foundation model—specifically, the Lag-Llama model—in forecasting unemployment levels. Its performance was benchmarked against two more basic architectures, long short-term memory (LSTM) and transformer models.

## Approach
The dataset used was produced by U.S. Bureau of Labor Statistics, which contained monthly unemployment rates from 1948 to present day.

The project was developed on Google Colab using a T4 GPU. The code was split across two Python Jupyter notebooks to accommodate different environment needs of the software. Three models were tested:
1. **LSTM model (TensorFlow/Keras):** the workhorse in the field of time series forecasting 
2. **Transformer model (TensorFlow/Keras):** based on self-attention mechanisms
3. **Lag-Llama foundation model (PyTorch):** applies the successful Llama LLM architecture (decoder-only transformer) to time series predictions. This open-source model performs zero-shot forecasting of single time series, taking a probabilistic approach that outputs both prediction values and intervals.

## Key findings
Of the three models, the LSTM and Lag-Llama models performed better than the basic transformer model. For the Lag-Llama model, fine-tuning its parameters was crucial to improve performance—specifically, how far ahead to forecast and how many past values to use for prediction. Surprisingly, even after optimizing Lag-Llama, the LSTM performed the best overall. Thus, the main takeaway of the project was that even though new deep learning techniques are being developed all the time, data scientists should not discount more established methods—in the case of time series forecasting, LSTM models.

## Future work
After figuring out how to install Lag-Llama, actually running the model was easy, with a single function call that did not require splitting the dataset in to training/validation/test sets. Another pro was the fast runtime. However, the original training corpus was small compared to LLMs. It also only works on a single time series at once, when in reality many are interrelated, for example unemployment with gross domestic product. In the future, it may be worth exploring other time series foundation models that include a larger training corpus and multivariate inputs.

## Code on Google Colab
1. [Notebook 1: LSTM and transformer models](https://colab.research.google.com/drive/1wjZVJUMhfkWbIifu5fMvk1vWCJPZjvgO?usp=sharing)
2. [Notebook 2: Lag-Llama foundation model](https://colab.research.google.com/drive/1QeRhcESdNrW2g2slhKp-yee5DRrvHHKo?usp=sharing)