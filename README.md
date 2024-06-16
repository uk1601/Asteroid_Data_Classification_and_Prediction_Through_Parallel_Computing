
# Asteroid Data Classification and Prediction Through Parallel Computing

## Introduction

This project aims to analyze a comprehensive dataset containing various attributes of known asteroids. By utilizing advanced machine learning algorithms, we classify different types of asteroids based on their properties and predict their potential trajectories and behavior. This research enhances our understanding of space objects and their likelihood of impacting Earth, aiding space agencies, researchers, and policymakers in devising effective planetary defense and risk mitigation strategies.

## Project Motivation

Asteroids, remnants of the early solar system, carry vital information about its formation and evolution. Understanding their characteristics and behavior is essential due to their potential threat to Earth. By leveraging advanced data analysis techniques, we aim to enhance our ability to predict and classify asteroid trajectories and properties, contributing to planetary defense and deepening our knowledge of the cosmos.

## Project Goal

The primary goal of this project is to develop a model for asteroid classification using parallel computing methods for pre-processing and model training. We aim to compare the speed-up performance achieved through parallel processing on various hardware configurations, enhancing the predictive accuracy and efficiency of asteroid classification models.

## Methodology

### 1. Data Preprocessing and Cleaning
- **Data Loading**: Utilized Dask Dataframe and Pandas to load the large dataset efficiently. Dask allows for parallel computation, enabling the handling of large datasets by distributing the load across multiple CPU cores.
- **Feature Extraction**: Implemented Dask dataframe to perform tasks like data extraction, cleaning, and normalization in parallel, improving efficiency and reducing computation time.
- **Missing Values Handling**: Employed Dask SimpleImputer for filling null values with the median strategy, ensuring data consistency and robustness for further analysis.
- **Feature Selection**: Used ExtraTreesClassifier for identifying and selecting the most informative features. This step was parallelized using joblib and multiprocessing to expedite the process.

### 2. Exploratory Data Analysis (EDA)
- **Statistical Overview**: Obtained statistical properties of the dataset using Dask's describe().compute() function, which provides insights into data distribution and potential skewness.
- **Handling Null Values**: Identified and filled missing values using the median strategy with Dask SimpleImputer, ensuring a robust dataset for model training.
- **Feature Importance Analysis**: Analyzed the relative importance of each feature using ExtraTreesClassifier, which helps prioritize the most informative features for the classification task.

### 3. Model Building
- **Machine Learning Models**: Implemented multiple machine learning algorithms, including Random Forests, KNN, XGBoost, and Feed Forward Neural Networks, to perform classification and prediction tasks.
  - **Random Forests**: Used Dask and multiprocessing to parallelize the training process, significantly reducing computation time.
  - **KNN (K-Nearest Neighbors)**: Parallelized using Dask and multiprocessing for faster execution.
  - **XGBoost**: Implemented using Dask XGBoost for efficient parallel training.
  - **Feed Forward Neural Networks**: Utilized PyTorch for neural network training, implementing data parallelism and distributed data parallelism to leverage multiple CPUs and GPUs.
- **Parallelization Techniques**: 
  - **Data Parallelism**: Distributed data across multiple CPUs/GPUs to perform parallel computations, significantly reducing training time.
  - **Distributed Data Parallelism**: Leveraged PyTorch's DDP module to distribute training across multiple GPUs, enhancing efficiency and reducing execution time.

### 4. Performance Evaluation
- **Execution Time Measurement**: Compared execution times for serial and parallel implementations to evaluate the efficiency gains from parallel processing.
- **Visualization**: Visualized performance gains using graphs to clearly present the efficiency improvements achieved through parallel processing.
- **Optimization**: Fine-tuned parallel implementations by optimizing task distribution and resource allocation, iterating based on performance insights to further enhance efficiency.

### 5. Dataset Description
- **Dataset Source**: NASA JPL and Kaggle.
- **Size and Features**: 958,524 records, 45 quantitative parameters including orbital parameters, physical properties, and more. This diverse dataset allows for detailed and nuanced modeling approaches.

## Results and Analysis

- **Hardware Configuration**: Utilized Northeastern University's Discovery supercomputer equipped with Nvidia V100-SXM2 GPUs (2 GPUs with 32 GB memory each) and 28 CPUs with 128 GB memory.
- **Performance Evaluation**: Observed significant speedups with parallel processing compared to serial processing. Optimal performance was achieved using a combination of CPUs and GPUs, demonstrating the effectiveness of parallelization techniques.

## Code Files Description

1. `SequentialExecution.ipynb`: Contains code for serial execution to establish baseline performance metrics.
2. `RFDask.ipynb`: Implements Random Forest using Dask for parallel processing.
3. `RFMP.ipynb`: Implements Random Forest using the multiprocessing library for parallel processing.
4. `KNNDask.ipynb`: Implements KNN using Dask for parallel processing.
5. `KNNMP.ipynb`: Implements KNN using the multiprocessing library for parallel processing.
6. `XGBoostParallel.ipynb`: Implements XGBoost using Dask XGBoost for parallel processing.
7. `NNDataParallel.ipynb`: Implements neural network training with data parallelism using PyTorch.
8. `DDPNeuralNetworkon1GPU.ipynb`: Implements Distributed Data Parallel on 1 GPU using PyTorch.
9. `DDPNeuralNetworkon2GPU.ipynb`: Implements Distributed Data Parallel on 2 GPUs using PyTorch.
10. `DDP Neural Network on CPU`: Executable script for running neural network training on CPU.
11. `DDP Neural Network on GPU`: Executable script for running neural network training on GPU.

## Conclusion

Parallelization markedly decreases execution times for machine learning models compared to serial execution, with more pronounced reductions as CPU cores increase or with more number of GPUs. Effective resource management and parallelization strategy selection are crucial for optimizing performance without excessive resource utilization. This project demonstrates the potential of parallel processing techniques in enhancing the efficiency of machine learning models for asteroid classification and prediction.

## References

1. Hossain, Mir Sakhawat, & Zabed, Md. (2023). Machine Learning Approaches for Classification and Diameter Prediction of Asteroids. dx.doi.org/10.1007/978-981-19-7528-8_4
2. Dhariwal, Kunal. "Pandas with Dask for an Ultra-Fast Notebook." Medium Towards Data Science, 31 Oct. 2021, https://towardsdatascience.com/pandas-with-dask-for-an-ultra-fast-notebook-e2621c3769f.
3. NASA. (2020 April). JPL Small-Body Database Search Engine. https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/
4. Dask Development Team. (n.d.). Dask DataFrame. Dask Documentation. https://docs.dask.org/en/stable/dataframe-create.html
5. PyTorch. (n.d.). Documentation. https://pytorch.org/docs/stable/index.html
6. Hu, L. (2020). Distributed parallel training: Data parallelism and model parallelism. Towards Data Science. https://towardsdatascience.com/distributed-parallel-training-data-parallelism-and-model-parallelism-ec2d234e3214
