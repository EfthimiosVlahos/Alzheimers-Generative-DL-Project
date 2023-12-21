# A Generative Deep Learning Approach for Alzheimer’s Disease Drug Discovery

## Table of Contents
- [Introduction](#introduction)
- [Model Overview](#model-overview)
- [Data and Evaluation](#data-and-evaluation)
- [Project Structure](#project-structure)
- [Results](#results)

## Introduction
In the realm of drug discovery, especially for complex neurological conditions such as Alzheimer's disease, the journey from concept to viable treatment is notoriously prolonged, often extending beyond a decade. This slow pace is largely attributed to the overwhelming complexity of the chemical space, estimated at around 10^60 potential drug-like molecules. Traditional approaches often involve empirical selection from this vast space, a method that lacks efficiency and speed.

The advent of deep learning offers a transformative solution to this challenge. Leveraging its capacity to process and learn from large datasets, recent research advocates for the adoption of deep learning methodologies to hasten the drug discovery process. Notably, the integration of generative deep learning models, utilizing techniques akin to natural language processing (NLP), has shown promising potential in generating novel chemical structures. This innovative approach is poised to revolutionize how we identify and develop new compounds, particularly in the context of complex diseases.

Drawing inspiration from studies like those of Gupta et al. and Segler et al., which emphasize the effectiveness of LSTM RNNs in drug discovery and molecular generation, our project focuses on β-secretase 1 (BACE-1), a critical protein in Alzheimer’s disease progression. By integrating advanced LSTM and GRU neural networks, we aim to accelerate the generation of potential inhibitors for BACE-1, thereby contributing to the fight against Alzheimer's disease.

## Model Overview
We implemented three models from scratch:
- **Baseline Model (Model A)**: Consists of an LSTM layer and a dropout layer. It serves as a foundation for comparison and further model development.
- **Novel Model (Model B)**: A hybrid model combining GRU and LSTM layers to explore their efficacy in molecular generation.
- **Advanced Model (Model C)**: A complex LSTM-only model with enhanced depth.

<p align="center">
    <img src="assets/model_structs.png" width="500"/>
</p>

## Data and Evaluation
In my project, I utilized a two-pronged approach involving a generic chemical dataset and a focused dataset for Alzheimer's disease and BACE-1, both sourced from ChEMBL. The generic dataset, comprising 300,000 compounds, enabled the models to learn broad chemical syntax. In contrast, the focused dataset, containing 1,560 BACE-1 inhibitors, allowed for learning complex features specific to Alzheimer's disease. The datasets were preprocessed using the Keras tokenizer and featured a modified n-grams sequence processing for character prediction.

## Project Structure
- **Baseline LSTM Code**: [Baseline Training Code](baseline_lstm.py)
- **Improved LSTM Model**: [Improved Training Code](improved_lstm.py)
- **Hybrid GRU/LSTM Model**: [Improved Training Code](hybrid_lstm.py)
- **Transfer Learning**: Leveraging Alzheimer's data for specific chemical structure generation. [Code](TL_hybrid.py)
- **Evaluation**: Scripts for model evaluation. [Evaluate General](evaluateModelGeneralDataset.py) | [Evaluate Transfer Learn](evaluateModelsSpecificDataset.py)
- **Molecule Generation**: Script for generating new molecules. [Molecule Generation](molecule_generator.py)


## Results
Our models showed promising results in learning chemical structures and generating potential drug compounds:

- Baseline Model (Model A) showed an initial accuracy of 65.5%.
- Hybrid Model (Model B) achieved an accuracy of 88.0% after transfer learning.
- Advanced Model (Model C) showed the highest accuracy of 88.1%.

<p align="center">
    <img src="assets/test_alz.png" width="500"/>
</p>

### Analysis and Adjustments
Our initial analysis revealed a high bias in the baseline model, prompting the need for a larger network. This led to the development of Model C with its five LSTM layers, resulting in a remarkable training accuracy of 94.2% and testing accuracy of 88.1%, a substantial improvement from the baseline model's 56.0% testing accuracy.

### Molecule Generation
Using Model C, we successfully generated 18 new molecules, drawing from the Alzheimer’s dataset. Despite the molecules being slightly smaller than the reference due to the presence of a coded stop character, we demonstrated the model's capability to create valid chemical structures from molecular fragments.

<p align="center">
    <br><img src="assets/generated_molecules.png" width="500"/>
</p>

### Conclusion
The hybrid model, combining GRU and LSTM layers, proved to be not only accurate but also computationally efficient. This approach, diverging from traditional models that solely employed either LSTM or GRU, showcased the benefits of integrating both types of neural networks. The results from our project provide a promising direction for future research in drug discovery, especially in areas requiring the generation of novel molecular structures.


