# A Generative Deep Learning Approach for Alzheimer’s Disease Drug Discovery

## Table of Contents
- [Introduction](#introduction)
- [Model Overview](#model-overview)
- [Data and Evaluation](#data-and-evaluation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Conclusion and Future Work](#conclusion-and-future-work)

## Introduction

In the realm of drug discovery, especially for complex neurological conditions such as Alzheimer's disease, the journey from concept to viable treatment is notoriously prolonged, often extending beyond a decade. This slow pace is largely attributed to the overwhelming complexity of the chemical space, estimated at around 10^60 potential drug-like molecules. Traditional approaches often involve empirical selection from this vast space, a method that lacks efficiency and speed.

The advent of deep learning offers a transformative solution to this challenge. Leveraging its capacity to process and learn from large datasets, recent research advocates for the adoption of deep learning methodologies to hasten the drug discovery process. Notably, the integration of generative deep learning models, utilizing techniques akin to natural language processing (NLP), has shown promising potential in generating novel chemical structures. This innovative approach is poised to revolutionize how we identify and develop new compounds, particularly in the context of complex diseases.

Drawing inspiration from studies like those of Gupta et al. and Segler et al., which emphasize the effectiveness of LSTM RNNs in drug discovery and molecular generation, our project focuses on β-secretase 1 (BACE-1), a critical protein in Alzheimer’s disease progression. By integrating advanced LSTM and GRU neural networks, we aim to accelerate the generation of potential inhibitors for BACE-1, thereby contributing to the fight against Alzheimer's disease.


## Model Overview
I implemented three models from scratch:
- **Baseline Model (Model A)**: Consists of an LSTM layer and a dropout layer. It serves as a foundation for comparison and further model development.
- **Novel Model (Model B)**: A hybrid model combining GRU and LSTM layers to explore their efficacy in molecular generation.
- **Advanced Model (Model C)**: A complex LSTM-only model with enhanced depth.

<p align="center">
    <img src="assets/model_structs.png" width="500"/>
</p>

## Data and Evaluation
In my project, I utilized a two-pronged approach involving a generic chemical dataset and a focused dataset for Alzheimer's disease and BACE-1, both sourced from ChEMBL. The generic dataset, comprising 300,000 compounds, enabled the models to learn broad chemical syntax. In contrast, the focused dataset, containing 1,560 BACE-1 inhibitors, allowed for learning complex features specific to Alzheimer's disease. The datasets were preprocessed using the Keras tokenizer and featured a modified n-grams sequence processing for character prediction.

## Project Structure
The project encompasses several key components, each playing a vital role in the model's development and evaluation:

- **Baseline LSTM Code**: This part of the project, found in [Baseline Training Code](baseline_lstm.py), lays the foundation for more complex models and provides a point of comparison.
- **Improved LSTM Model**: The [Improved Training Code](improved_lstm.py) enhances the baseline model, adding depth and complexity to the LSTM layers.
- **Hybrid GRU/LSTM Model**: A novel approach combining GRU and LSTM layers for molecular generation, available in [Improved Training Code](hybrid_lstm.py).
- **Transfer Learning**: This process, detailed in [Code](TL_hybrid.py), adapts the models to Alzheimer's data for generating more specific chemical structures.
- **Evaluation**: The evaluation phase is split into two scripts - [Evaluate General](evaluateModelGeneralDataset.py) for the general dataset and [Evaluate Transfer Learn](evaluateModelsSpecificDataset.py) for the Alzheimer’s dataset.
- **Molecule Generation**: The [Molecule Generation](molecule_generator.py) script is responsible for creating new molecular structures based on the learned data.

The models underwent rigorous testing to evaluate their capability in learning chemical syntax and features. This included assessments on both the generic and Alzheimer's-specific datasets, gauging their effectiveness in real-world applications.


## Usage
Instructions on setting up the environment and running the models are provided in each script. Ensure you have the necessary dependencies installed as listed in `requirements.txt`.

## Methods

### Model Overview
In our quest to validate our hypothesis through rigorous testing, we developed three distinct models from scratch:

- **Baseline Model (Model A)**: This model is composed of a single LSTM layer with 32 units and a dropout layer with a probability of 0.2. Serving as our initial benchmark, it provides a fundamental basis for comparison and further development of complex models.
  
- **Novel Hybrid Model (Model B)**: Model B innovatively combines a series of GRU layers (128 units) with LSTM layers (512 units). This design is rooted in the potential of both GRU and LSTM layers in molecular generation, a synergy not extensively explored in prior research.
  
- **Advanced Model (Model C)**: Building upon the baseline, this model consists exclusively of LSTM layers with 512 units, forming a more complex structure with five LSTM layers. Each model, including this one, concludes with a Dense output layer of 43 units, reflecting the diverse output possibilities in our dataset.

### Model Training & Transfer Learning
Our approach to enhancing the models' learning capabilities involved the strategic use of transfer learning. Initially, the models were trained on a generic chemical dataset, followed by fine-tuning their weights using a focused dataset specific to Alzheimer’s disease. We adhered to a training regimen of 40 epochs, incorporating an early stopping mechanism with a patience of five epochs to preserve model accuracy and prevent overfitting.

### Hyperparameter Tuning
To fine-tune our models' hyperparameters, we employed Bayesian optimization—a method that utilizes information from previous trials to inform the selection of optimal parameters in subsequent trials. This method proved more efficient and effective than traditional random search techniques. Specifically, for Models B and C, we conducted Bayesian optimization over three epochs and three trials each, using the insights gained to guide our hyperparameter adjustments.


<p align="center">
    <img src="assets/test_alz.png" width="500"/>
</p>

## Results/Discussion

### Model Metrics
Our evaluation process focused on the models' performance in learning chemical syntax from both general and Alzheimer's-specific datasets:

- **General Dataset (Before Transfer Learning):**
  - Baseline Model (Model A): Achieved 65.5% accuracy with a loss of 1.193.
  - Hybrid Model (Model B): Recorded 76.7% accuracy and a loss of 0.788.
  - Improved Baseline Model (Model C): Attained 77.4% accuracy and a loss of 0.770.

These results underscored the baseline model as a solid foundation, upon which the hybrid and improved models built more sophisticated learning capabilities.

- **Alzheimer’s Disease Dataset (After Transfer Learning):**
  - Baseline Model (Model A): Showed an accuracy of 56.0% and a loss of 1.476.
  - Hybrid Model (Model B): Improved to an accuracy of 88.0% with a loss of 0.468.
  - Improved Baseline Model (Model C): Reached the highest accuracy of 88.1% and a loss of 0.473.

The post-transfer learning phase marked a significant enhancement, particularly in the hybrid and improved models, indicating their effectiveness in learning specific chemical features of Alzheimer’s disease.

### Analysis and Adjustments
Our initial analysis revealed a high bias in the baseline model, prompting the need for a larger network. This led to the development of Model C with its five LSTM layers, resulting in a remarkable training accuracy of 94.2% and testing accuracy of 88.1%, a substantial improvement from the baseline model's 56.0% testing accuracy.

### Molecule Generation
Using Model C, we successfully generated 18 new molecules, drawing from the Alzheimer’s dataset. Despite the molecules being slightly smaller than the reference due to the presence of a coded stop character, we demonstrated the model's capability to create valid chemical structures from molecular fragments.

### Conclusion
The hybrid model, combining GRU and LSTM layers, proved to be not only accurate but also computationally efficient. This approach, diverging from traditional models that solely employed either LSTM or GRU, showcased the benefits of integrating both types of neural networks. The results from our project provide a promising direction for future research in drug discovery, especially in areas requiring the generation of novel molecular structures.

---

<p align="center">
    <br><img src="assets/generated_molecules.png" width="500"/>
</p>


