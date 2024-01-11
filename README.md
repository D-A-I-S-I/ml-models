# ML-models

Repository dedicated to ML model training and evaluation.

## Table of Contents
- [System Calls Model](#system-calls-model)
    - [Dependencies](#syscalls-dependencies)
    - [Install Requirements](#install-requirements)
    - [Running Grid Search](#running-grid-search)
    - [Training Data](#syscalls-training-data)
    - [Training Data Limitations](#training-data-limitations)
- [Understanding the Autoencoder Model](#understanding-the-autoencoder-model)
- [Network Model](#network-model)
    - [Dependencies](#network-dependencies)
    - [Training Data](#network-training-data)
- [Understanding the Network Model](#understanding-the-network-model)
- [License and Acknowledgements](#license-and-acknowledgements)


## System Calls Model

**Dependencies:**
- [torch](https://pytorch.org/)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)


`grid_search.py` is used to start the parallelized grid search for hyperparameter optimization. The search space for hyperparameters can be altered from that file. 

Install Requirements (with pip):

```python
pip install -r requirements.txt
```

Run grid search:

```python
python grid_search.py
```

For visualized training results, see `grid_search_notebook.ipynb`


### Training Data

Our model used to classify syscall sequences as malicious or normal was trained using the ADFA-LD Dataset:

https://cloudstor.aarnet.edu.au/plus/s/Gpr0FLAGGZZ1TL8/download

Sample from individual input text file:
``265 4 168 168 3 168 265 265 3`` (Syscall IDs)

### Training Data Limitations
The training data's comprehensiveness is key in this context. We used the ADFA-LD dataset, which has 174 unique syscall numbers. However, many Linux kernels have more than twice this number. This gap means that syscall sequences with syscalls not covered in the training might be wrongly identified as intrusions, simply because the model hasn't seen them before, leading to false positives. It's important to have training data that covers a wider range of syscalls typical in Linux systems to reduce this issue.

## Understanding the Autoencoder Model

The Autoencoder model in `Autoencoder.py`, tailored for system call sequence processing, integrates an embedding layer and an encoder-decoder architecture to detect intrusions. It starts by converting system calls into dense vectors through an embedding layer. The encoder then compresses this embedded input into a lower-dimensional space, and then the decoder reconstructs the original input from its compressed representation. In the forward pass, the model embeds, reshapes, encodes, and decodes the input, with the final output reshaped to match the original format. This approach allows the model to accurately detect anomalies by comparing the reconstruction error (loss) against normal system call patterns, identifying significant deviations that may indicate intrusions. The intended scenario is that the loss associated with a malicious syscall sequence is significantly higher than that of a normal sequence. This substantial difference allows for the establishment of a threshold with a considerable margin of safety, enabling reliable classification of sequences as either normal or potentially intrusive.


## Network Model

**Dependencies:**
- [sklearn](https://scikit-learn.org/)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)

Before running, specify in the file (Deepfence_ML_flowmeter.py) which benign and malicious data to train the model on (set correct paths).



### Training Data

The model was trained on data acquired from Deepfence.

Network dataset download:

(Malicious)
https://deepfence-public.s3.amazonaws.com/pcap-datasets/webgoat.pcap

(Normal)
https://deepfence-public.s3.amazonaws.com/pcap-datasets/benign_2017-05-02_kali-normal22.pcap

## Understanding the Network Model
The network model is a logistic regressor. The model takes in labeled flow data which includes features that can be seen in `Deepfence_ML_flowmeter.py`. Each feature is associated with a weight, and the model includes a bias term. The logistic regressor calculates a linear combination of the input features, weights, and bias. The linear combination is then passed through a logistic function, which transforms the linear combination into a value between 0 and 1, representing the probability of the instance belonging to the positive class (in our case, malicious). A threshold (e.g. 0.5) is set, and if the predicted probability is above this threshold, the instance is classified as belonging to the positive class; otherwise, it is classified as belonging to the negative class (benign). During the training phase, the model's parameters (weights and bias) are optimized. The goal is to adjust the parameters to minimize the difference between the predicted probabilities and the actual class labels in the training data.

### License and Acknowledgements


Network model code and datasets both taken from [Deepfence/FlowMeter](https://github.com/deepfence/FlowMeter) (under Apache 2.0 License)


Thus this repository is offered under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.txt)

The file `Deepfence_ML_flowmeter.py` is taken directly from the above mentioned github. Changes that have been made to this file are as follows:
- Certain imports have been removed (commented out).
- Code to save the model has been added
- Code to test the model has been added


