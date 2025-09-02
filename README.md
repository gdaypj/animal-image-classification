# Image Classification

By developing and training a Convolutional Neuron Network Model using [Tensorflow](https://www.tensorflow.org/), this application is for classifying cats and dogs' pictures.

To play around with the app, here: [kitty-or-doggo](https://kitty-or-doggo.streamlit.app/)

## Data
The dataset used in this project is called [**The CIFAR-10 dataset**](https://www.cs.toronto.edu/~kriz/cifar.html) 

### Load/Access data using Tensorflow
```
  from tensorflow.keras.datasets import cifar10
```

## Set up environment

The python version in this project is python3.12
### Anaconda/ Conda Environment:
```
conda create env your-env python=3.12
```
To activate your env: 

```
 conda activate your env
```
### Or, Python Virtual Environment:
```
 python3.12 -m venv my_venv
``` 
To activate your virtual env:

```
source my_venv/bin/activate
```
## Install packages needed:

Inside your env:
```
pip install -r requirements.txt
```

## Deployment
[Streamlit framework](https://streamlit.io/) has been employed to deploy this application

### Run the app with Streamlit:
```
streamlit run app.py
```