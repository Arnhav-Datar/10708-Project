# 10708-Project: Language Information Conditioned Graph Generative GAN Model
By **Abishek Sridhar, Arnhav Datar, Chifan Lo**

This repository contains code for the LIC-GAN project that was done as a part of 10-708: Probabilistic Graphical Models course at CMU in Spring 2023.

## Overview
We propose a GAN that generates graph based from natural language description of desired properties.
We additionally come up with a random graph dataset providing list of adjacency matrices and properties dicts to train and evaluate our methods, details of which are mentioned below:
| Dataset | Size   |
|---------|--------|
| Train   | 50000  |
| Val     | 10000  |
| Test    | 500    |
We compare our GAN model to a baseline that we created by prompting GPT-3.5-Turbo to generate graphs with the textual descriptions.

## Setup
We used python 3.10 to develop this code, though we expect it will work without problems for python versions >= 3.6. You can install the necessary libraries by running:
```bash
pip install -r requirements.txt
```

## Data Generation
The folder [GraphGen](/GraphGen/) contains the code for generating the dataset with the desired properties and size. You can obtain a dataset similar to what we create using:
```bash
python gen.py
```

## LIC-GAN Experiment
The code for the LIC-GAN method is in [LicGan](/LicGan/). You can run the training[/testing] of our final proposed model by:
```bash
python main_gan.py --mode train[/test]
```
There are a lot more optional arguments that you can pass to the training, the details of which can be found in [args.py](/LicGan/args.py).
