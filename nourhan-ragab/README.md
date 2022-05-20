# Nourhan Ragab - Soccer Athlete Performance Prediction using Time Series Analysis

## Abstract

Regardless of the sport you prefer, your favorite athlete has almost certainly disappointed you at some point. 
Did you jump to a conclusion and dismissed it as "not their day"? 
Or, did you consider the underlying causes for their poor performance on that particular day? 
Under-performance can have big consequences in team sports such as soccer and affect the entire team dynamic. 
Basal needs like sleep quality and wellness parameters such as mood, fatigue and muscle soreness can affect an athlete's performance. 
In this context, the practice of using wearable sensor devices to quantify athlete health and performance is gaining popularity in sports science. 
This thesis aims to predict how ready a soccer athlete is to train or play a match based on the subjectively reported wellness parameter _readiness to play_, collected by the PMSys athlete performance monitoring system [[1](https://forzasys.com/pmSys.html), [2](https://doi.org/10.3389/fphys.2018.00866), [3](https://dl.acm.org/doi/10.1145/3395035.3425300)]. 
Even though women's soccer is receiving increasingly more attention, with a recent record in game day attendance [marking over 90.000 spectators](https://www.si.com/fannation/soccer/futbol/news/womens-soccer-attendance-record-broken-at-barcelona-vs-real-madrid), the vast majority of soccer studies are conducted on male athletes. 
In this sense, we explore a relatively new domain using the PMSys dataset, which is from two Norwegian elite female soccer clubs over the period of 2020 and 2021. 
We predict readiness by utilizing the long short-term memory (LSTM) method and the [Tsai](https://github.com/timeseriesAI/tsai) state-of-the-art deep learning library. 
We develop a framework that is able to handle univariate multistep time series prediction and easily allows for further development. 
The experimental results show that it is possible to train a machine learning (ML) model on a team and predict a single player's readiness, detecting detect peaks closely to actual values. 
It is possible to use the previous week to predict the upcoming day, or even the upcoming week, as the model does not require much data to get started. 
The model works well on data from the entire team for a shorter period than a larger set of data for a longer period, which allows the teams to quickly start using the system with existing data. 
Hyperparameters are easily configurable and can be changed as required to optimize the model. 
Our results can be used for evidence based decisions, such as benching the team star so she doesn't get injured for the rest of the season. 
As a first milestone, this framework will be incorporated in PMSys and used in the Norwegian the elite female soccer league, Toppserien, but the overall approach can be part of a standardized athlete performance monitoring system that is globally used by athletes in all sports. 

**_Keywords_** machine learning (ML), artificial intelligence (AI), long short-term memory (LSTM), univariate, deep learning, time series prediction, female soccer, Tsai, Python, Amazon Web Services (AWS)


## Instructions

We present a single- and multi-step time series prediction framework implemented in Python based on the [Tsai](https://github.com/timeseriesAI/tsai) deep learning library.

Clone the repository:
```
git clone https://github.com/simula/pmsys
```

Navigate to project directory:
```
cd nourhan-ragab
```

Install requirements:
```
pip install -r requirements.txt
```

Run Jupyter Notebook:
```
jupyter notebook
```

Navigate to the folders `Team A` and `Team B` to run the desired notebook.

