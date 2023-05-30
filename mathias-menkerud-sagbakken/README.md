# Using Machine Learning to Predict Elite Female Athletes' Readiness to Play in Soccer

<p align="center">
<img src="front-page-soccer.png" alt="Soccer analytics visualized"
	title="Soccer analytics visualized" width="40%"/>
</p>



Table of contents
=================

<!--ts-->
   * [Introduction](#Introduction)
      * [Abstract](#Abstract)
      * [Pipeline](#Pipeline)
      * [Privacy](#Privacy)
   * [Requirements to use the codebase](#Requirements-to-use-the-codebase)
      * [Required Packages](#Required-Packages)
      * [Data Access](#Data-Access)
      * [How to Run](#How-to-Run)
<!--te-->


Introduction
============
This codebase uses machine learning to predict elite female athletes' readiness to play in soccer. The data is statistically analyzed using quantitative data and further pre-processed to be used in several machine learning models. We use several state-of-the-art machine learning models to generate predictions for both regression and classification tasks. 

Abstract
-----

In today's world, both sports and technological advancements hold a key role in our society. Machine learning is becoming a more relevant tool in the industry due to the abundance of data available to analyze players and strategies. Especially soccer, the most watched sport in the world, embraces this technological shift to improve training conditions among players and provide insight into game strategies. To maximize performance in soccer, athletes push themselves to the limits of their potential. This rigorous process puts athletes at continuous risk of negative developments such as injuries and illness. Therefore, athletes and coaches coordinate and execute training based on experience and data from different monitoring tools to achieve a safer athletic progression. These tools result in large amounts of data, which can be difficult to interpret. A machine learning model can utilize these data to make predictions of an athlete's future performance and make the transition from raw data to strategy easier. These strategies can include choosing the most relevant players for important events and improving training conditions. Most attempts at using time series data to predict future performance among soccer players have been limited to male soccer using either wellness or positional data paired with a small selection of often simple machine learning models. In this thesis, we present a pipeline conducting several experiments to determine important data and model configurations when predicting readiness to play among professional female soccer players. The pipeline comprises data extraction, pre-processing and data analysis, experiments, and evaluation, where we visualize and quantitatively present results. We discover that by leveraging complex imputation for multivariate data, we reduce error by up to 16\%. We present three use-cases and show their ability to generate actionable data that enhances player and team performance. Our proposed experiments and extensive data analysis show how utilizing an approach that can dynamically capture the unique response of players improves results. The methods used in this thesis further contribute to generalizing these time series analyses to other sports. 


Pipeline
-----

<p align="center">
<img src="pipeline_v2.png" alt="Soccer analytics visualized"
	title="Soccer analytics visualized" width="40%"/>
</p>

The pipeline shows the overall workflow of our code. We extract data from a MySQL database and process it. The data is further used for experiments, then stored, and finally visualized.

Privacy
-----

The work in this codebase centers around using wellness and positional data from professional Norwegian female soccer teams. Therefore, the data is made anonymous. To be more specific, this means the removal of all metadata and the use of randomly generated file names. Each athlete has also given their consent and knows what data is collected and in what way it is being used. Further, this data has been exempt from further demand of consent from users since it is anonymous and certified by the Norwegian Privacy Data Protection Authority.


Requirements to use the codebase
============
Following is a description on how to use the code.

Required Packages
-----
Install requirements:
```
pip install -r requirements.txt
```

Data access
-----
To use the data it is necessary to gain access to the MySQL database. When access is gained, the mysql-config.json file needs to be configured with the following data:
```
{"host":"host",
"database":"database",
"user":"user",
"passwd": "passwd"}
```

How to Run 
-----

The experiments are exectuted by running the scripts, for example:
```
python all_models_benchmark.py
```
The Jupyter Notebooks can be run with the following command:
```
jupyter notebook
```
