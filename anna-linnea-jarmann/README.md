# Anna Linnea Jarmann - Identifying Injury Risk Factors in Elite Soccer Teams

## Abstract
Soccer is a sport millions of people enjoy worldwide, with a significant amount of resources devoted to injury prevention for player health and team performance. Various methods are used to assess injury risk as part of injury prevention. One commonly employed method is using the training load metric Acute Chronic Workload Ratio (ACWR) to measure injury risk. However, its limitations have sparked discussions and scepticism. Machine learning techniques have also emerged as a method of injury prevention by recognising injury risk factors and predicting injuries. However, few studies have explored the use of survival analysis for this task.
This thesis aims to fill this gap by extending survival analysis beyond its traditional use in medical research to injury risk assessment in sports. We investigate injury risk factors in subjective training load and wellness data from two elite female soccer teams and extract the variables with the most significant impact on injury outcomes. We experiment with different approaches and apply the Cox Proportional Hazards Model (CPH) to estimate the magnitude of each variable's impact on injury risk. We also explore time-varying analysis using the Cox Time-Varying Model (CTV).
Our results show that combining recurrent injuries with averaged variables from all days prior to injury provides the most accurate and reliable results, also allowing for time-varying analysis. We perform feature selection using regularisation to extract the most significant factors for injury risk, which include prior injuries, sleep quality, fatigue, and ACWR. Using cross-validation, we determine the optimal penalty term for regularisation based on the lowest Bayesian Information Criterion (BIC) and highest Concordance Index (C-index). 
Our research significantly contributes to computer and sports science by offering a novel approach for extracting injury risk factors in a dataset using survival analysis. We deliver valuable insights into the factors affecting injury risk in female soccer players. Additionally, our results provide possibilities for developing targeted injury prevention programs and improving player health and performance in various sports and injury types.

## Experiments
Our research is divided into four experiments for finding the most optimal approach to our issue.

**Experiment 1:**
Compares the survival functions of univariate survival models with the injury distribution in the dataset by plotting the functions against a histogram of injury frequency in a team.

**Experiment 2:**
Multivariate analysis with the Cox proportional hazards model and chosen covariates to measure the effects of the covariates on injury risk. Uses day-of-the-event values and first and recurrent injuries.

**Experiment 3:**
Multivariate analysis with the Cox time-varying model and chosen covariates. Uses time-dependent covariates and only first injuries.

**Experiment 4:**
Multivariate analysis with the Cox proportional hazards model with regularization. Uses averaged covariate values from whole durations and first and recurrent injuries.

## Instructions
Clone repository:

    git clone https://github.com/simula/pmsys
    
Access correct directory:

    cd anna-linnea-jarmann
    
Install the required packages:
    
    pip install -r requirements.txt
    
Download data from SoccerMon in the same directory:

https://osf.io/uryz9/

Files -> DropBox -> Subjective -> Download

Navigate to plotting experiments:

    cd plotting

Run an experiment:
    
    python3 experiment_1.py

Run all experiments:

    python3 experiment_1.py & python3 experiment_2.py & python3 experiment_3.py & python3 experiment_4.py
    
