# Lars Hoel - Using Soccer Athlete GPS Monitoring Data to Visualize and Predict Features


## Abstract
Football is a globally popular sport with millions of players and fans
engaging in the game across all levels of competition. As one of the world’s
most-watched sports, football demands constant improvements in data
analysis tools. This master’s thesis presents a comprehensive pipeline for
feature extraction, data visualization, and injury prediction, utilizing GPS
data collected from two Norwegian women’s soccer teams. It outlines the
development of a systematic process, commencing with preprocessing and
feature extraction from raw GPS data, to facilitate subsequent analysis and
model training. This process culminates in the creation of two distinct
datasets - ’Session’ and ’High Intensity Run’ - which offer invaluable
insights into player performance and physical attributes.
The study then delves into the creation of several visualization tools,
utilizing a mix of the aforementioned datasets, raw data, subjective
performance data, and match data. The resulting visualizations serve
diverse purposes, providing insights into high-intensity runs, player
positions, team heatmaps, and the relationships between subjective game
performance, objective GPS metrics, and match data. These tools exhibit
potential in assisting players, coaches, medical staff, researchers, and sports
scientists in a multitude of scenarios, such as managing tactics, preparing
for high-intensity periods, and evaluating player mindsets.
Lastly, the thesis explores injury prediction through the deployment of
various machine learning models. After testing several models, including
Logistic Regression, Decision Tree, xGBoost, LSTM, GRU, and ROCKET,
the ROCKET model is found to outperform others for the given dataset,
with precision of 0.4167 and recall of 0.4545 (TP:5, TN:2978, FP:6, FN:7).
However, the model’s performance is found lacking in consistently
predicting injuries, thereby underscoring the need for continued research
in this field. This study’s comprehensive process and findings contribute
significantly to enhancing our understanding of the application of GPS data
in professional sports, while pinpointing areas for future investigation.

## Instructions

Clone the repository:
```
git clone https://github.com/simula/pmsys
```

Navigate to project directory:
```
cd lars-hoel
```

Install requirements:
```
pip install -r requirements.txt
```

Run Jupyter Notebook:
```
jupyter notebook
```

To use the modules, you need to download the SoccerMon dataset, found here:
https://osf.io/uryz9/


## Ethical Considerations
To preserve personal data, the code for creating visualization based on non-public data such as results are removed from this repo.