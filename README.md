# auto_ml_on_compressive_strength_of_concrete_mixes
✍️ Auto ML on the compressive strength of different concrete mixes. The dataset is given in the UCI machine learning repository.
Link for the dataset: https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength


## Training and Comparing

train concrete.csv and test concrete.csv contain data about the compressive strength of several different concrete mixes: https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
We will write a program called concrete train.py that will use pycaret’s compare models (no turbo!) to try a large variety of regression algorithms on train concrete.csv.  
It will pick the best six (based on R2 ) and it will tune (using at least 24 different parameter
combinations) and finalize each before saving the finalized model to a pickle file. Thus six .pkl
files will be created.  
Run the program and save the output to train.txt. 

When I run this, I end up with a pickle file for the top six models:
• LGBMRegressor.pkl
• CatBoostRegressor.pkl
• MLPRegressor.pkl
• ExtraTreesRegressor.pkl
• RandomForestRegressor.pkl
• GradientBoostingRegressor.pkl

## Testing

We will write a program called concrete test.py that will scan the current directory for .pkl
files. It will use pycaret to load those in one at time.  
Each model will be tested on concrete test.py. The program will print the time that inference
required and the R2 value.  
Run the program and save the output to test.txt
My test.txt looks like this:  

GradientBoostingRegressor:  
Inference: 0.0095 seconds  
R2 on test data = 0.9110  
RandomForestRegressor:  
Inference: 0.0319 seconds  
R2 on test data = 0.8815  
AdaBoostRegressor:  
Inference: 0.0147 seconds  
R2 on test data = 0.7239  
ExtraTreesRegressor:  
Inference: 0.0170 seconds  
R2 on test data = 0.9007  
MLPRegressor:  
Inference: 0.0052 seconds  
R2 on test data = 0.7861  
CatBoostRegressor:  
Inference: 0.0030 seconds  
R2 on test data = 0.9079  
LGBMRegressor:  
Inference: 0.0071 seconds  
R2 on test data = 0.9101  
