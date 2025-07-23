## Project Overview:
  - The purpose of this project is to design a model that accurately calculates CVaR and VaR for a given equity portfolio.
  - The project is roughly halfway complete, and I am working on developing an unsupervised classification model that classifies economic regimes. 
    
## Files:
  - [utils](https://github.com/henrycosentino/state_dependent_cvar_var/blob/main/State%20Dependent%20CVaR%20%26%20VaR/utils.py): holds all helper functions, and the primary classes.
  - [processing](https://github.com/henrycosentino/state_dependent_cvar_var/blob/main/State%20Dependent%20CVaR%20%26%20VaR/processing.ipynb): loading, processing, and feature engineering of the data occurs here.
  - [classification](https://github.com/henrycosentino/state_dependent_cvar_var/blob/main/State%20Dependent%20CVaR%20%26%20VaR/classification.ipynb): development of an unsupervised classification algorithm (KMeans) to classify economic regimes occurs here.
  - [model_dev](https://github.com/henrycosentino/state_dependent_cvar_var/blob/main/State%20Dependent%20CVaR%20%26%20VaR/model_dev.ipynb): development and evaluation of the final model (logistic regression), which will assign probabilities to the chance of an economic regime occurring one day forward.
  - Once the analysis and research processes have been completed, a final file (main) will be created to streamline the processes and calculate CVaR and VaR!
