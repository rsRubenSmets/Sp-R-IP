This repository accompanies the paper "Sp-R-IP: A Decision-Focused Learning Strategy for Linear Programs that Avoids Overfitting" submitted to ICLR 2024.

This includes both Julia (v1.6.7) and Python (v3.8.1) scripts for different types of reforecasters. These are the folders:

* **data** contains both the data used for training the initial forecaster, as well as the data used for training the decision-focused re-forecaster; next to that, it contains pre-trained neural networks for warm starting the re-forecaster
* **experiment** contains code for running the training procedure for the problem of optimizing ESS profits in the day-ahead energy market. The files **exec_da_training.jl** and **exec_da_training.py** can be run to execute the training procedure. At the top of the files, users can make decisions on hyperparameters of the re-forecaster
* **initial_forecaster_training** contains a file that runs the training of the initial forecaster
* **results** 
