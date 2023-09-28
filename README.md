This repository accompanies the paper "Sp-R-IP: A Decision-Focused Learning Strategy for Linear Programs that Avoids Overfitting" submitted to ICLR 2024.

This includes both Julia (v1.6.7) and Python (v3.8.1) scripts for different types of reforecasters. These are the folders:

* **data** contains both the data used for training the initial forecaster, as well as the data used for training the decision-focused re-forecaster; next to that, it contains pre-trained neural networks for warm starting the re-forecaster
* **experiment** contains code for running the training procedure for the problem of optimizing ESS profits in the day-ahead energy market. The files '
