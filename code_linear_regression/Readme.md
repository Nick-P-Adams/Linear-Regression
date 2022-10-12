# Implement and evaluate linear regression 

All coding happens in `linear_regression.py`.

## A real-world house price data set 

Also enclosed is a real-world house price data set in the CSV format.

- x_train, y_train: the training data set samples and labels.

- x_test, y_test: the validation data set samples and labels.

Reminder and suggestion: 

- When you read in the data set into numpy array, do not misalign/mess up the samples and their labels. 

- An easy way to read in CSV files is to use the `Pandas` library. 




## Sub-project 1.1: Finish `_fit_cf()`.   (30 points)

This function is to implement the closed-form method for linear regression. 

In this subproject, you do not need to worry about and do not need to touch the parameter `lam`, which will be used in a later subproject. 

## Sub-project 1.2: Finish `predict()` and `error()`.  (20 points. 10 points each)

See the instruction in the docstring and comment of the function in the source code file.



## Sub-project 2: Finish `_fit_gd()`. (50 points)

This function is to implement the gradient descent based method for linear regression. 

In this subproject, you do not need to worry about and do not need to touch the parameter `lam`, which will be used in a later subproject.


## Sub-project 3: Revise `_fit_cf()`. (25 points)

In this subproject, you will revise your Subproject 1's code for function `_fit_cf()`, so that the function will use the parameter `lam` for regularization in the closed-form method for linear regression.


## Sub-project 4: Revise `_fit_gd()`. (25 points)

In this subproject, you will revise your Subproject 2's code for function `_fit_gd()`, so that the function will use the parameter `lam` for the regularization in the gradient descent based method for linear regression.


## Sub-project 5: Train and evaluate the model (50 points)

**(Sub-project 5 is mandatory for CSCD596, but is for extra credits for CSCD496)**


After you finished the above four sub-projects, **for each method**, play (automate your play via scripting) with your code with different choices of hyperparameters and find which model (the $w$ vector) is the best one you will use. The model you will choose is the one that minimizes the out-of-sample error for the validation set. 

- The set of tunable hyperparameters for the closed-form method:

    - the degree of the $Z$ space

    - the $\lambda$ for regularization

-  The set of Tunable parameters for the gradient descent based method: 

    - the degree of the $Z$ space

    - the $\lambda$ for regularization

    - the learning rate $\eta$

    - the number of epochs




**For each method**: Plot and show the change of the in-sample errors (using the house price training set) and the out-of-sample errors (using the house price validation data set) while using a variety of different combinations of the hyperparameters listed above. You can decide your own way to plot the pictures with the principle being that the plot 1) shall show the error changes over the changes of hyperparameters, and 2) shall present why the one you chose is the best model.


Refer to the PLA discussion's slide that explains the under-fitting/overfitting concept that tells you which model to pick. Feel free to add necessary code in the `LinearRegression` class to produce/save those in-sample/out-of-sample errors that you can plot and observe.

Report what you find in a PDF file that includes the plots and a reasonable amount of text that explains what you did and which model you chose and why, for each method. 


# Data normalization 

When you use gradient descent based methods (for ex., sub-projects 2, 4, and 5), it is strongly suggested that your sample features be normalized. There are multiple different ways to normalize the data. For the house price data, every feature value is non-negative, I suggest to normalize them into the [0,1] range. The enclosed `utils.py` module has a tool function for that normalization (see examples in the notebook for testing that I provided). **Please know that:** If you deploy your model for production, every future sample's features need to be normalized using **the same scale, range, formula, and min/max** used for the normalization that you used during the training phase. 

If you do not normalize the data, during the training phase, the gradient may become very large at some certain axis(es) and thus may cause over shooting. You can try it with the house price data and you will see it. 


# Your submission:


- Compress `linear_regression.py`  and the PDF report file into a zip file named as `YourLastName_Your6digitEWUID.zip`.

- Submit the zip file.

- Note: If you are a student in the CSCD496 section and you choose not to do the sub-project 5, your zip file will only include the `linear_regression.py` file. 




# Misc.

- Also enclosed are two notebooks for testing purpose. One plays with randomly generate data; The other plays with the real-world real-estate house price data. Feel free to use/change them. You do not have to use them, if you want to have your own code for testing. 




