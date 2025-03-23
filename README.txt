# Introduction-to-Artificial-Intelligence-coursework

Explaination about code for Linear regression predicting california house pricing: 
I start by importing all the necessary libraries and scikit-learn modules to execute the code. I then load the dataset using fetch_california_housing() from sklearn.datasets. The dataset is then converted into Pandas DataFrame while target variable MedHouseVal is added to the DataFrame. A Scatter plot of MedInc vs MedHouseVal was then created. The summary statistics were calculated for both variables. 

Next, process the data by splitting it to be 80% training and 20% testing using setstrain_test_split(). The feature (MedInc) is standardized using StandardScaler() to ensure that the model performs well by scaling the data to have a mean of 0 and a standard deviation of 1.

The model is now trained using batch gradient descent through the LinearRegression() model from sklearn.linear_model used to fit the training data. For the stochastic gradient descent the SGDRegressor() model is used with max_iter=1000 and learning_rate="optimal" to fit the training data.

Both models are used to predict the median house value for a specific example where MedInc = 8.0 using med_inc_example = np.array([[8.0]]). 

Both models are evaluated using mean squared error mse = mean_squared_error(y_test, y_pred)

Lastly, we plot the linear regression line over the scatter plot 


Explanation about code for Human Activity Recognition using SVM:
I start by importing all the necessary libraries and scikit-learn modules to execute the code. I then load the data by pathing it to the HAR file downloaded and further pathing the relevant training and testing files. Next step is to list the feature names using feature_names = features_df["feature"].tolist(). Due to the duplicate name issue I then had to make the feature names unique to fix the problem. Next we convert the features into binary form.

The pipeline was created using StandardScaler for feature scaling, PCA for dimensionality reduction and svc for the 3 kernel types. The PCA reduced the features from 561 to 100 through PCA(n_components=100). 

I train and evaluate the SVM models with different kernels (linear, poly, rbf) and evaluates their performance on the test set. I also added the confusion matrix for teh kernels required by confusion_matrix(y_test["Binary"], y_pred). Each kernel will calculate accuracy, generate a confusion matrix, and print a classification report.

Next we perform GridSearchCV for Hyperparameter tuning. I set up a parameter grid for GridSearchCV to find the best hyperparameters for the SVM model. This performs a grid search with 3-fold cross-validation to find the best combination of hyperparameters (C, gamma, degree for different kernels). The best model is then used to predict on the test set, and the results (confusion matrix and classification report) are printed. Finally, The script prints the best parameters found by GridSearchCV, the best cross-validation accuracy, and the performance metrics (confusion matrix and classification report) for the best model on the test set.
