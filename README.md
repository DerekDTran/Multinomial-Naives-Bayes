# Multinomial-Naives-Bayes
The Twitter US Airline Sentiment dataset contains 14640 elements with 15 characteristics. For this code, we will focus on only 3 characteristics, airline_sentiment, airline, and text. With this dataset, we will use the Naive Bayes Classifier for text classification.

## Preprocessing
Before building the model, we first preprocess the text characteristic. We convert all values into lowercase, then use CountVectorizer and TfidfTransformer to transform the data. We then convert the airline_sentiment characteristic from categorical values to numerical values using LabelEncoder.

## Default Multinomial Naive Bayes model
We split the data into training and testing dataset where the test dataset contains 10% of the original data. We pass the training data into MultinomialNB() to build a multinomial naive bayes model using the default parameters, alpha = 1 and fit_prior = True. For context, the alpha parameter sets the additive smoothing parameter where 0 is no smoothing. The fit_prior parameter simply notifies the program if it can learn class prior probabilities or not. If the parameter is set to false, then a uniform prior us used to build the model. We then calculate the accuracy of the model and view its classification report.

## Multinomial Naive Bayes model using different parameters
To see if the model can be improved, we use different parameters when building the model. After storing the different parameters used with its respective model accuracy, we notice how the accuracy does increase when alpha is set to a lower value with its value depending on whether the fit_prior is set to True or False. This would imply that the model improves when the there is less smoothing.

## Data analysis of airlines and airline sentiment
In this code, we also calculate the average airline sentiment for each airline group. We can conclude that Virgin America has the highest positive sentiment compared to the other airlines.
