# INM706-Twitter_Sentiment_Analysis

Sentiment analysis (SA) is a procedure that uses Deep Learning techniques to extract emotions, perspectives, and opinions from tweets, status, and other mediums. Sentiment analysis involves categorising textual opinions into two categories such as "positive" or "negative".

In this coursework, we will be making use of the “Twitter sentiment analysis training corpus dataset”. The dataset contains a corpus of tweets that have already been sentimentally categorised. It consists of 1,578,627 classified tweets in which each row is marked as “1” for
positive sentiment and “0” for negative sentiment. The dataset can be downloaded from the following link "http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/".

The code has the following outline, on which I will add more on below.
• Processing the data
• Making a custom class for data
• Building model
• Implementing ignite functionalities (callbacks)

Processing data –
Initially we need to install and import the libraries which we will be using in the code. We then read our dataset using read_csv and store it in a dataframe and do some analysis of the data we are working with. We check the column names and if there are any null values in the data. The dataset has 1,578,612 tweets which we will be using to predict the sentiments. There are only two classes (0 and 1), and both the classes have almost similar number of samples. Next, we will split the dataset into two, for training and validation (80-20).

Custom class for data –
Now, we defined a custom class SentimentDataset to process the tweets. We will use the same class for both training and validation dataset. The additional white spaces from the data have been stripped away and made clean. If we are working with training dataset, we will have to build a vocabulary. So, we use a counter to build the word count and then build the vocabulary getting the word2idx and idx2word. If we are working with the validation dataset, we don’t have to build the vocabulary, we simply need to use the word2idx and idx2word which we had developed for the training dataset. We then pass the training and validation dataset to this custom class to get the dataset ready for our model. To prepare the data for dataloader, we defined our collate function. In this we have padded the tweets to make them all uniform length. This function returns the padded vectors, labels, and the original length of the tweets. Finally, to get our data ready for the model, we pass the train and validation datasets through the DataLoader and apply the collate_fn and set the batch size. We have used a batch size of 1024. Now we have train_dl and val_dl which is ready to be used as input for our model.

Building model –
We will be using GRU as our base model. However, to build up on that I have used concat pooling with GRU. In concat pooling, we take max and average of the output at all the timesteps and then concatenate it with the last hidden state before passing it to the output layer.

Ignite callbacks –
Ignite has an important feature of callbacks which can be used for training and evaluating neural networks in PyTorch. We define a single training and validation loop. In the process_function we set a single training iteration to be used with trainer engine. In the eval_function there is an evaluator loop which is used to evaluate the performance of the training and validation dataset. We then attach the metrics we will be using for our model. The accuracy and the loss (binary cross entropy) of the model in the epoch is attached to the trainer engine.
We then created a function to log the results for the training and validation datasets. This function runs at the end of each epoch and reports the accuracy and loss.
Finally, we just have to run the model. We pass the data to the trainer.run() function and set the number of epochs to train the model. In this case, the model is trained for 10 epochs.










