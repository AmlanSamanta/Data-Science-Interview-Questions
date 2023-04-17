# Data-Science-Interview-Questions
This repo would contain important data science and ML interview questions.


## What do you mean by Precision and Recall?

### In classification problems, precision attempts to answer following question:

What proportion of positive identifications was actually correct?

Mathematically, it is defined as follows:

Precision: TP/(TP+FP)


### Recall attempts to answer following question:

What proportion of actual positives was correctly identified?

Mathematically, it is defined as follows:

Recall: TP/(TP+FN)


Consider following example:

Suppose our model is tasked with analysing and classifying 100 scans of tumours as malignant(positive class) and benign(negative class).   Let's say, the model is doing as following on the data:

True Positive - Predicted: Malignant & Actual: Malignant -> 1
False Positive - Predicted: Malignant & Actual: Benign -> 1
False Negative - Predicted: Benign & Actual: Malignant -> 8
True Negative - Predicted: Benign & Actual: Benign -> 90

So the model's accuracy comes out to be:

(TP+TN)/(TP+TN+FP+FN) = (1+90)/(1+90+1+8) = 0.91, i.e. 91%, pretty cool, right?

Let's dive deeper inside the results to gain more insight into our model's performance:

Of 100 tumour examples, 91 are benign (i.e. 90 TNs and 1 FP) and 9 are malignant (i.e. 8 FNs and 1 TP). Out of 91 benign tumours, the model classifies 90 images as benign correctly which is good, but out of 9 malignant tumours, it correctly identifies 1 image as malignant which is quite bad, as 8 out of 9 malignants got undiagnosed ! Another benign tumour classifier model which always predicts benign would achieve same level of accuracy for our dataset, means our model is no better than one that has zero predictive capacity to differentiate between malignant and benign tumours. So accuracy in this kind of cases would not capture the complete truth of the model's performance, where the dataset is class-imbalanced.

Let's calculate the precision and recall of the model:

Precision is TP/(TP+FP) = 1/(1+1) = 0.5. i.e. 50% means when our model predicts a tumour as malignant, it's correct 50% of the times

and

Recall is TP/(TP+FN) = 1/(1+8) = 0.11, i.e. 11% means our model correctly identifies 11% of all malignant tumours.



## In which case Precision is to be used and in which case Recall is to be used?

To better explain the answer to this question we have to understand the notion of false positive and false negative, i.e. Type I and Type II error respectively. False positive or Type I error occurs when the null hypothesis is rejected but in fact it was true. And we commit Type II error when the null hypothesis is accepted but in fact it was false.

Suppose, we have developed a model to determine a stock will be good for investment or not. Here the +ve class is 'good for investment' and -ve class is 'not good for investment'. We are not concerned with mislabelling a good stock as not good, i.e. False negative is not so important in this use case because we can afford to miss few profitable stock here and there as long as our money is going to an appreciating stock correctly predicted by our model. Rather if we mislabel a bad stock as good that would be terrible. And we would want to reduce that, i.e. False positive. So Precision will be our go-to metric for evaluating our model's performance. 	 

Let's say we have developed a poisonous apple classifier, i.e. The +ve class is poison and -ve class is not poison. We are not too concerned with mislabelling an apple as poisonous because we would rather be safe than sorry, i.e. False positive is not important for this use case. Rather if we miss poisonous apple and mislabel it as not poisonous, i.e.the False negatives get increased, the model will not be effectively perform. So Recall would be appropriate.

So, it entirely depends on the goal of your model to determine the appropriate evaluation metric for our model




