# Santander-Customer-Transaction-Prediction
Kaggle competition


# Purpose

Identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. 
competition link:https://www.kaggle.com/c/santander-customer-transaction-prediction

# Summary and Great kernel

I'm start with this simple and beautiful kernel.
## LGB 2 leaves + augment
https://www.kaggle.com/niteshx2/beginner-explained-lgb-2-leaves-augment

It use augmentation to deal with data imbalance, it's really helpful at first.
It can quick help you to reach 0.901.

And then, I found there is a problem with it.
No matter how hard I try,I can't break 0.901 with it.
The information is not enough, I need to do some FE.
So I start to do EDA again.
After EDA 'again', I've tried a lot of ways to do the FE(maybe 20 ways), but nothing work.

I was wondering what ML can't see? what ML can't do?
I found this kernel that inspired me.
## Why your model is overfitting/not making progress
https://www.kaggle.com/felipemello/why-your-model-is-overfitting-not-making-progress

This kernel make me walking on the right path.
I found the secret of the data.
And I won a Silver medal(TOP 5%).
All I need to do is give the information that ML can't see.

But there is one thing I missed it.
Someone has found that there are some fake datas in test set.I've read the discussions.
I didn't realize that is so importantce.
These fake data in test set block me out from the Gold medal.
this is an unsual issue. but I've learned it now.

## "When something is wierd in data, we all need to take look of it, no matter what it is."

## List of Fake Samples and Public/Private LB split
https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split
