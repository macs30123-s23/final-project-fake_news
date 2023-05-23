# MACS 30123 - Large-Scale Computing for the Social Sciences

## Final Project: Fake News Detection

### Introduction & Motivation
Fake news has become a major problem. A Pew Research Center survey conducted in 2016 found that nearly two-thirds of U.S. adults say that fake news has caused a great deal of confusion about basic facts of current events and nearly a quarter have said that they have either knowingly or unknowingly shared fake news online[^1]. Due to the widespread nature of social media, the spread of misinformation is fast and can have significant consequences for individuals, communities, and societies. For example, fake news can lead to distrust of the media, undermining of the democratic process/contribute to political divisiveness, and influence erroneous decision-making[^2].

By developing accurate and effective methods to predict fake news, social scientists can: 1. understand how people consume information (specifically cognitive biases, social homophily and inattention) which can shed light on the psychological, social, and technological factors that contribute to the viral spread of misinformation; 2. assess the impact of fake news on society on the most vulnerable communities/individuals most susceptible of sharing misinformation without proper verification and develop interventions to mitigate the harmful effects; and finally, 3. understand the specifics of fake news that lead to manipulation of populations exposed and how to combat this phenomenon.

### Research Questions
1. Can fake news be predicted and if so, how well?
2. What are the biggest differences between articles from reliable and unreliable sources and are there topics that are more susceptible to being faked?

### Why Large-Scale Computing is Important in NLP (non-exhaustive list):
1. Text processing, that is, the act of turning text into a numeric input that a machine learning model can take in, would benefit largely from parallelization. Namely, tokenization, the act of breaking down chunks of text into smaller subunits (e.g., words) is a necessary step that can be computationally expensive, especially when dealing with large documents.
2. Feature extraction such as obtaining n-grams from text can lead to extremely wide dataframes (high dimensions--count vectorizers increase in folds of the length of the vocabulary size, which can be in the tens of thousands), requiring substantial memory resources.
3. Large language models (not used in this project, but can be applied to increase accuracy), have millions of parameters leading to the need for more compute-intensive resources.
4. Model fine-tuning often involve computationally expensive and time consuming procedures such as hyperparameter tuning via grid search.

### Project Structure

#### Data


[^1]: https://www.pewresearch.org/journalism/2016/12/15/many-americans-believe-fake-news-is-sowing-confusion/
[^2]: https://libguides.exeter.ac.uk/fakenews/consequences