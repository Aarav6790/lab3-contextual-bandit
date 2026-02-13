# Lab 3: Contextual Bandit-Based News Article Recommendation System

Name: Aarav;
Roll No.: U20230092;
Course: Reinforcement Learning Fundamentals.


## Project Overview

This project implements a Contextual Multi-Armed Bandit framework for news article recommendation. The system classifies users into three categories (User1, User2, User3) and learns to recommend news articles from four categories (Entertainment, Education, Tech, Crime) to maximize user engagement.

## Problem Setup

The system operates with:
- 3 user contexts (User1, User2, User3)
- 4 news categories per context
- 12 total arms (3 contexts × 4 categories)
- 10,000 training time steps

## Implementation Summary

### Data Preprocessing
- Handled missing values in user features using median imputation for numerical data and mode imputation for categorical data
- Removed browser_version column as it had 1973 unique values out of 2000 users, providing no useful pattern for learning
- Encoded categorical features and normalized all features for classification
- Processed 2,000 training users and 209,527 news articles

### User Classification
- Implemented Random Forest Classifier to predict user categories
- Split training data into 80% training and 20% validation
- Achieved 90% accuracy on validation set

Classification Performance:
```
              precision    recall  f1-score   support

      user_1       0.90      0.86      0.88       142
      user_2       0.98      0.88      0.93       142
      user_3       0.84      0.98      0.90       116

    accuracy                           0.90       400
   macro avg       0.90      0.91      0.90       400
weighted avg       0.91      0.90      0.90       400
```

### Bandit Algorithms
Three contextual bandit strategies were implemented and evaluated:

1. Epsilon-Greedy: Balances exploration and exploitation using probability parameter epsilon
2. Upper Confidence Bound (UCB): Uses confidence intervals to guide exploration
3. SoftMax: Probabilistic selection based on estimated rewards with temperature parameter

## Results and Performance Comparison

### Algorithm Performance

The three bandit algorithms showed different performance characteristics across the three user contexts, with each algorithm's design philosophy directly influencing its behavior:

**Epsilon-Greedy:**
- Simple implementation with random exploration
- Performance sensitive to epsilon parameter selection
- Best performance achieved with moderate epsilon values around 0.05-0.1

Epsilon-Greedy's main limitation is its fixed exploration rate - it explores randomly with probability epsilon regardless of accumulated knowledge. At time step 9,000, it explores the same way as at step 100, wasting pulls on clearly inferior arms. This causes linear regret growth over time. Low epsilon risks premature convergence to suboptimal arms, while high epsilon wastes opportunities on random exploration even after finding good arms.

**Upper Confidence Bound (UCB):**
- Systematic exploration based on uncertainty
- More robust to hyperparameter selection
- Best overall performance among the three algorithms

UCB uses optimism under uncertainty - it preferentially explores arms with high uncertainty while exploiting known good arms. The confidence bound decreases with more samples, creating automatic transition from exploration to exploitation. This achieves logarithmic regret growth compared to Epsilon-Greedy's linear growth. UCB efficiently identifies and eliminates inferior arms while investigating uncertain options, making it more sample-efficient across all contexts.

**SoftMax:**
- Probabilistic selection based on estimated rewards
- Stable performance with fixed temperature of 1.0
- Competitive but slightly lower than UCB performance

SoftMax uses a Boltzmann distribution to convert values into selection probabilities - better arms are chosen more often but no arm is completely ignored. This smooth probabilistic approach prevents getting stuck on bad arms but also prevents fully committing to optimal choices. With fixed temperature, SoftMax maintained selection probabilities for inferior arms even late in training (e.g., 20-30% for clearly bad arms), leading to persistent inefficient exploration. This explains why it underperformed UCB despite being more stable than Epsilon-Greedy.

### Performance Metrics

UCB achieved the highest cumulative rewards across all contexts and converged the fastest, as its confidence-bound strategy efficiently balanced exploration and exploitation. In contrast, Epsilon-Greedy performed worst overall, with both convergence speed and cumulative reward heavily dependent on the choice of ε; larger ε values led to excessive exploration, while smaller values risked premature convergence to suboptimal actions. SoftMax performed between the two, showing smoother and more stable improvement over time but slower convergence than UCB. Overall, the regret growth followed theoretical expectations: Epsilon-Greedy exhibited linear regret, UCB logarithmic regret, and SoftMax sub-linear regret.


### Hyperparameter Sensitivity

**Epsilon-Greedy (tested epsilon values: 0.01, 0.05, 0.1, 0.2, 0.3):**
Performance varied dramatically with epsilon. Very low values (0.01) risked premature convergence, while high values (0.2-0.3) wasted 2,000-3,000 pulls on random exploration. Optimal performance around 0.05-0.1 balanced discovery with exploitation.

**UCB (tested C values: 0.5, 1.0, 2.0, 3.0, 5.0):**
Much more robust to hyperparameter variation. Performance remained strong across C values 1.0-3.0 because UCB's exploration adapts based on uncertainty rather than fixed probabilities. C around 2.0 provided optimal balance between exploration and exploitation.

### Recommendation System Performance

Successfully generated 2,000 recommendations for test users:
- User1 → Tech: 682 recommendations
- User2 → Crime: 706 recommendations  
- User3 → Entertainment: 612 recommendations

All recommendations were successfully mapped to specific articles from the news database.

## Key Findings

1. UCB demonstrated best overall performance due to its uncertainty-driven exploration that automatically adapts as confidence increases, achieving logarithmic regret growth.

2. Epsilon-Greedy required careful hyperparameter tuning and suffered from fixed exploration rate that wasted samples on known inferior arms late in training.

3. SoftMax provided stable middle-ground performance with smooth probabilistic selection, though it couldn't match UCB's efficiency.

4. Context-aware recommendation significantly improved over random selection by learning distinct preferences for each user type (User1→Tech, User2→Crime, User3→Entertainment).

5. Algorithm performance was highly dependent on reward structure - UCB's advantage was pronounced in ambiguous contexts (User2) and when dominant arms made wasteful exploration costly (User3).

## Conclusions

The contextual bandit approach proved effective for personalized news recommendation. UCB emerged as the most robust algorithm, while Epsilon-Greedy offered a simpler alternative with good performance when properly tuned. The system successfully classified users and learned context-specific preferences, demonstrating the value of contextual information in multi-armed bandit problems.

The recommendation engine successfully integrated user classification with bandit policies to deliver personalized article suggestions for all 2,000 test users, validating the complete system architecture.
