```table-of-contents
```


## Fairness in AI

### Definition
- Fairness in AI involves equitable distribution of benefits and costs, and ensuring that individuals and groups are not subject to unfair bias.

### Consequences of Unfair Bias
- May lead to marginalization of vulnerable groups.
- Can amplify existing prejudices and discrimination.

### Ethical Practice
- AI practitioners are advised to handle ethical dilemmas through reasoned, evidence-based reflection rather than relying on intuition.

### Summary
Fairness aims to protect against biases that can disadvantage individuals or groups, ensuring that AI systems operate justly and equitably.

---
## Unfairness in COMPAS Score Usage

### Overview
- The COMPAS score is used in the courtroom to predict the likelihood of a defendant reoffending and their risk of failing to appear in court.

### Controversy
- Analysis has shown discrepancies in risk scores that can lead to unfair outcomes, potentially influenced by racial bias.
- For instance, some individuals with minor offenses are rated as high risk, while others with more serious charges are rated as low risk, with patterns suggesting racial disparities.

### Ethical Implications
- These disparities raise serious concerns about the use of such AI tools in legal settings, where they can affect the lives and liberties of individuals.
- The controversy highlights the need for transparent, fair, and accountable AI systems in high-stakes decision-making.

### Reference
- ProPublica conducted an in-depth analysis that brought to light the potential biases in COMPAS scores, leading to widespread debate and scrutiny.

### Summary
The fairness of AI systems like COMPAS in legal proceedings is under debate due to evidence suggesting that the scores may reflect and amplify existing societal biases.

---
## Public Safety Assessment (PSA)

### Overview
- The PSA is a risk assessment tool used by judges to aid in pretrial decisions, such as whether a defendant should be detained or released before trial.

### Key Features
- **Risk Scores**: It provides scores estimating the likelihood of a defendant not appearing in court, committing new criminal activity, or committing violent crime if released.
- **Non-Discriminatory Factors**: The PSA calculates scores based on nine factors, explicitly excluding race, gender, education, and other sensitive attributes to promote fairness.

### Criticism
- A quote associated with the PSA highlights a critical view: "Prediction looks to the past to make guesses about the future. In a racially stratified world, any method of prediction will project the inequalities of the past into the future."

### Goal
- The PSA aims to increase safety, reduce taxpayer costs, and enhance fairness and efficiency in the justice system.

### Limitations
- While designed to be objective, the PSA and tools like it can inadvertently perpetuate historical biases present in the data they are trained on.

### Summary
The PSA seeks to assist judges with data-driven decisions while striving for neutrality; however, its effectiveness and fairness are subjects of ongoing debate.

For more information, visit the Arnold Foundation's website provided in the source.

---
## AI in Healthcare and Racial Bias

### Overview
- AI is used in healthcare to identify patients for high-risk care management, predicting risk scores to determine who gets enrolled in special programs.

### Algorithm Details
- **Task**: Predict which patients require high-risk care management.
- **Policy**: Patients above the 97th percentile risk score are enrolled.
- **Inputs**: Include demographics, insurance type, diagnosis and procedure codes, medications, and costs. Notably, race is excluded.
- **Output**: Health costs, used as a proxy for health needs.

### Issue Highlighted
- Black patients, despite generating lower medical expenses due to unequal access to care, are considered healthier by the algorithm than equally sick White patients at the same level of predicted risk.

### Implication
- The algorithm inadvertently perpetuates racial bias by associating healthcare costs with health needs, which disadvantages Black patients.

### Reference
- Obermeyer, et al. "Dissecting racial bias in an algorithm used to manage the health of populations." *Science*, 2019.
### Summary
This case exemplifies the challenges in AI healthcare algorithms, where seemingly neutral factors can lead to racially biased outcomes, emphasizing the need for more equitable measures of health needs.

---
## Bias in Machine Learning

### Cognitive Bias
- Refers to cognitive shortcuts that may lead to discriminatory judgments or decisions, often resulting from human prejudices or stereotypes.

### Statistical Bias
- In statistics, bias is the systematic tendency of a statistic to overestimate or underestimate a population parameter.

### Bias-Variance Tradeoff
- A fundamental concept in machine learning, where:
  - **High Bias**: Models are overly simplistic with low variance, potentially leading to underfitting.
  - **Low Bias**: Models have high complexity with high variance, which can cause overfitting.

### Fairness
- In the context of AI, fairness is often questioned whether it equates to the absence of bias. True fairness in AI requires careful calibration to avoid both statistical biases and discriminatory cognitive biases.

### Summary
Bias in machine learning encompasses both a tendency to err systematically in statistical prediction and the human-like cognitive errors that can lead to unfair outcomes. Balancing bias and variance is crucial for model accuracy, while fairness seeks to eliminate discriminatory biases.

---
## Race and Ethnicity

### Race
- **Concept**: A social construct that categorizes people based on physical characteristics.
- **Validity**: Has no biological or genetic basis, as humans are a single species with a wide genetic diversity and no clear genetic boundaries between groups.
- **Examples**: Common racial categories include White, Black, Asian, Native American, Pacific Islander, and mixed-race.

### Ethnicity
- **Concept**: A shared cultural heritage, language, and religion that identifies a particular group.
- **Flexibility**: Individuals can belong to multiple ethnic groups and can identify with different ethnicities at different stages of life.
- **Examples**: Common ethnicities include African, Arab, Asian, European, Hispanic, Jewish, etc.

### Summary
Race is a categorization based on perceived physical differences with no scientific foundation, while ethnicity refers to cultural identity and can encompass multiple affiliations for an individual.

---
## Biases in Machine Learning

### Types of Bias
1. **Historical Bias**: Pre-existing biases that are already present in the real world and can be reflected in the data.
2. **Representation Bias**: Occurs when the sample data doesn't accurately represent the environment it's meant to model.
3. **Measurement Bias**: Arises when the tools or methods used to collect data are flawed, leading to inaccuracies in the data.
4. **Aggregation Bias**: Happens when a model overly simplifies or ignores the diversity within a dataset, leading to generalizations that may not apply to all groups.
5. **Evaluation Bias**: Introduced during the model evaluation phase if the evaluation criteria or dataset are not aligned well with the true requirements of the task.
6. **Deployment Bias**: Emerges when a model is used in real-world settings for which it wasn't properly designed, or when it's interpreted by humans in a biased way.

### Explanation of Biases
- **Historical Bias**: Reflects societal stereotypes and norms from the past that may unfairly influence ML predictions.
- **Representation Bias**: Leads to models that perform well for majority groups but poorly for minority groups not well-represented in the training data.
- **Measurement Bias**: Distorts the true features or labels of the training data, potentially reinforcing stereotypes or incorrect assumptions.
- **Aggregation Bias**: Neglects subgroup specificities which can result in a model that is unfair or ineffective for particular groups.
- **Evaluation Bias**: Can give a false sense of performance if the test data used to evaluate the model contains bias.
- **Deployment Bias**: Can cause harm when models are used in contexts where their decisions reinforce or exacerbate existing inequalities.

### Summary
Machine learning can exhibit various types of bias at different stages, from data generation to model deployment, each affecting the fairness and accuracy of the outcomes.

---
## Fairness in Machine Learning

### Concept
- Fairness in machine learning (ML) can be sought through "unawareness," where sensitive features are not directly used in the model.

### Approach
- By excluding sensitive attributes like ethnicity from the dataset, the model is not explicitly influenced by these factors during the learning process.

### Challenge
- Even with direct exclusion, sensitive features may still influence the model indirectly through correlated variables.

### Example
- In a hiring model, not considering the candidate's ethnicity, but using features like "Loves tacos" may inadvertently introduce bias if this feature correlates with a specific ethnic group.

### Summary
Achieving fairness in ML through unawareness involves omitting sensitive features. However, indirect bias can still occur if other features correlate with sensitive attributes.

---
## Fairness Criterion: Independence (Demographic Parity)

### Definition
- Independence, or demographic parity, in the context of machine learning fairness, is when the decision-making process is statistically independent of any sensitive attribute.

### Formula
- Represented as \( R \perp\!\!\!\perp A \), meaning the probability of a positive outcome \( P(R = 1) \) should be the same across different groups defined by attribute \( A \).

### Example
- In a binary classification setting with two groups (e.g., \( A=0 \) and \( A=1 \)), the positive rate \( P(R = 1|A = 1) \) is the same as \( P(R = 1|A = 0) \).

### Implication
- This implies that the true positive (TP) and false positive (FP) rates should be equal across different demographic groups to satisfy demographic parity.

### Summary
Independence as a fairness criterion aims to ensure that sensitive attributes, such as race or gender, do not influence the likelihood of a positive prediction outcome.

---
## Fairness Criterion: Separation (Equality of Odds)

### Definition
- **Separation**, also known as equality of odds, is a fairness criterion where a model's predictions are independent of a sensitive attribute at each level of the actual outcome.

### Formula
- Expressed as \( R \perp\!\!\!\perp A | Y \), meaning the probability of a positive prediction \( P(R = 1) \) should be the same across different groups defined by attribute \( A \), given the true outcome \( Y \).

### Example
- For a binary classification:
  - \( P(R = 1|A = 1, Y = y) = P(R = 1|A = 0, Y = y) \) for \( y \in \{0, 1\} \).
  - This implies that true positive (TP) and false positive (FP) rates should be similar across groups like \( A=0 \) and \( A=1 \).

### Implication
- Ensures that the model has equal true and false positive rates for each group, aiming for fairness in terms of both error types.

### Summary
Separation demands that a model's accuracy is consistent across groups when it comes to correctly predicting positive and negative outcomes.

---## Fairness Criterion: Equal Opportunity

### Definition
- **Equal Opportunity** is a concept in fairness that suggests individuals should have an equal chance of being correctly classified for a positive outcome by a predictive model, regardless of their membership in a protected group.

### Formula
- Expressed as \( P(R = 1|A = 1, Y = 1) = P(R = 1|A = 0, Y = 1) \), ensuring that the true positive rate is the same across different groups.

### Example
- For two groups \( A=0 \) and \( A=1 \), if individuals from both groups actually belong to a positive class \( Y=1 \), they should have an equal probability \( P(R = 1) \) of being predicted as positive by the model.

### Goal
- To provide an equal probability of receiving a beneficial outcome when the actual, true condition warrants it, across all groups.

### Summary
Equal opportunity in machine learning aims for fairness by ensuring that all individuals have the same chance of receiving accurate positive predictions, offering equal treatment for the positive class across protected groups.

---
## Summary of Fairness Criteria

### Fairness Criteria in Machine Learning:
1. **Unawareness**
   - **Criteria**: Exclude sensitive attribute \( A \) in prediction.
   - **Goal**: To not use any sensitive attribute directly in the decision-making process.

2. **Demographic Parity**
   - **Criteria**: \( P(R = 1|A = 1) = P(R = 1|A = 0) \)
   - **Goal**: The decision should be independent of the sensitive attribute.

3. **Equality of Odds**
   - **Criteria**: \( P(R = 1|A = 1, Y) = P(R = 1|A = 0, Y) \)
   - **Goal**: The model's predictions should be independent of the sensitive attribute, given the true outcome.

4. **Equal Opportunity**
   - **Criteria**: \( P(R = 1|A = 1, Y = 1) = P(R = 1|A = 0, Y = 1) \)
   - **Goal**: The true positive rate should be the same across different groups defined by the sensitive attribute.

### Summary
These criteria are different approaches to defining and achieving fairness in machine learning models, each with a focus on how the sensitive attribute \( A \) is factored into the model's predictions \( R \), particularly in relation to the true outcome \( Y \).

---
## Disparity Metrics in Machine Learning

### Purpose
- Disparity metrics are used to quantify the extent to which a predictive model's outcomes differ across groups defined by sensitive attributes.

### Key Metrics
1. **Demographic Parity Difference/Ratio**
   - **Difference**: $\epsilon = P(R = 1|A = 1) - P(R = 1|A = 0)$
   - **Ratio**: $\rho = \frac{P(R = 1|A = 1)}{P(R = 1|A = 0)} - 1$

2. **Equal Opportunity Difference/Ratio**
   - Measures disparities in the true positive rates between groups.

3. **Average Odds Difference/Ratio**
   - Compares the average of the true positive rate and false positive rate between groups.

### Example Calculation
- For Demographic Parity, if the probability of positive prediction for \( A=1 \) is 0.6 and for \( A=0 \) is 0.4, then \( \epsilon = 0.2 \) and \( \rho = 0.5 \).

### Summary
Disparity metrics evaluate a model's fairness by calculating the difference and ratio of positive outcomes between protected and unprotected groups.

---
## Mitigation Strategy: Pre-processing for Fair Machine Learning

### Objective
- Implement strategies during the data pre-processing stage to minimize or eliminate bias in machine learning models.

### Strategies
1. **Feature Adjustment**
   - Adjust features so they are uncorrelated with the sensitive attribute.
   - Methods include "fair" representation learning and adversarial learning.

2. **Addressing Representation Bias**
   - Reweighing: Assign weights to examples in each (group, label) combination to balance representation.
   - Oversampling: Increase the representation of minority groups in the dataset.

3. **Data Augmentation**
   - Synthesize data for underrepresented groups.
   - Example: If data contains "he is a doctor," synthesize "she is a doctor" to balance gender representation.

### Reweighing Example
- To ensure fairness before classification, weights are calculated as the ratio of expected versus observed probabilities.
- If males receive loans 40% of the time and females 20%, but both should be equal (50%), reweighing adjusts the influence of each group's data points in the training process accordingly.

### Summary
Pre-processing mitigation strategies involve modifying the training data to ensure fair treatment across different groups by the resulting machine learning models.

---
## Mitigation Strategy: In-processing for Fair Machine Learning

### Objective
- Adjust the machine learning algorithm during the training phase to reduce bias.

### Strategies
1. **Separate Models**
   - Create a unique model for each protected group denoted by $A$ to customize decisions.

2. **Adversarial Debiasing**
   - Implement adversarial learning to optimize prediction accuracy while preventing the model from learning the protected attribute.
   - The system comprises two parts:
     - **Predictor**: Outputs $R = r(X)$ based on features $X$.
     - **Adversary**: Attempts to determine the protected attribute $A$ from the predictions $R$.
   - The objective is to minimize the loss function for the outcome $Y$ (denoted by $\text{Loss}(Y, R)$) and the loss related to the adversary's accuracy in predicting $A$ (denoted by $\text{Loss}(A, S)$).

### Example
- In a credit scoring model, the goal would be to accurately assess creditworthiness ($Y$) without allowing an adversary to infer the individual's sensitive attributes ($A$), like race or gender, from the credit score ($R$).

### Summary
In-processing is a mitigation technique applied during model training to directly embed fairness considerations, ensuring that the resulting models are fair regarding the sensitive attribute $A$.

---