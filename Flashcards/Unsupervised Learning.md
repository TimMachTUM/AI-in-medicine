```table-of-contents
```


## Definition

Unsupervised learning is a type of machine learning where algorithms infer patterns from a dataset without reference to known or labeled outcomes. Unlike supervised learning, where models are trained on data labeled with the correct answer, unsupervised learning algorithms must discover the structure and relationships within the data on their own.

Common unsupervised learning methods include clustering, where the algorithm groups data points with similar features into clusters (like K-means or hierarchical clustering), and dimensionality reduction, where algorithms like PCA (Principal Component Analysis) reduce the number of variables under consideration. Unsupervised learning is useful for exploratory analysis, discovering hidden patterns, or compressing the data.

---
## Dimensionality Reduction

### Basic Idea
Dimensionality reduction refers to the process of reducing the number of random variables under consideration in a dataset, by obtaining a set of principal variables. It can be achieved through various techniques that simplify the complexity of the data while preserving as much relevant information as possible.

### Motivation
- **Curse of Dimensionality**: High-dimensional spaces can lead to issues where many machine learning algorithms fail to perform well.
- **Noise Reduction**: Reducing dimensions can help remove noise and redundant features, improving the signal-to-noise ratio.
- **Computational Efficiency**: Less data means faster processing and less computational resources.
- **Visualisation**: It enables the visual representation of high-dimensional data in 2D or 3D.

### Applications
- **Data Visualization**: Tools like t-SNE and PCA are used to visualize high-dimensional data in two or three dimensions.
- **Feature Extraction**: Techniques such as PCA and Autoencoders are used to extract important features from raw data.
- **Data Compression**: Reducing the dataset size while maintaining its integrity for storage and efficient processing.
- **Improving Model Performance**: Simplifying the data without losing essential information can lead to better model performance.

#### Techniques
- **Principal Component Analysis (PCA)**: Identifies the principal components that maximize variance and projects the data along these components.
- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: Reduces dimensionality while keeping similar instances close and dissimilar instances apart.
- **Linear Discriminant Analysis (LDA)**: Finds the linear combinations of features that characterize or separate two or more classes of objects or events.
- **Autoencoders**: Neural networks designed to learn efficient representations of the input data, called encodings, for the purpose of dimensionality reduction. 

---
## Principal Component Analysis (PCA) Overview

### Understanding PCA
PCA is a statistical procedure that uses orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.

### Steps in PCA
- **Step 1:** Build the design matrix $X$.
- **Step 2:** Center the design matrix to obtain $X'$.
- **Step 3:** 
  - Variant 1: Decompose the $m \times m$ covariance matrix to find $U$ and $D$ via $UDV^T = SVD(X'X^T)$.
  - Variant 2: Decompose the $m \times n$ design matrix directly to find $U$ and $D$ via $UDV^T = SVD(X')$.

### Key Components
- $U = (u_1 \ldots u_m)$: principal directions.
- $D \triangleq diag(\lambda_i)$: 
  - Variant 1: variances along $u_i$'s.
  - Variant 2: standard deviations along $u_i$'s.

### Differences Between Variants
- **Variant 1:** Involves the decomposition of the covariance matrix of the centered data, which is a square matrix of dimensions $m \times m$. It's focused on the variance.
- **Variant 2:** Decomposes the actual centered design matrix, which may not be square, and is of dimensions $m \times n$. It deals with standard deviation, which is the square root of variance.

---
## Autoencoders Explained

### Concept
Autoencoders are a type of neural network used to learn efficient codings of unlabeled data. They are composed of an encoder and a decoder network with non-linearities.

### Structure
- **Encoder Network** $(E_\phi)$: Compresses the input $x \in \mathbb{R}^m$ into a lower-dimensional code $z \in \mathbb{R}^d$.
- **Decoder Network** $(D_\theta)$: Attempts to reconstruct the input from the code, producing $\hat{x} \approx x$.

### Reconstruction Loss
- The quality of the autoencoder's output is measured by the reconstruction loss $\mathcal{L}(x, \hat{x}) = ||x - \hat{x}||^2$.
- $\hat{x} = D_\theta(E_\phi(x))$ is the reconstructed input.

### Training
- **Forward Evaluation**: Passing the input through the encoder and decoder to get the reconstruction.
- **Backpropagation/Learning**: Adjusting the weights of the networks to minimize reconstruction loss.

### Preventing Identity Learning
To avoid trivial solutions like the identity map:
- Introduce a bottleneck: $dim(z) << dim(x)$ to force the network to learn efficient representations.
- Add regularization to the loss function: $\mathcal{L}_{\phi,\theta} + \lambda \cdot R(\phi, \theta)$, e.g., Regularized AE, Sparse AE, Contractive AE.
- Use dropout techniques: Randomly set a percentage of network activations to 0 during training.

### Use Cases
Autoencoders are useful for dimensionality reduction, denoising, or learning generative models of data.

---
## Denoising Autoencoders (DAEs) as Generative Models

### Overview
Denoising Autoencoders (DAEs) are an extension of autoencoders that are capable of acting as generative models to estimate the data generation process.

### Core Principle
- DAEs work by implicitly learning the conditional distribution $p(x|\hat{x})$ where $\hat{x}$ is a corrupted version of $x$.
- They apply a corruption process $C(\hat{x}|x)$ to the input data before encoding and attempt to recover the original data from this corrupted input during decoding.

### Generative Process Intuition
- Iteratively adding noise to a typical sample $x$ and denoising it, a DAE often reproduces the sample, thus learning the data distribution.

### Estimation Technique
- Utilizes Markov chain (MC) sampling by alternating between the denoising model and the corruption process to approximate $p(x)$.

### Challenges
- Markov chain convergence can be computationally expensive and challenging to assess for convergence accuracy.

### Note
Variational Autoencoders (VAEs) are now more commonly used as generative models due to their robustness and easier convergence.

---
## Variational Autoencoders (VAEs) Explained

### What are VAEs?
Variational Autoencoders (VAEs) are a type of generative model that use the framework of autoencoders for the efficient encoding of data into a latent space.

### Key Components
- **Encoder**: Maps input data to a probability distribution in latent space.
- **Decoder**: Reconstructs input data from sampled points in the latent space distribution.

### Probabilistic Approach
- VAEs assume that the data is generated by a random process involving an unobserved continuous random variable.
- The encoder learns the parameters (mean and variance) of this probabilistic model.

### Loss Function
- The loss function for VAEs consists of two terms:
  - Reconstruction loss, which encourages the decoded samples to match the original inputs.
  - Regularization term (KL divergence), which keeps the learned distribution close to a prior distribution, typically a Gaussian.

### Statistical Motivation

VAEs aim to minimize the Kullback-Leibler (KL) divergence between the true posterior $p(z|x)$ and the approximation $q(z|x)$, which is achieved by:

- Minimizing $\text{KL}(p(z|x), q(z|x))$,
- Equivalent to maximizing the expected log likelihood $\mathbb{E}_{q(z|x)} \log p(x|z)$ minus the KL divergence between $q(z|x)$ and the prior $p(z)$,
- This process ensures the encoder's output distribution $q(z|x)$ is similar to the prior distribution $p(z)$.

### Training Details

- The loss function $\mathcal{L}(\theta, \phi, x, z) = \mathbb{E}_{q_\phi(z|x)} \log p_\theta (x|z) - \text{KL}(q_\phi(z|x), p(z))$ balances reconstruction against divergence from the prior.
- A challenge arises as the network contains a sampling operator, which is non-differentiable and hence complicates backpropagation.

### Addressing the Sampling Problem

- The "reparameterization trick" is used to allow backpropagation through the sampling process by expressing the random variable $z$ as a deterministic function of the input and another independent random variable $\epsilon$.

### Advantages of VAEs
- They do not just compress data but learn the distribution of the data, allowing them to generate new data points similar to the original input data.
- VAEs can help in disentangling the underlying factors of variation in the data.

### Gaussian Prior
In the VAE framework, the prior over latent variables $p(z)$ is typically assumed to be a Gaussian distribution.

### Latent Variable Inference
Determining the posterior $q(z|x)$ involves estimating the mean $\mu$ and the standard deviation $\sigma$ of the latent Gaussian distribution.

### Neural Network Estimation
- A neural network is employed to map the input $x$ to the latent representation $z$ (encoding phase).
- Another neural network is used to map the latent variables $z$ back to the reconstructed input $x'$ (decoding phase).

### Importance of Gaussian Assumption
- The Gaussian assumption simplifies the computation of the KL divergence term in the VAE loss function.
- It allows the use of the reparameterization trick for backpropagation through stochastic nodes.

---
## Generative Adversarial Networks (GANs) and Wasserstein GANs

### GAN Fundamentals
- GANs consist of two neural networks, the generator $G$ and the discriminator $D$.
- $D(x)$ represents the probability that $x$ came from the real dataset rather than from $G$.
- $D$ is trained to maximize the probability of correctly classifying both real and fake data.
- $G$ is trained to create data that is indistinguishable from real data, minimizing $\log(1 - D(G(z)))$.

### Minimax Game
- The training involves a two-player minimax game with value function $V(G, D)$:
$$
\min_{G} \max_{D} V(G, D) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

### Wasserstein GANs
- Wasserstein GANs utilize Wasserstein loss, derived from the Wasserstein distance (Earth Mover's Distance).
- The discriminator $D$ learns to maximize the discrepancy between real and fake data, with a constraint on the gradient to be within a 1-Lipschitz function space.
- Weights of $D$ are clipped to prevent the gradient from becoming unbounded, addressing the issue of vanishing gradients.

### Wasserstein Loss
$$
\max_{\|D\|_L \leq 1} \mathbb{E}_{x \sim p_{data}(x)} [D(x)] - \mathbb{E}_{x \sim p_{model}(x)} [D(x)]
$$

### Notes
- GANs enable the generation of new, synthetic instances of data that are similar to a given real dataset.
- Wasserstein GANs represent an advancement in stabilizing GAN training and improving the quality of generated data.

---
## Evaluating GANs: Inception Score and Frechet Inception Distance

### GAN Evaluation Overview
Evaluating the quality of Generative Adversarial Networks (GANs) is crucial and challenging. Two popular measures are the Inception Score (IS) and the Frechet Inception Distance (FID).

### Inception Score (IS)
- **Purpose**: Measures the quality and diversity of generated images.
- **How it Works**: Uses the Inception model to classify generated images. High IS implies both clear and varied images.
- **Calculation**: 
  $$ \text{IS}(G) = \exp(\mathbb{E}_{x} [KL(p(y|x) || p(y))]) $$
  Where $p(y|x)$ is the conditional label distribution and $p(y)$ is the marginal label distribution.

### Frechet Inception Distance (FID)
- **Purpose**: Assesses the similarity between generated images and real images.
- **How it Works**: Compares the feature vectors of the real and generated images extracted by the Inception model.
- **Calculation**:
  $$ \text{FID}(x, g) = ||\mu_x - \mu_g||^2_2 + \text{Tr}(\Sigma_x + \Sigma_g - 2(\Sigma_x\Sigma_g)^{1/2}) $$
  Where $\mu_x, \Sigma_x$ and $\mu_g, \Sigma_g$ are the mean and covariance of the real and generated images' feature vectors, respectively.

### Key Points
- **Inception Score**: Good for measuring the clarity and diversity of images but does not account for the match with the real data distribution.
- **Frechet Inception Distance**: Considered more reliable as it compares the statistics of generated images directly with real images, reflecting both diversity and realism.

---
## Self-Supervised Learning Explained

### What is Self-Supervised Learning?
Self-supervised learning is a form of unsupervised learning where the data itself provides the supervision.

### Mechanism
- The algorithm generates its own labels from the input data, often by setting a prediction task based on the input data structure.
- Common tasks include predicting a part of the data given the rest (e.g., predicting the next word in a sentence).

### Advantages
- Does not require external labeling, thus can learn from a large corpus of unlabeled data.
- Helps in learning representations that can be useful for a wide range of tasks.

### Applications
- Natural language processing (e.g., word embeddings like Word2Vec).
- Computer vision (e.g., predicting missing parts of images).
- Any domain with abundant unlabeled data.

### Key Takeaways
- Self-supervised learning leverages the underlying structure of the data to learn useful representations.
- It bridges the gap between supervised and unsupervised learning, exploiting unlabeled data while avoiding the need for manually annotated labels.

---

## Context Restoration in Self-Supervised Learning

### Pretext Task: Context Restoration
- **Objective**: Enhance the model's understanding of image context and structure.
- **Method**: 
  - Randomly select two small patches in an image and swap their locations.
  - The process is repeated $T$ times to generate a set of altered images.
  - The model attempts to restore the original image by correctly repositioning the swapped patches.

### Importance
- **Intensity Distribution**: Remains unchanged during the swapping, ensuring that the model cannot rely on pixel intensity for restoration.
- **Spatial Information**: Is altered, which challenges the model to understand the correct context and positioning of different image parts.

### Goal
The task forces the model to learn robust and high-level feature representations by focusing on the spatial relationships within the data, which is critical in understanding image content, especially in medical imaging.

### Application
Such a self-supervised learning task is particularly useful in medical image analysis where labeled data can be scarce or expensive to obtain.

---
## Contrastive Learning Explained

### What is Contrastive Learning?
Contrastive Learning is a self-supervised technique that learns to encode similar items close together and dissimilar items further apart in a feature space.

### Principle
- The approach relies on comparing pairs or groups of samples to learn representations.
- Typically involves a "positive" pair (similar items) and "negative" pairs (dissimilar items).

### Methodology
- A neural network is used to generate embeddings for each item in a pair or group.
- A contrastive loss function, like Noise Contrastive Estimation (NCE) or Triplet Loss, is used to adjust the embeddings.


---
## Pairwise and Triplet Contrastive Loss in Contrastive Learning

### Pairwise Contrastive Loss
- **Objective**: Differentiate between similar (positive) and dissimilar (negative) pairs of images.
- **Computation**: Calculate the distance $D_i = ||x, x_i||_2$ between two images.
- **Loss for Positive Pair**: $L_i = D_i^2$, penalizing large distances between similar images.
- **Loss for Negative Pair**: $L_i = max(0, \epsilon - D_i)^2$, penalizing small distances between dissimilar images, where $\epsilon$ is a margin.

### Triplet Loss
- **Components**: Consists of an anchor image, a positive image (similar to the anchor), and a negative image (dissimilar to the anchor).
- **Distances**:
  - Positive Distance $D_{pos} = ||x, x_{pos}||_2$,
  - Negative Distance $D_{neg} = ||x, x_{neg}||_2$.
- **Loss**: $L = max(0, D_{pos}^2 - D_{neg}^2 - \text{margin})$, where the margin is a hyperparameter that defines the desired separation between positive and negative pairs.

### Visual Representation
- The illustrations show how the embedding space is structured, with similar images being pulled closer and dissimilar images being pushed apart.
- The goal is to have a clear margin of separation between the positive and negative examples relative to the anchor.

### Importance
- These losses are fundamental in training models to understand and differentiate between the nuances of data, especially in image recognition tasks.

---

## InfoNCE Loss in Contrastive Learning

### What is InfoNCE Loss?
InfoNCE loss is a contrastive loss function used to measure the similarity between different representations of data, typically in self-supervised learning tasks.

### Calculation of InfoNCE Loss
The loss is computed as follows:
$$
\mathcal{L} = -\mathbb{E}_x \left[ \log \frac{\exp(s(f(x), f(x^+)))}{\exp(s(f(x), f(x^+))) + \sum_{j=1}^{N-1} \exp(s(f(x), f(x_j^-)))} \right]
$$
Where:
- $s(f_1, f_2) = \frac{f_1^T f_2}{\|f_1\| \|f_2\|}$ computes the cosine similarity between two feature vectors.
- $f(x)$ is the feature representation of the input.
- $f(x^+)$ is the feature representation of a positive sample, similar to the input.
- $f(x_j^-)$ is the feature representation of a negative sample, different from the input.
- $N$ is the number of negative samples.

### Purpose
- This loss function aims to maximize the mutual information between the feature representations of positive pairs while minimizing it for negative pairs.
- The negative part of InfoNCE loss serves as a lower bound on the mutual information between the representations of $f(x)$ and $f(x^+)$.

### Implications
- Maximizing mutual information encourages the model to learn features that capture important factors of variation in the data.
- The larger the set of negative samples ($N$), the tighter the lower bound on the mutual information, providing stronger training signals to the model.

---
## Momentum Contrast (MoCo) for Self-Supervised Learning

### What is Momentum Contrast (MoCo)?
MoCo is a method for self-supervised learning on visual data that builds dynamic dictionaries for contrastive loss.

### MoCo Mechanism
- **Query Encoder**: Generates a feature vector (query $q$) from the input image ($x_{query}$).
- **Key Encoder**: Generates feature vectors (keys $k_0, k_1, k_2, \ldots$) from augmented or different images.
- **Queue**: A memory bank stores a large number of keys to provide negative samples for contrastive learning.
- **Contrastive Loss**: Calculates similarity between the query and keys. Gradients are backpropagated only through the query encoder.

### Differences to SimCLR
- Unlike SimCLR, MoCo uses a queue (memory bank) to store the keys of negative samples, allowing for a larger and more consistent set of negatives.
- The gradients are not propagated through the key encoder; instead, it is updated using a momentum-based approach.

### Momentum Update Rule
To maintain consistency in the queue, the parameters of the key encoder ($\theta_k$) are updated with the momentum update rule:
$$
\theta_k = \beta\theta_k + (1 - \beta)\theta_q
$$
Where:
- $\beta$ is the momentum coefficient.
- $\theta_q$ are the parameters of the query encoder.

### Benefits of MoCo
- The momentum encoder allows for a smoother and more consistent update of the key representations, leading to stable training.
- The large queue enables the use of many negative samples, which is crucial for contrastive learning's effectiveness.

---