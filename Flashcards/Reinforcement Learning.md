```table-of-contents
```

## Reinforcement Learning Overview

- **Definition**: Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by performing actions and receiving feedback in the form of rewards.
- **Goal**: The agent seeks to maximize cumulative reward over time, adapting its strategy based on trial-and-error and delayed rewards.
- **Learning Process**: Unlike supervised learning, in RL the agent is not given explicit instructions on the actions to take; it must discover the optimal actions through experience.
- **Key Characteristics**:
  - The ability to learn from the consequences of actions (trial-and-error).
  - Coping with delayed rewards (long-term benefits versus immediate gains).
- **Context**: RL is distinct from other machine learning approaches like supervised and unsupervised learning because it focuses on learning a policy of actions that maximizes reward through interactions with the environment.

---

## Sequential Decision Making

### Definition:
Sequential decision making in machine learning involves selecting actions to maximize the total future reward. This process often requires balancing the immediate reward against potential long-term benefits.

### Key Concepts:
- **Actions**: Choices made at each step to achieve a goal, considering their long-term consequences.
- **Reward**: The payoff or return received from taking a particular action, which may not be immediate.
- **Policy**: The strategy or rule that an agent follows to choose actions, aiming to maximize cumulative rewards over time.

### Strategy:
- **Exploitation**: Choosing the best-known action to maximize immediate reward.
- **Exploration**: Trying different actions to discover their rewards, which may lead to better long-term results.

### Formalization:
- **Action Value Function  $Q$**: Estimates the expected reward of an action at a given time, updated incrementally as new actions are taken and new rewards are observed.
- **Incremental Update Rule**: New estimates are computed by adjusting old estimates towards the target based on the step size and the observed reward.

### Decision Policies:
- **Greedy Action Selection**: Always choosing the current best action.
- **Epsilon Greedy**: Choosing the best action with probability $1 - \epsilon$ and a random action with probability $\epsilon$ for exploration.
- **SoftMax**: Selecting actions according to a probability distribution that favors better actions more but still allows for exploration.

---
## History and State in Reinforcement Learning

### History:
- **Definition**: The history is the sequence of observations, actions, and rewards up to the current time step $t$, denoted as $h_t = o_1, r_1, a_1, ..., a_{t-1}, o_t, r_t$.
- **Function**: It represents all the observable variables an agent has encountered up to time $t$.

### State:
- **Definition**: The state is the information that summarizes the history and is used to determine what happens next in the environment.
- **Formalization**: The state at time $t$, denoted as $S_t$, is a function of the history $h_t$, where $S_t = f(h_t)$.
- **Purpose**: The state guides the agent's decision on which action to select next.

### Observability:
- **Agent's Perspective**: The observability of the environment state refers to how well an agent can infer the state of the environment from the history it has observed.
- **Full Observability**: The agent has access to all the information that defines the environmental state.
- **Partial Observability**: The agent sees a limited or noisy version of the environmental state, which can make decision-making more complex.

---
## Reinforcement Learning (RL) Overview

### Reinforcement Learning (RL)
- **Definition**: RL is the process of learning the mapping from situations to actions to maximize the sum of rewards throughout the learner's life.
- **Policy π**: A policy defines the agent's action selection in a state, represented as π: $S \rightarrow A$.

### Characteristics of RL
- **Trial-and-Error**: Agents learn by trial and error, discovering actions that yield the most reward without explicit instruction.
- **Delayed Rewards**: Actions may impact not only the immediate reward but also future rewards.

### Terminology
- **State**: Information available to the agent about the environment.
- **Terminal State**: The end state with no further actions, leading to a reset.
- **Episode**: A sequence from the initial to the terminal state: $(s_0, a_0, r_0), (s_1, a_1, r_1), ..., (s_n, a_n, r_n)$.
- **Cumulative Reward**: The sum of rewards throughout an episode, with a discount factor γ (0 < γ ≤ 1): $R = \sum_{t=0}^{n} \gamma^t r_{t+1}$.

### RL Components
- **Policy π**: The agent's strategy to choose an action for each state, which can be deterministic or stochastic.
- **Optimal Policy π\***: The theoretical policy that maximizes the expectation of cumulative rewards.
- **Reward r**: A scalar signal indicating the agent's performance at step $t$.
- **Value Functions V(s)**: The value function \(V(s)\) represents the expected return (sum of rewards) starting from state \(s\), and then following a particular policy. It evaluates how good it is to be in a particular state.
- **State action Value Function Q(s, a)**: The action-value function \(Q(s, a)\) represents the expected return starting from state \(s\), taking action \(a\), and then following a specific policy. It evaluates how good it is to perform a certain action in a particular state.

---
## Optimal \(Q\) Function and the Bellman Equation

### The Optimal \(Q\) Function

- The optimal action-value function, denoted as $Q^*(s, a)$, gives the maximum expected return achievable for a state-action pair \(s, a\), under the best policy.

### The Bellman Equation for \(Q^*\)

- The Bellman equation for $Q^*$ is a fundamental principle in reinforcement learning that provides a recursive relationship for determining $Q^*$. It is given by:

  $$
  Q^*(s, a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'}Q^*(S_{t+1}, a') | S_t = s, A_t = a]
  $$

- Here, $R_{t+1}$ is the reward received after taking action $a$ in state \(s\), $\gamma$ is the discount factor, and $S_{t+1}$ and $a'$ represent the next state and any possible action in that state, respectively.

### Significance

- This equation reflects the idea that the value of a state-action pair \(s, a\) is the immediate reward $R_{t+1}$ plus the maximum discounted future value achievable from the next state $S_{t+1}$.
- It forms the basis for many algorithms in reinforcement learning that aim to find the optimal policy by learning $Q^*$, such as Q-learning and Deep Q-Networks (DQN).

---