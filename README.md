# Transformers for Reinforcement Learning

This repo summarizes literature and additional online resources on transformers for reinforcement learning. Though we are trying our best to include all relevant works, it's possible that we might have missed your work. Please feel free to create an issue if you want your work to be added.

## Papers

### Decision Transformer: Reinforcement Learning via Sequence Modeling

**NeurIPS'21** [[Paper]](https://arxiv.org/abs/2106.01345) [[Code]](https://github.com/kzl/decision-transformer)
A seminal work that proposed a supervised learning framework based on transformers for sequential decision making tasks. It tackles RL as a sequence generation task. Given a pre-collected sequence decision making dataset, the Decision Transformer (DT) is trained to generated the action sequence that can lead to the expected return-to-go, which is used as the input to the transformer model.

### Offline Reinforcement Learning as One Big Sequence Modeling Problem

**NeurIPS'21** [[Paper]](https://arxiv.org/abs/2106.02039) [[Code]](https://github.com/JannerM/trajectory-transformer)
This is another seminal work on applying transformers to RL and it was concurrent to Decision Transformer. The authors proposed Trajectory Transformer (TT) that combines transformers and beam search as a model-based approach for offline RL.

### Generalized Decision Transformer for Offline Hindsight Information Matching

**ICLR'22** [[Paper]](https://arxiv.org/abs/2111.10364) [[Code]](https://github.com/frt03/generalized_dt)
The paper derived a RL problem formulation called Hindsight Information Matching (HIM) from many recently proposed RL algorithms that use future trajectory information to accelerate the learning of a conditional policy. The authors discussed three HIM variations including Generalized DT, Categorical DT, and Bi-Directional DT.

### Online Decision Transformer

**ICML'22** [[Paper]](https://arxiv.org/abs/2202.05607) [[Code]](https://github.com/frt03/generalized_dt)
This is another seminal work on applying transformers to RL and it was concurrent to Decision Transformer. The authors proposed Trajectory Transformer (TT) that combines transformers and beam search as a model-based approach for offline RL.


### Prompting

**ICLR'22** [[Paper]](https://arxiv.org/abs/2111.10364) [[Code]](https://github.com/frt03/generalized_dt)
This is another seminal work on applying transformers to RL and it was concurrent to Decision Transformer. The authors proposed Trajectory Transformer (TT) that combines transformers and beam search as a model-based approach for offline RL.

### Addressing Optimism Bias in Sequence Modeling for Reinforcement Learning



## A Generalist Agent**
arXiv [[Paper]](https://arxiv.org/abs/2111.10364) [[Code]](https://github.com/frt03/generalized_dt)

**Bootstrapped Transformer for Offline Reinforcement Learning**
asdf

**Can Wikipedia help offline reinforcement learning?**

**Behavior Transformers: Cloning k modes with one stone**

**Deep Reinforcement Learning with Swin Transformer**

**Efficient Planning in a Compact Latent Action Space**

**Going Beyond Linear Transformers with Recurrent Fast Weight Programmers**

**GPT-critic: offline reinforcement learning for end-to-end task-oriented dialogue systems**

**Multi-Game Decision Transformers**

**Offline pre-trained multi-agent decision transformer: one big sequence model tackles all smac tasks**

**Q-learning Decision Transformer: Leveraging Dynamic Programming for Conditional Sequence Modelling in Offline RL**

**StARformer: Transformer with State-Action-Reward Representations for Robot Learning**

**Switch Trajectory Transformer with Distributional Value Approximation for Multi-Task Reinforcement Learning**

**Transfer learning with causal counterfactual reasoning in Decision Transformers**

**Transformers are Adaptable Task Planners**

**Transformers are Meta-Reinforcement Learners**

**Transformers are Sample Efficient World Models**

**You Can’t Count on Luck: Why Decision Transformers Fail in Stochastic Environments**

**Hierarchical Decision Transformer**

**PACT: Perception-Action Causal Transformer for Autoregressive Robotics Pre-Training**

**Towards Flexible Inference in Sequential Decision Problems via Bidirectional Transformers**

**Phasic Self-Imitative Reduction for Sparse-Reward Goal-Conditioned Reinforcement Learning**

**When does return-conditioned supervised learning work for offline reinforcement learning?**

## Other Resources