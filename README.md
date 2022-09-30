# Transformers for Reinforcement Learning

This repo tracks literature and additional online resources on transformers for reinforcement learning. We provide a short summary of each paper. Though we have tried our best to include all relevant works, it's possible that we might have missed your work. Please feel free to create an issue if you want your work to be added.

While we were preparing this repo, we noticed the [Awesome-Reinforcement-Learning](https://github.com/opendilab/awesome-decision-transformer) repo that also covers decision transformer literature. Awesome-Reinforcement-Learning does not provide paper summaries but lists the experiment environment used in each paper. We believe both repos are helpful for beginners to get started on Transformers for RL. If you find these resources to be useful, please follow and star both repos!  

- [Transformers for Reinforcement Learning](#transformers-for-reinforcement-learning)
  - [Papers](#papers)
    - [Decision Transformer: Reinforcement Learning via Sequence Modeling](#decision-transformer-reinforcement-learning-via-sequence-modeling)
    - [Offline Reinforcement Learning as One Big Sequence Modeling Problem](#offline-reinforcement-learning-as-one-big-sequence-modeling-problem)
    - [Generalized Decision Transformer for Offline Hindsight Information Matching](#generalized-decision-transformer-for-offline-hindsight-information-matching)
    - [Online Decision Transformer](#online-decision-transformer)
    - [Prompting Decision Transformer for Few-Shot Policy Generalization](#prompting-decision-transformer-for-few-shot-policy-generalization)
    - [Addressing Optimism Bias in Sequence Modeling for Reinforcement Learning](#addressing-optimism-bias-in-sequence-modeling-for-reinforcement-learning)
    - [Can Wikipedia help offline reinforcement learning?](#can-wikipedia-help-offline-reinforcement-learning)
    - [A Generalist Agent](#a-generalist-agent)
    - [Bootstrapped Transformer for Offline Reinforcement Learning](#bootstrapped-transformer-for-offline-reinforcement-learning)
    - [Towards Flexible Inference in Sequential Decision Problems via Bidirectional Transformers](#towards-flexible-inference-in-sequential-decision-problems-via-bidirectional-transformers)
    - [Phasic Self-Imitative Reduction for Sparse-Reward Goal-Conditioned Reinforcement Learning](#phasic-self-imitative-reduction-for-sparse-reward-goal-conditioned-reinforcement-learning)
    - [Behavior Transformers: Cloning k modes with one stone](#behavior-transformers-cloning-k-modes-with-one-stone)
    - [Deep Reinforcement Learning with Swin Transformer](#deep-reinforcement-learning-with-swin-transformer)
    - [Efficient Planning in a Compact Latent Action Space](#efficient-planning-in-a-compact-latent-action-space)
    - [Going Beyond Linear Transformers with Recurrent Fast Weight Programmers](#going-beyond-linear-transformers-with-recurrent-fast-weight-programmers)
    - [GPT-critic: offline reinforcement learning for end-to-end task-oriented dialogue systems](#gpt-critic-offline-reinforcement-learning-for-end-to-end-task-oriented-dialogue-systems)
    - [Multi-Game Decision Transformers](#multi-game-decision-transformers)
    - [Offline pre-trained multi-agent decision transformer: one big sequence model tackles all smac tasks](#offline-pre-trained-multi-agent-decision-transformer-one-big-sequence-model-tackles-all-smac-tasks)
    - [Q-learning Decision Transformer: Leveraging Dynamic Programming for Conditional Sequence Modelling in Offline RL](#q-learning-decision-transformer-leveraging-dynamic-programming-for-conditional-sequence-modelling-in-offline-rl)
    - [StARformer: Transformer with State-Action-Reward Representations for Robot Learning](#starformer-transformer-with-state-action-reward-representations-for-robot-learning)
    - [Switch Trajectory Transformer with Distributional Value Approximation for Multi-Task Reinforcement Learning](#switch-trajectory-transformer-with-distributional-value-approximation-for-multi-task-reinforcement-learning)
    - [Transfer learning with causal counterfactual reasoning in Decision Transformers](#transfer-learning-with-causal-counterfactual-reasoning-in-decision-transformers)
    - [Transformers are Adaptable Task Planners](#transformers-are-adaptable-task-planners)
    - [Transformers are Meta-Reinforcement Learners](#transformers-are-meta-reinforcement-learners)
    - [Transformers are Sample Efficient World Models](#transformers-are-sample-efficient-world-models)
    - [You Can’t Count on Luck: Why Decision Transformers Fail in Stochastic Environments](#you-cant-count-on-luck-why-decision-transformers-fail-in-stochastic-environments)
    - [Hierarchical Decision Transformer](#hierarchical-decision-transformer)
    - [PACT: Perception-Action Causal Transformer for Autoregressive Robotics Pre-Training](#pact-perception-action-causal-transformer-for-autoregressive-robotics-pre-training)
    - [When does return-conditioned supervised learning work for offline reinforcement learning?](#when-does-return-conditioned-supervised-learning-work-for-offline-reinforcement-learning)
  - [Other Resources](#other-resources)
  - [License](#license)

## Papers

:warning: 09/29/2022: some paper summaries are short and we are still actively improving them.

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

**ICML'22** [[Paper]](https://arxiv.org/abs/2202.05607) [[Code]](https://github.com/facebookresearch/online-dt)
This work combines offline pretraining and online finetuning.

### Prompting Decision Transformer for Few-Shot Policy Generalization

**ICML'22** [[Paper]](https://arxiv.org/abs/2206.13499) [[Code]](https://github.com/mxu34/prompt-dt)  
The authors introduced prompt to DT for few-shot policy learning.  

### Addressing Optimism Bias in Sequence Modeling for Reinforcement Learning

**ICML'22** [[Paper]](https://arxiv.org/abs/2207.10295) [[Code]](https://github.com/avillaflor/SPLT-transformer)  

This work combines VAE and TT for policy learning in stochastic environment.  

### Can Wikipedia help offline reinforcement learning?

**arXiv** [[Paper]](https://arxiv.org/abs/2206.08569) [[Code]](https://github.com/machelreid/can-wikipedia-help-offline-rl)  

Training transformers on RL datasets from scratch could lead to slow convergence. This paper studies whether it’s possible to transfer knowledge from vision and language domains to offline RL tasks. The authors show that wikipedia pretraining can improve the convergence by 3-6x.  

### A Generalist Agent

**arXiv** [[Paper]](https://arxiv.org/abs/2205.06175)

A transformer-based RL agent (GATO) is trained on multi-modal data to perform robot manipulation, chat, play Atari games, caption images simultaneously. The agent will determine by itself what to output based on its context.

### Bootstrapped Transformer for Offline Reinforcement Learning

**arXiv** [[Paper]](https://arxiv.org/abs/2206.08569)  

To address the offline data limitation, this paper uses the learned dynamics model to generate data. It’s a data augmentation method. It uses trajectory transformer as the model.  

### Towards Flexible Inference in Sequential Decision Problems via Bidirectional Transformers

**ICLR'22 Generalizable Policy Learning in the Physical World Workshop** [[Paper]](https://arxiv.org/abs/2204.13326)  

Applied random masking to pretrain transformers for RL.

### Phasic Self-Imitative Reduction for Sparse-Reward Goal-Conditioned Reinforcement Learning

*ICML'22** [[Paper]](https://arxiv.org/abs/2204.13326)  

Alternating between online and offline learning for tackling sparse-reward goal-conditioned problems.  

### Behavior Transformers: Cloning k modes with one stone

**arXiv** [[Paper]](https://arxiv.org/abs/2206.11251) [[Code]](https://github.com/notmahi/bet)  

The authors proposed Behavior Transformer to model unlabeled demonstration data with multiple modes. It introduces action correction to predict multi-modal continuous actions.  

### Deep Reinforcement Learning with Swin Transformer

**arXiv** [[Paper]](https://arxiv.org/abs/2206.15269)  

This paper studies replacing the convolutional neural networks used in online RL with Swin Transformer and show that it leads to better performance.  

### Efficient Planning in a Compact Latent Action Space

**arXiv** [[Paper]](https://arxiv.org/abs/2208.10291)  

This work combines VQ-VAE with TT to allow efficient planning in the latent space.  

### Going Beyond Linear Transformers with Recurrent Fast Weight Programmers

**arXiv** [[Paper]](https://arxiv.org/abs/2106.06295)  

A new transformer architecture is proposed and experiments on RL show large improvement over LSTM in several Atari games.  

### GPT-critic: offline reinforcement learning for end-to-end task-oriented dialogue systems

**ICLR'22** [[Paper]](https://openreview.net/pdf?id=qaxhBG1UUaS)

GPT-2 trained in an offline RL manner for dialogue generation.  

### Multi-Game Decision Transformers

**arXiv** [[Paper]](https://arxiv.org/abs/2205.15241)  

Similar to GATO, this paper studies the applying a single transformer-based RL agent to play multiple games.  

### Offline pre-trained multi-agent decision transformer: one big sequence model tackles all smac tasks

**arXiv** [[Paper]](https://arxiv.org/abs/2112.02845)  

The authors studies offline pre-training and online finetuning in the MARL setting. The authors show that offline pretraining significantly improves sample efficiency.  

### Q-learning Decision Transformer: Leveraging Dynamic Programming for Conditional Sequence Modelling in Offline RL

**arXiv** [[Paper]](https://arxiv.org/abs/2206.11251)  

This works combined dynamics programming with decision transformer to introduce the stitching ability.  

### StARformer: Transformer with State-Action-Reward Representations for Robot Learning

**arXiv** [[Paper]](https://arxiv.org/abs/2209.03993)  

Proposed a transformer architecture for robot learning representations.  

### Switch Trajectory Transformer with Distributional Value Approximation for Multi-Task Reinforcement Learning

**arXiv** [[Paper]](https://arxiv.org/abs/2203.07413)  

Multi-task offline RL problems. The value function is modeled as a distribution.  

### Transfer learning with causal counterfactual reasoning in Decision Transformers

**arXiv** [[Paper]](https://arxiv.org/abs/2110.14355)  

Transfer a learned policy to a new environments.  

### Transformers are Adaptable Task Planners

**arXiv** [[Paper]](https://arxiv.org/abs/2207.02442)  

Prompt-based task planning.  

### Transformers are Meta-Reinforcement Learners

**arXiv** [[Paper]](https://arxiv.org/abs/2206.06614)  

Applied transformers for meta-RL.  

### Transformers are Sample Efficient World Models

**arXiv** [[Paper]](https://arxiv.org/abs/2209.00588)  

Use discrete autoencoder and Transformer to learn world models.  

### You Can’t Count on Luck: Why Decision Transformers Fail in Stochastic Environments

**arXiv** [[Paper]](https://arxiv.org/abs/2205.15967)  

Issues of transformers in stochastic environment. The proposed method learns to cluster trajectories and conditions on average cluster returns.  

### Hierarchical Decision Transformer

**arXiv** [[Paper]](https://arxiv.org/abs/2209.10447)  

A hierarchical decision transformer that has both a high-level and a low-level controller.  

### PACT: Perception-Action Causal Transformer for Autoregressive Robotics Pre-Training

**arXiv** [[Paper]](https://arxiv.org/abs/2209.11133)  

A generative transformer-based architecture for pretraining with robot data in a self-supervised manner.  

### When does return-conditioned supervised learning work for offline reinforcement learning?  

**arXiv** [[Paper]](https://arxiv.org/abs/2206.01079)

This paper provides a study of the capabilities and limitations of return-conditioned supervised learning for RL.  

## Other Resources  

- [Amazon Accessible RL SDK](https://github.com/awslabs/amazon-accessible-rl-sdk): an open-source Python package for sequential decision making with transformers.  
- [Stanford CS25: Decision Transformers Lecture](https://www.youtube.com/watch?v=w4Bw8WYL8Ps&list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM&index=4)  

## License  

This repo is released under Apache License 2.0.  
