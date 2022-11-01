# Awesome Transformers for Reinforcement Learning  

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repo tracks literature and additional online resources on transformers for reinforcement learning. We provide a short summary of each paper. Though we have tried our best to include all relevant works, it's possible that we might have missed your work. Please feel free to create an issue if you want your work to be added.

While we were preparing this repo, we noticed the [Awesome-Decision-Transformer](https://github.com/opendilab/awesome-decision-transformer) repo that also covers decision transformer literature. Awesome-Reinforcement-Learning does not provide paper summaries but lists the experiment environment used in each paper. We believe both repos are helpful for beginners to get started on Transformers for RL. If you find these resources to be useful, please follow and star both repos!  

- [Awesome Transformers for Reinforcement Learning](#awesome-transformers-for-reinforcement-learning)
  - [Papers](#papers)
    - [Stabilizing Transformers for Reinforcement Learning](#stabilizing-transformers-for-reinforcement-learning)
    - [Decision Transformer: Reinforcement Learning via Sequence Modeling](#decision-transformer-reinforcement-learning-via-sequence-modeling)
    - [Offline Reinforcement Learning as One Big Sequence Modeling Problem](#offline-reinforcement-learning-as-one-big-sequence-modeling-problem)
    - [Efficient Transformers in Reinforcement Learning using Actor-Learner Distillation](#efficient-transformers-in-reinforcement-learning-using-actor-learner-distillation)
    - [Generalized Decision Transformer for Offline Hindsight Information Matching](#generalized-decision-transformer-for-offline-hindsight-information-matching)
    - [RvS: what is essential for offline RL via supervised learning?](#rvs-what-is-essential-for-offline-rl-via-supervised-learning)
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
    - [Deep Transformer Q-Networks for Partially Observable Reinforcement Learning](#deep-transformer-q-networks-for-partially-observable-reinforcement-learning)
    - [Contextual transformer for offline reinforcement learning](#contextual-transformer-for-offline-reinforcement-learning)
    - [MCTransformer: combining transformers and monte-carlo tree search for offline reinforcement learning](#mctransformer-combining-transformers-and-monte-carlo-tree-search-for-offline-reinforcement-learning)
    - [Pretraining the vision transformer using self-supervised methods for vision based deep reinforcement learning](#pretraining-the-vision-transformer-using-self-supervised-methods-for-vision-based-deep-reinforcement-learning)
    - [Preference Transformer: Modeling Human Preferences using Transformers for RL](#preference-transformer-modeling-human-preferences-using-transformers-for-rl)
    - [Skill discovery decision transformer](#skill-discovery-decision-transformer)
    - [Decision transformer under random frame dropping](#decision-transformer-under-random-frame-dropping)
    - [Token turing machines](#token-turing-machines)
    - [SMART: self-supervised multi-task pretraining with control transformers](#smart-self-supervised-multi-task-pretraining-with-control-transformers)
    - [Hyper-decision transformer for efficient online policy adaptation](#hyper-decision-transformer-for-efficient-online-policy-adaptation)
    - [Multi-agent multi-game entity transformer](#multi-agent-multi-game-entity-transformer)
    - [Evaluating Vision Transformer Methods for Deep Reinforcement Learning from Pixels](#evaluating-vision-transformer-methods-for-deep-reinforcement-learning-from-pixels)
    - [Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos](#video-pretraining-vpt-learning-to-act-by-watching-unlabeled-online-videos)
    - [Behavior Cloned Transformers are Neurosymbolic Reasoners](#behavior-cloned-transformers-are-neurosymbolic-reasoners)
    - [Exploiting Transformer in Reinforcement Learning for Interpretable Temporal Logic Motion Planning](#exploiting-transformer-in-reinforcement-learning-for-interpretable-temporal-logic-motion-planning)
    - [Transformers for One-Shot Visual Imitation](#transformers-for-one-shot-visual-imitation)
    - [Self-Attentional Credit Assignment for Transfer in Reinforcement Learning](#self-attentional-credit-assignment-for-transfer-in-reinforcement-learning)
  - [Other Resources](#other-resources)
  - [License](#license)

## Papers

:warning: 09/29/2022: some paper summaries are short and we are still actively improving them.

### Stabilizing Transformers for Reinforcement Learning

**ICML'20** [[Paper]](http://proceedings.mlr.press/v119/parisotto20a/parisotto20a.pdf)  

One of the first works succssfully applying transformers in the RL settings. This work aims to replace LSTM used in online RL with Transformers. The authors observed that training large-scale transformers in RL settings is unstable. Thus they proposed the Gate Transformer-XL architecture and showed that the novel architecture outperformed LSTMs in the DMLab-30 benchmark with a good training stability.  

--------

### Decision Transformer: Reinforcement Learning via Sequence Modeling

**NeurIPS'21** [[Paper]](https://arxiv.org/abs/2106.01345) [[Code]](https://github.com/kzl/decision-transformer)  
A seminal work that proposed a supervised learning framework based on transformers for sequential decision making tasks. It tackles RL as a sequence generation task. Given a pre-collected sequence decision making dataset, the Decision Transformer (DT) is trained to generated the action sequence that can lead to the expected return-to-go, which is used as the input to the transformer model.  

--------

### Offline Reinforcement Learning as One Big Sequence Modeling Problem

**NeurIPS'21** [[Paper]](https://arxiv.org/abs/2106.02039) [[Code]](https://github.com/JannerM/trajectory-transformer)  
This is another seminal work on applying transformers to RL and it was concurrent to Decision Transformer. The authors proposed Trajectory Transformer (TT) that combines transformers and beam search as a model-based approach for offline RL.  

--------

### Efficient Transformers in Reinforcement Learning using Actor-Learner Distillation

**ICLR'21** [[Paper]](https://openreview.net/forum?id=uR9LaO_QxF)
This paper introduces a distillation procedure that transfers learning progress from a large capacity learner model to a small capacity actor model. The proposed method can reduce the inference latency of the deployed RL agent.  

--------

### Generalized Decision Transformer for Offline Hindsight Information Matching

**ICLR'22** [[Paper]](https://arxiv.org/abs/2111.10364) [[Code]](https://github.com/frt03/generalized_dt)  
The paper derived a RL problem formulation called Hindsight Information Matching (HIM) from many recently proposed RL algorithms that use future trajectory information to accelerate the learning of a conditional policy. The authors discussed three HIM variations including Generalized DT, Categorical DT, and Bi-Directional DT.  

--------

### RvS: what is essential for offline RL via supervised learning?

**ICLR'22** [[Paper]](https://arxiv.org/pdf/2112.10751.pdf) [[Code]](https://github.com/scottemmons/rvs)  

DT solves reinforcement learning through supervised learning. It was hypothesized that the large model capacity of transformers could lead to better policies. The authors of this paper challenged the hypothesis and showed that a simple two-layer feedforward MLP led to similar performance with transformer-based methods. The findings of this paper imply that current designs of transformer-based reinforcement learning algorithms may not fully leverage the potential advantages of transformers.  

--------

### Online Decision Transformer

**ICML'22** [[Paper]](https://arxiv.org/abs/2202.05607) [[Code]](https://github.com/facebookresearch/online-dt)  

This work combines offline pretraining and online finetuning.  

--------

### Prompting Decision Transformer for Few-Shot Policy Generalization

**ICML'22** [[Paper]](https://arxiv.org/abs/2206.13499) [[Code]](https://github.com/mxu34/prompt-dt)  

The authors introduced prompt to DT for few-shot policy learning.  

--------

### Addressing Optimism Bias in Sequence Modeling for Reinforcement Learning

**ICML'22** [[Paper]](https://arxiv.org/abs/2207.10295) [[Code]](https://github.com/avillaflor/SPLT-transformer)  

This work combines VAE and TT for policy learning in stochastic environment.  

--------

### Can Wikipedia help offline reinforcement learning?

**arXiv** [[Paper]](https://arxiv.org/abs/2206.08569) [[Code]](https://github.com/machelreid/can-wikipedia-help-offline-rl)  

Training transformers on RL datasets from scratch could lead to slow convergence. This paper studies whether it’s possible to transfer knowledge from vision and language domains to offline RL tasks. The authors show that wikipedia pretraining can improve the convergence by 3-6x.  

--------

### A Generalist Agent

**arXiv** [[Paper]](https://arxiv.org/abs/2205.06175)

A transformer-based RL agent (GATO) is trained on multi-modal data to perform robot manipulation, chat, play Atari games, caption images simultaneously. The agent will determine by itself what to output based on its context.  

--------

### Bootstrapped Transformer for Offline Reinforcement Learning

**arXiv** [[Paper]](https://arxiv.org/abs/2206.08569)  

To address the offline data limitation, this paper uses the learned dynamics model to generate data. It’s a data augmentation method. It uses trajectory transformer as the model.  

--------

### Towards Flexible Inference in Sequential Decision Problems via Bidirectional Transformers

**ICLR'22 Generalizable Policy Learning in the Physical World Workshop** [[Paper]](https://arxiv.org/abs/2204.13326)  

Applied random masking to pretrain transformers for RL.  

--------

### Phasic Self-Imitative Reduction for Sparse-Reward Goal-Conditioned Reinforcement Learning

**ICML'22** [[Paper]](https://arxiv.org/abs/2204.13326)  

This work combines online RL and offline SL. The online phase is used for both RL training and data collection. In the offline phase, only successful trajectories are used for SL. The authors show that this approach performs well in sparse-reward settings. The authors tested DT for the SL phase and found that it was brittle and performed worse than a simple BC. This result show that the DT training stability requires more research. 

--------

### Behavior Transformers: Cloning k modes with one stone

**arXiv** [[Paper]](https://arxiv.org/abs/2206.11251) [[Code]](https://github.com/notmahi/bet)  

The authors proposed Behavior Transformer to model unlabeled demonstration data with multiple modes. It introduces action correction to predict multi-modal continuous actions.  

--------

### Deep Reinforcement Learning with Swin Transformer

**arXiv** [[Paper]](https://arxiv.org/abs/2206.15269)  

This paper studies replacing the convolutional neural networks used in online RL with Swin Transformer and show that it leads to better performance.  

--------

### Efficient Planning in a Compact Latent Action Space

**arXiv** [[Paper]](https://arxiv.org/abs/2208.10291)  

This work combines VQ-VAE with TT to allow efficient planning in the latent space.  

--------

### Going Beyond Linear Transformers with Recurrent Fast Weight Programmers

**arXiv** [[Paper]](https://arxiv.org/abs/2106.06295)  

A new transformer architecture is proposed and experiments on RL show large improvement over LSTM in several Atari games.  

--------

### GPT-critic: offline reinforcement learning for end-to-end task-oriented dialogue systems

**ICLR'22** [[Paper]](https://openreview.net/pdf?id=qaxhBG1UUaS)

GPT-2 trained in an offline RL manner for dialogue generation.  

--------

### Multi-Game Decision Transformers

**arXiv** [[Paper]](https://arxiv.org/abs/2205.15241)  

Similar to GATO, this paper studies the applying a single transformer-based RL agent to play multiple games.  

--------

### Offline pre-trained multi-agent decision transformer: one big sequence model tackles all smac tasks

**arXiv** [[Paper]](https://arxiv.org/abs/2112.02845)  

The authors studies offline pre-training and online finetuning in the MARL setting. The authors show that offline pretraining significantly improves sample efficiency.  

--------

### Q-learning Decision Transformer: Leveraging Dynamic Programming for Conditional Sequence Modelling in Offline RL

**arXiv** [[Paper]](https://arxiv.org/abs/2206.11251)  

The original DT completely requires on supervised learning to learn a value-conditioned behavior policy. By increasing the condition value, DT could obtain greater returns than the maximum return in the offline dataset. However, the RCSL framework does not tap into trajectory stitching, i.e., combining sub-trajectories of multiple sub-optimal trajectories to obtain an optimal trajectory. In this paper, the authors combine Q-learning and DT. The estimated Q-values are used to relabel the return-to-gos in the training data.  

--------

### StARformer: Transformer with State-Action-Reward Representations for Robot Learning

**arXiv** [[Paper]](https://arxiv.org/abs/2209.03993)  

Proposed a transformer architecture for robot learning representations.  

--------

### Switch Trajectory Transformer with Distributional Value Approximation for Multi-Task Reinforcement Learning

**arXiv** [[Paper]](https://arxiv.org/abs/2203.07413)  

Multi-task offline RL problems. The value function is modeled as a distribution.  

--------

### Transfer learning with causal counterfactual reasoning in Decision Transformers

**arXiv** [[Paper]](https://arxiv.org/abs/2110.14355)  

The authors leverage the casual knowledge of a source environment's structure to generate a set of counterfactual environments to improve the agent's adaptability in new environments. 

--------

### Transformers are Adaptable Task Planners

**arXiv** [[Paper]](https://arxiv.org/abs/2207.02442)  

Prompt-based task planning.  

--------

### Transformers are Meta-Reinforcement Learners

**arXiv** [[Paper]](https://arxiv.org/abs/2206.06614)  

Applied transformers for meta-RL.  

--------

### Transformers are Sample Efficient World Models

**arXiv** [[Paper]](https://arxiv.org/abs/2209.00588)  [[Code]](https://github.com/eloialonso/iris)  

With the goal of improving sample efficiency of RL methods, the authors build a transformer model on Atari environments. They borrowed ideas from VQGAN and DALL-E to map raw image pixels to a much smaller amount of image tokens, which are used as the input to autoregressive transformers. After training the transformer world model, the RL agents then learns exclusively from the model imaginations.  

--------

### You Can’t Count on Luck: Why Decision Transformers Fail in Stochastic Environments

**arXiv** [[Paper]](https://arxiv.org/abs/2205.15967)  

Issues of transformers in stochastic environment. The proposed method learns to cluster trajectories and conditions on average cluster returns.  

--------

### Hierarchical Decision Transformer

**arXiv** [[Paper]](https://arxiv.org/abs/2209.10447)  

The original DT highly depends on a carefully chosen return-to-go as the initial input to condition on. To address this challenge, this work proposed to predict subgoals (or options) to replace the return-to-go. Two transformers are trained together while one is used for predicting the subgoals and the the other one is used to predict actions conditioned on the subgoals. Through experiments on D4RL, the authors show that this hierarchical approach can outperform the original DT, especially in tasks that invovle long episodes.  

--------

### PACT: Perception-Action Causal Transformer for Autoregressive Robotics Pre-Training

**arXiv** [[Paper]](https://arxiv.org/abs/2209.11133)  

A generative transformer-based architecture for pretraining with robot data in a self-supervised manner.  

--------

### When does return-conditioned supervised learning work for offline reinforcement learning?  

**arXiv** [[Paper]](https://arxiv.org/abs/2206.01079)

Focusing on offline reinforcement learning, the authors provide a study on the capabilities and limitations of return-conditioned supervised learning (RCSL). The authors found that RCSL requires assumptions stronger than the dynamic programming to return optimal policies. Specifically, the authors pointed out that RCSL requires nearly deterministic dynamics and proper condition values. The authors claim that RCSL alone is unlikely to be a general solution for offline RL problems. However, it may perform well with high quality behavior data.  

--------

### Deep Transformer Q-Networks for Partially Observable Reinforcement Learning

**OpenReview Submission to ICLR'23** [[Paper]](https://openreview.net/forum?id=cddqs4kvC20)

Recurrent neural networks are often used for encoding an agent's history when solving POMDP tasks. This paper proposed to replace the recurrent neural networks with transformers. Results show that transformers can solve POMDP faster and more stably than methods based on recurrent neural networks. 

--------

### Contextual transformer for offline reinforcement learning

**OpenReview Submission to ICLR'23** [[Paper]](https://openreview.net/pdf?id=7pl0FRiS0Td)  

This paper proposed an approach for learning context vectors that can be used as prompts for the transformers. With the prompts, the authors developed a contextual meta transformers that can leverage the prompt as the task context to improve he performance on unseen tasks.  

--------

### MCTransformer: combining transformers and monte-carlo tree search for offline reinforcement learning

**OpenReview Submission to ICLR'23** [[Paper]](https://openreview.net/pdf?id=-94tJCOo7OM)  

The authors combine transformers and MCTS for efficient online finetuning. MCTS is used as an effective approach to balance exploration and exploitation.  

--------

### Pretraining the vision transformer using self-supervised methods for vision based deep reinforcement learning

**OpenReview Submission to ICLR'23** [[Paper]](https://openreview.net/pdf?id=CEhy-i7_KfC)  

This work replaces CNNs used in image-based RL agents with pre-trained Vision Transformers. Interestingly, the authors found Vision Transformers still perform similarly or worse than CNNs.  

--------

### Preference Transformer: Modeling Human Preferences using Transformers for RL

**OpenReview Submission to ICLR'23** [[Paper]](https://openreview.net/forum?id=Peot1SFDX0)

--------

### Skill discovery decision transformer

**OpenReview Submission to ICLR'23** [[Paper]](https://openreview.net/pdf?id=mb7PtrUbHa)  

This work applies unsupervised skill discovery to DT. The skill embedding is used as an input to the DT. This can be thought as a hierarchical RL approach. 

--------

### Decision transformer under random frame dropping

**OpenReview Submission to ICLR'23** [[Paper]](https://openreview.net/pdf?id=NmZXv4467ai)

--------

### Token turing machines

**OpenReview Submission to ICLR'23** [[Paper]](https://openreview.net/pdf?id=3m_awcLrg8E)

--------

### SMART: self-supervised multi-task pretraining with control transformers

**OpenReview Submission to ICLR'23** [[Paper]](https://openreview.net/pdf?id=9piH3Hg8QEf)

--------

### Hyper-decision transformer for efficient online policy adaptation

**OpenReview Submission to ICLR'23** [[Paper]](https://openreview.net/pdf?id=AatUEvC-Wjv)  

This work focuses on adapting DT to unseen novel tasks. An adaptation module is added to the DT with its parameters initialized by a hyper-network. When adapting to a new task, only the parameters of the adaptation module is finetuned. The results show that adapting the module leads to faster learning than the 

--------

### Multi-agent multi-game entity transformer

**OpenReview Submission to ICLR'23** [[Paper]](https://openreview.net/pdf?id=cytNlkyjWOq)

--------

### Evaluating Vision Transformer Methods for Deep Reinforcement Learning from Pixels

**arXiv** [[Paper]](https://arxiv.org/pdf/2204.04905.pdf)

The authors compared ViTs and CNNs in image-based DRL tasks. They found that CNNs still perform better than ViTs.  

--------

### Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos

**arXiv** [[Paper]](https://arxiv.org/pdf/2206.11795.pdf)

The authors apply semi-supervised imitation learning to enable agents to learn to act by watching online unlabeled videos.  

--------

### Behavior Cloned Transformers are Neurosymbolic Reasoners

**arXiv** [[Paper]](https://arxiv.org/pdf/2210.07382.pdf)

--------

### Exploiting Transformer in Reinforcement Learning for Interpretable Temporal Logic Motion Planning

**arXiv** [[Paper]](https://arxiv.org/pdf/2209.13220.pdf)

--------

### Transformers for One-Shot Visual Imitation

**CoRL** [[paper]](https://proceedings.mlr.press/v155/dasari21a.html)

--------

### Self-Attentional Credit Assignment for Transfer in Reinforcement Learning

**arXiv** [[paper]](https://arxiv.org/pdf/1907.08027.pdf)

--------

## Other Resources  

- [Amazon Accessible RL SDK](https://github.com/awslabs/amazon-accessible-rl-sdk): an open-source Python package for sequential decision making with transformers.  
- [Stanford CS25: Decision Transformers Lecture](https://www.youtube.com/watch?v=w4Bw8WYL8Ps&list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM&index=4)  
- Benchmark Environments
  - [D4RL](https://github.com/Farama-Foundation/D4RL)
  - [MuJoco](https://mujoco.org/)
  - [DM-Control](https://github.com/deepmind/dm_control)
  - [CARLA](https://leaderboard.carla.org/)
  - [SMAC - StarCraft Multi-Agent Challenge](https://github.com/oxwhirl/smac)
  - [MiniGrid](https://github.com/Farama-Foundation/Minigrid)
  - [C-MAPSS Aricraft Engine Simulator Data](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq)

## License  

This repo is released under Apache License 2.0.  
