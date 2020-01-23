---
layout: page-rl-theory-practice
title: 'Key Concepts of Modern Reinforcement Learning'
permalink: key-concepts-of-modern-reinforcement-learning/
---

The fundamental level of a reinforcement learning setting consists of an Agent interacting with an Environment in a feedback loop. The Agent selects an action for each state at time $$s_t$$ of the Environment based on the response that it received from the Environment in the previous state at time $$s_{t-1}$$. From this basic setup, we can already identify two principal components in a reinforcement learning setting, which is the <span style="font-weight:bolder;">Agent</span> and the <span style="font-weight:bolder;">Environment</span>.

<div class="fig figcenter fighighlight">
     <img src="/assets/rl_theory_practice/rl-expanded-agent-environment-interaction.png"> 
     <div class="figcaption" style="text-align: center;">
        <span style="font-weight:bolder;">A recursive representation of the Agent-Environment interface.</span> At time step <code>t</code>, the Agent receives an instance of the Environment state <code>s_t</code>. The Agent then selects an action <code>a_t</code> from the set of actions available in state <code>s_t</code>. In the next iteration, the Agent receives a new state instance <code>s_{t+1}</code> and an immediate reward <code>r_{t+1}</code> based on the action <code>a_t</code> taken in the previous state <code>s_t</code>.
     </div>
</div>

As the Agent interacts with the Environment, it learns a <span style="font-weight:bolder;">policy</span>. A policy is a "learned strategy" that governs the agents' behaviour in selecting an action at a particular time $$t$$ of the Environment. A policy can be seen as a mapping from states of an Environment to the actions taken in those states.

The goal of the reinforcement Agent is to maximize its long-term rewards as it interacts with the Environment in the feedback configuration. The response the Agent gets from each state-action cycle (where an Agent selects an action from a set of actions at each state of the Environment) is called the reward function. The <span style="font-weight:bolder;">reward function</span> (or simply rewards) is a signal of the desirability of that state based on the action made by the Agent.

A "favourable" reward may indicate a good immediate event (i.e. state-action pair) for the Agent. On the other hand, an "unfavourable" reward may indicate a bad event for the Agent. The reward function is unique to the problem faced by the reinforcement agent and influences the choice of the optimal policy the Agent makes. The reward function largely defines the reinforcement learning task.

The other critical component is the idea of a <span style="font-weight:bolder;">value function</span> (or simply values). When an Agent takes an action in a particular state of the Environment, the reward function communicates to the Agent the immediate and intrinsic desirability of the state. It may, however, turn out that a state with an immediate high reward may likely lead to other states that are highly undesirable. This is not good as the goal of the RL Agent is to maximize long-term rewards. The <span style="font-weight:bolder;">value function</span> of a state is the expected long-term desirability of the current state by taking into consideration the likely future states and their reward functions.

In the final analysis, while the goal of the RL Agent is to maximize values, rewards are the primary signals received by the Agent as it interacts with the Environment. The idea of estimating values is to increase the quality of rewards at each state of the Agent-Environment interaction. Hence, when an Agent takes an action in a state, it does so based on the value estimates so that it can transit to new states with high values that consequently results in long-term rewards.

Rewards are cheap to obtain as they are essentially feedback received directly from the Environment. Values, on the other hand, must be continuously evaluated as the Agent iteratively interacts with the Environment and gathers more information. The task of finding an efficient technique for estimating values is central to designing modern reinforcement learning algorithms.

However, it is essential to note that while estimating value functions has influenced a lot of the ideas in modern RL literature, reinforcement learning problems can still be solved without estimating values. But of course, the efficiency, suitability and scalability of such methods is a different discussion.

Finally, we need a model of the Environment to learn an optimal policy for the reinforcement learning agent. The Environment model must somehow represent the stochastic nature of the Environment and return a next state and a response to the Agent upon taking an action. Having a model of the Environment is useful in planning, where an agent considers possible future outcomes before taking an action. In any case, reinforcement learning systems can also be rudimentary trial and error learners, as seen in Learning Automata theory. An agent that learns by trial and error can also learn the model of the Environment and later use it for calculative planning.


### Bibliography
<ul>
    <li>Narendra, K. S., & Thathachar, M. A. (2012). Learning automata: An introduction. Courier Corporation.</li>
    <li>Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: An introduction. MIT press.</li>
</ul>