---
layout: page-rl-theory-practice
title: 'Learning with Uncertainty'
permalink: rl-theory-practice/learning-with-uncertainty/
---

The ideas behind modern reinforcement learning are built from theories of trial-and-error learning and computational adaptive control. The general goal of these approaches is to build an agent that maximizes rewards for a certain behaviour as it interacts with a random Environment in a feedback loop. The agent updates its policy or strategy for making decisions in the face of uncertainty by the responses the agent receives from the Environment.

<div class="fig figcenter fighighlight">
     <img src="/assets/rl_theory_practice/rl-framework.png" width="60%" height="60%">
     <div class="figcaption" style="text-align: center;">
        <span style="font-weight:bolder;">General reinforcement learning framework.</span> An agent interacts with an Environment in a feedback configuration and updates its strategy for choosing an action based on the responses it gets from the Environment.
     </div>
</div>

A trial-and-error search is an approach to learning behaviour from the field of animal psychology. Thorndike, Pavlov and Skinner were major proponents in this field of learning. The theory of trial and error learning concerns itself with how agents learn behaviour by the strengthening or weakening of mental bonds based on the satisfaction or discomfort the agent perceives from the Environment after carrying out a particular action (Thorndike, 1898). This idea of learning was called the “law of effect” where “satisfaction” is the reinforcing of an accompanying action based on a “reward” and “discomfort” leads to the discontinuation of an action due to “penalty”. These ideas of rewards and penalties were explored further by B.F. Skinner’s with his work on operant conditioning, which posits that the agent voluntarily reinforces its behaviour based on the stimuli or action resulting in a response from the Environment (Skinner, 1938). On the other hand, Pavlov’s classical conditioning argues that the pairing of stimuli (of which the first is the unconditioned stimulus) creates an involuntary response in behaviour by the agent (Pavlov, 1927). Both behavioural theories of learning involve the notion of some sort of associative pairing of stimuli to the response whereby an agent’s behaviour is conditioned by the repetition of actions in a feedback loop.

<div class="fig figcenter fighighlight">
     <img src="/assets/rl_theory_practice/rat-maze.png" width="60%" height="60%">
     <div class="figcaption" style="text-align: center;">
        <span style="font-weight:bolder;">T-maze.</span> The T-maze is used in conditioning experiments to examine the behaviour of rodents as they learn to find food from successive trials using different schedules.
     </div>
</div>

Trial-and-error learning and the law of effects have two distinct properties that have influenced modern reinforcement learning techniques in that they are selectional and associative. Modern RL is selectional given that for a particular state of the Environment, an action is sampled from a set of actions, and it is associative given that favourable actions with their associated states are remembered (i.e. stored in memory) (Sutton and Barto, 1998).

The field of adaptive control is concerned with learning the behaviour of a controller (or an agent) in a complex dynamical system where uncertainties exist in the parameters of the controlled system. (Bellman, 1961) categorized control problems as deterministic, stochastic and adaptive. In an adaptive control system, a considerable level of uncertainty exists in the system where little is known about the structure of the Environment or the distribution of the parameters. While experimentation may be used to obtain some information about the system, the time taken will make such an approach infeasible. Hence, the need to learn the behaviour of the controller in an “online” configuration. (Bellman, 1957a)  showed the Bellman equation as a function that captures the state and value function of a dynamical system and introduced dynamic programming as a class of methods for finding the optimal controller for an adaptive control problem. (Bellman, 1957b) formulated the Markov Decision Processes (MDP) as a discrete-time stochastic control process for modelling the reinforcement learning framework where the agent interacts with the Environment in a feedback loop. The Markov property assumes that the current state captures all the information necessary to predict the next state and its expected response without relying on the previous sequence of states. In other words, the Markov property is the conditional probability that the future states of the Environment only depends on the current state. Hence it is conditionally independent of the past states given that we know the current state. The MDP is based on the theoretical assumption that the states of the Environment possess the Markov property. 

### Bibliography
<ul>
    <li>Pavlov IP (1927). Translated by Anrep GV. "Conditioned Reflexes: An Investigation of the Physiological Activity of the Cerebral Cortex". Nature. 121 (3052): 662–664. Bibcode:1928Natur.121..662D. doi:10.1038/121662a0.</li>
    <li>Skinner, B. F. (1938). The behaviour of organisms: an experimental analysis. Appleton-Century.</li>
    <li>Sutton, R. S., & Barto, A. G. (1998). Introduction to reinforcement learning (Vol. 2, No. 4). Cambridge: MIT Press.</li>
    <li>Thorndike, E. L. (1898). Animal intelligence: An experimental study of the associative processes in animals. The Psychological Review: Monograph Supplements, 2(4), i–109. https://doi.org/10.1037/h0092987.</li>
    <li>Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: An introduction. MIT press.</li>
    <li>Bellman, R. E. (1961). Adaptive Control Processes - A Guided Tour. Princeton: Princeton University Press.</li>
    <li>Bellman, R. E. (1957a). Dynamic Programming. Princeton: Princeton University Press.</li>
    <li>Bellman, R. E. (1957b). A Markovian Decision Process. Journal of Mathematics and Mechanics, 6(5), 679-684. Retrieved from www.jstor.org/stable/24900506</li>
</ul>