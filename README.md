# drl-sac
Deep Reinforcement Learning with Soft Actor Critic model for jetbot

This is my first attempt to implement a deep reinforcement model for the jetbot:

For this, it has attached a camera and six buttons as sensors.

This is an implementation of a SAC agent (Soft Actor Critic) with pytorch.

The goal is that the robot can drive as long as possible without collision with obstacles.

I don't use any simulation for this (like gym) but I want the robot to start learning from zero.

Hope it works :)

*NOTE*:

This was a trial-and-error with OpenAIs Chat-GPT-4 (version as of April 6th 2023).
The resulting code is full of bugs, the robot never moved :( At one point the deep learning model (pytorch) tried to allocate 56GB Ram.
I don't think that ChatGPT4 can replace software engineers that know what they do :D

Maybe GPT5 ???
