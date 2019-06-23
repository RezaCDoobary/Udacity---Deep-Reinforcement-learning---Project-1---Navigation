# Udacity Reinforcement learning Nanodegree project 1 - Navigation

## Introduction
This is the first project that is introduced in the reinforcement learning nanodegree offered by udacity.

Broadly speaking the goal of the project is to train an agent placed in a world of blue and yellow bananas, to maximise the number of yellow bananas whilst minimising the number of blue bananas obtained.

## Project Description
Being more precise, the task of the project is the train an agent to collect as many yellow bananas as possible whilst avoiding the blue bananas. 

* The material goal of the task is reflected on the **reward** function by giving the agent +1 upon collecting a yellow banana, whilst giving the agent -1 if it collects a blue banana. This represents the reaction from the environment.

* The **state space** of the agent is 37 dimensional and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.

* The space of **actions** of the agent can take is 4 dimensional and have physical interpretations, namely to move:
    * 0 - forwards
    * 1 - backwards
    * 2 - left
    * 3 - right

* The task is deemed solved if the agent gets an average score of +13 over 100 consecutive episodes.

## Setup
* A complete set-up of python, the machine learning agents, openAI and much more can be found in https://github.com/udacity/deep-reinforcement-learning#dependencies. Of particular relevance will be unityagents from within the python folder in the corresponding repository.

* The banana application itself can be downloading from the following locations:
    * win x64 : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip
    * win x32 : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip
    * Linux : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
    * Mac : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip

* We employ the use of environment variables so as to not distribute personal computer information. As a result, please set 

    * \_DRL_LOCATION_ : The path of the python subfolder in the deep reinforcement learning rep descirbed above, generically this would be "...\drlnd\deep-reinforcement-learning\python"
    * \_BANANA_LOCATION_ : The path to the banana executable, this generically would look like "...\Banana_Windows_x86_64\Banana.exe"

## Code and result structure
There are three components to the solution. The first is the source code itself which implements the agent, the underlying model and further nessecary componenents. The second is the results folder, which contains the results of the training of the various models studied. Finally, the jupyter notebook named navigation.ipynb which acts as interface between the source code and the results folder, whilst itself displaying the results.

The detailed rundown is as follows:
* **The source code** can be found in the folder `\src` and include four python files:
    * `agent.py` : Contains the complete agent implementation subject the underlying model chosen.
    * `environment.py` : Is a very rudimentary wrapper for the environment to make it feel a little more like the OpenAI environments.
    * `model.py` :  The model implementations - in this case a neural network.
    * `replaybuffer.py` : The replay buffer implmentation for memory.

* **The results folder** contains subfolders which are named according to the model chosen, for example if the model in question has hidden layeres [128,64,128], and drop out probability of 0.56, include dueling DQN's but no double DQNs the we name the folder in this precise order: `\128_64_128_30_True_False`. In this folder, 
    * Are the checkpoints (`checkpoint.pth`) upon succesfully solving the task.
    * The corresponding scores against episodes plots.
N.B. The precise models employed are will stated at the end and detailed in the attached report.pdf.

* **The interfacing jupyter notebook** is considered the interface layer in which the user can decide on what precise architectures and models to use for the model. With each trained model, the scores against episodes is plotted, with the results forwared to the relevant results subfolder. The modelled agent can also be played from here to the see the solved task at work. 



## Models
* The models studied in the project included two main improvement prescriptions. They are:
    * Double DQN
    * Dueling DQN

We consider these seperately and in combintaion in our analysis.

Below is displayed the playing policy trained with dueling DQN with architechture 64-128-64.
![](play.gif)