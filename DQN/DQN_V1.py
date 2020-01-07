####################################################################################################
# Import Packages

import gym
import numpy as np
import random

from collections       import deque
from gym.wrappers      import Monitor
from keras             import Sequential
from keras.activations import relu, linear
from keras.layers      import Dense
from keras.optimizers  import Adam
from os                import path
from pathlib           import Path


####################################################################################################
# Input Parameters

FILE_REWARDS = Path('Average_Rewards_V1.log')

CHECKPOINT_DIRECTORY = './Checkpoints_Baseline'

BATCH_SIZE    = 64
EPISODES      = 256
EPOCHS        = 1
EPSILON       = 1.0
EPSILON_DECAY = 0.98
EPSILON_MIN   = 0.02
GAMMA         = 0.98
HIDDEN_LAYERS = 2
HIDDEN_NODES  = 128
LEARNING_RATE = 1e-3
MEMORY_SIZE   = 1000000
RENDER        = True
STEPS         = 1024


####################################################################################################
# Define fully-connected feed-forward neural network

class Model():

    # Optimizer
    # (change manually)

    # Neural Network
    # (change manually)

    def __init__(self, action_space, state_space, hidden_nodes, hidden_layers, learning_rate):
        self.action_space  = action_space
        self.state_space   = state_space
        self.nodes         = hidden_nodes
        self.layers        = hidden_layers
        self.learning_rate = learning_rate
    
    def Build(self):
        model = Sequential()
        model.add(Dense(self.nodes, activation=relu, input_dim=self.state_space))               # Input Layer
        for _ in range (self.layers):
            model.add(Dense(self.nodes, activation=relu))                                       # Hidden Layer(s)
        model.add(Dense(self.action_space, activation=linear))                                  # Output Layer
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse', metrics=['accuracy'])
        model.summary()
        return model


####################################################################################################
# Define Agent

class Agent():
     
    # act

    def __init__ (self, action_space, state_space):
         self.action_space  = action_space
         self.batch_size    = BATCH_SIZE
         self.epochs        = EPOCHS
         self.epsilon       = EPSILON
         self.epsilon_min   = EPSILON_MIN
         self.epsilon_decay = EPSILON_DECAY
         self.gamma         = GAMMA
         self.memory        = deque(maxlen=MEMORY_SIZE)
         self.model         = Model(action_space,state_space,HIDDEN_NODES,HIDDEN_LAYERS,LEARNING_RATE).Build()

    def act (self, state):
        if np.random.rand() <= self.epsilon: # Occasionally explore the state space via a random action
            actions = range(self.action_space-1)
            action  = np.random.choice(actions)
        else:                                # Normally choose the predicted optimal action
            actions = self.model.predict(state)
            action  = np.argmax(actions[0])
        return action

    def learn (self):
        if len(self.memory) > self.batch_size:
            batch =  random.sample(self.memory, self.batch_size)
            states      = np.array([i[0] for i in batch])
            actions     = np.array([i[1] for i in batch])
            rewards     = np.array([i[2] for i in batch])
            next_states = np.array([i[3] for i in batch])
            dones       = np.array([i[4] for i in batch])
            states      = np.squeeze(states     )
            next_states = np.squeeze(next_states)

            # Action-state-value for the current state (Q-function)
            action_state_value      = self.model.predict_on_batch(states)
            
            # Optimal Action-state-value for the next state (Bellman equation)
            action_state_value_next = rewards + self.gamma*np.argmax(self.model.predict_on_batch(next_states),axis=1)*(1-dones)
            
            indices = np.array([i for i in range(self.batch_size)])
            action_state_value[[indices],[actions]] = action_state_value_next
            self.model.fit(states, action_state_value, epochs=self.epochs, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon*self.epsilon_decay


####################################################################################################
# Define Function for Saving Rewards

File = open(str(FILE_REWARDS), 'a+')

def Save_Rewards(Reward):
    File.write('{0:f}\n'.format(Reward))
    File.flush()


####################################################################################################
# 

def main ():
    
    env = gym.make('LunarLander-v2')
    
    env.seed(0)
    np.random.seed(0)

    action_space = env.action_space     .n
    state_space  = env.observation_space.shape[0]

    agent = Agent(action_space, state_space)
    
    if path.exists(CHECKPOINT_DIRECTORY):
        agent.model.load_weights(CHECKPOINT_DIRECTORY)

    rewards = []

    for episode in range(EPISODES):
        
        state = env.reset()
        state = np.reshape(state, (1,state_space))

        score = 0

        for _ in range(STEPS):

            if RENDER:
                env.render()

            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            
            next_state = np.reshape(next_state, (1,state_space))

            observation = (state, action, reward, next_state, done)

            agent.memory.append(observation)
            
            state = next_state

            agent.learn()
            
            score += reward

            if done:
                print("Episode: {}/{} | Reward: {:.2f} | Epsilon: {:.2f}\n"
                      .format(episode+1, EPISODES, score, agent.epsilon))
                break

        rewards.append(score)

        average_reward = np.mean(rewards[-100:])
        print("Average Reward: {:.2f}\n".format(average_reward))

        Save_Rewards(average_reward)

        agent.model.save_weights(CHECKPOINT_DIRECTORY)

    env.close()


####################################################################################################
#

if __name__ == "__main__":
    main()
