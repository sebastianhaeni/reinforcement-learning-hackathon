####################################################################################################
# Import Packages

import gym
import random
import numpy  as np

from collections       import deque
from keras             import Sequential
from keras.layers      import Dense
from keras.activations import relu, linear
from keras.optimizers  import Adam
from gym.wrappers      import Monitor
from os                import path
from pathlib           import Path


####################################################################################################
# Initialize Input Parameters

PATH_VIDEO   = './V2/video'
PATH_WEIGHTS = './V2/weigths'

FILE_EPSILON = Path('./V2/epsilon.log')
FILE_REWARDS = Path('./V2/rewards.log')

EPISODES      = 1000
EPSILON       = 0.50
GAMMA         = 0.98   # unchanged
HIDDEN_NODES  = 32
LEARNING_RATE = 1e-3   # unchanged
SIZE_BATCH    = 100
SIZE_MEMORY   = 100000 # unchanged
STEPS         = 1024   # unchanged

RENDER = False
RECORD = True

####################################################################################################
# Define Class for Neural Network

class Model ():

    def __init__ (self, action_space, state_space, hidden_nodes, learning_rate):
        self.action_space  = action_space
        self.state_space   = state_space
        self.hidden_nodes  = hidden_nodes
        self.learning_rate = learning_rate

    def neural_network(self):
        model = Sequential()
        model.add(Dense(self.hidden_nodes, activation=relu, input_dim=self.state_space))       # Input  Layer
        model.add(Dense(self.hidden_nodes, activation=relu                            ))       # Hidden Layer
        model.add(Dense(self.hidden_nodes, activation=relu                            ))       # Hidden Layer
        model.add(Dense(self.action_space, activation=linear                          ))       # Output Layer
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse', metrics=['accuracy'])
        model.summary()
        return model


####################################################################################################
# Define Class for Agent

class Agent ():

    def __init__ (self, action_space, state_space):
        self.action_space = action_space
        self.batch_size   = SIZE_BATCH
        self.epsilon      = EPSILON
        self.gamma        = GAMMA
        self.memory       = deque(maxlen=(SIZE_MEMORY))
        self.model        = Model(action_space, state_space, HIDDEN_NODES, LEARNING_RATE).neural_network()

    def act (self, state):
        if np.random.rand() <= self.epsilon:     # Random Search
            actions = range(self.action_space-1)
            action  = np.random.choice(actions)
        else:                                    # Optimal Action
            actions = self.model.predict(state)
            action  = np.argmax(actions[0])
        return action
    
    def learn (self):
        if len(self.memory) > self.batch_size:
            batch       = random.sample(self.memory, self.batch_size)
            states      = np.array([i[0] for i in batch])
            actions     = np.array([i[1] for i in batch])
            rewards     = np.array([i[2] for i in batch])
            next_states = np.array([i[3] for i in batch])
            dones       = np.array([i[4] for i in batch])
            states      = np.squeeze(states     )
            next_states = np.squeeze(next_states)

            action_state_value = self.model.predict_on_batch(states) # Q-function
            next_action_state_value = rewards + self.gamma * np.amax(self.model.predict_on_batch(next_states),axis=1)*(1-dones) # Bellman Equation                          
            
            indices = np.array([i for i in range(self.batch_size)])
            action_state_value[[indices],[actions]] = next_action_state_value

            self.model.fit(states, action_state_value, epochs=1, verbose=0)

            
####################################################################################################
# Run

File_Epsilon = open(str(FILE_EPSILON), 'a+')
File_Rewards = open(str(FILE_REWARDS), 'a+')

env = gym.make('LunarLander-v2')
if RECORD == True:
    env = Monitor(env=env, directory=PATH_VIDEO, force=True)
env.seed(0)

action_space = env.action_space     .n
state_space  = env.observation_space.shape[0]
agent = Agent(action_space, state_space)
if path.exists(PATH_WEIGHTS):
    agent.model.load_weights(PATH_WEIGHTS)
    
rewards = []
    
for episode in range(EPISODES):
    state = env.reset()
    state = np.reshape(state,(1,state_space))
        
    score = 0

    for STEP in range(STEPS):
        if RENDER == True:
            env.render()
            
        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, (1, state_space))

        agent.memory.append((state,action,reward,next_state,done))

        state = next_state
        
        if  STEP % SIZE_BATCH == 0:
            agent.learn()
        
        score += reward
            
        if done:
            rewards.append(score)
            print("Episode: {}/{} | Reward: {:.2f} | Epsilon (final): {:.2f}"
                  .format(episode+1, EPISODES, score, agent.epsilon))
            break
        
    average_reward = np.mean(rewards[-100:])
    print("Average Reward: {:.2f}\n".format(average_reward))

    File_Epsilon.write('{0:f}\n'.format(average_reward))
    File_Rewards.write('{0:f}\n'.format(agent.epsilon ))
    File_Epsilon.flush()
    File_Rewards.flush()

env.close()

agent.model.save_weights(PATH_WEIGHTS)