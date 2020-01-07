from os import path
import numpy as np
import gym
from gym.wrappers import Monitor

import random
import numpy as np
from collections import deque

ENVIRONMENT = 'LunarLander-v2'
CHECKPOINT_DIRECTORY = './dqn/output/model/checkpoint'
FIGURE_DIRECTORY = './dqn/output/summary.png'
VIDEO_DIRECTORY = './dqn/output/video'
EPISODES = 256
STEPS = 1024
MEMORY_SIZE = 1000000
BATCH_SIZE = 64
HIDDEN_NODES = 128
HIDDEN_LAYERS = 2
LEARNING_RATE = 1e-3
GAMMA = 0.98
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.98
RENDER = True
RECORD = False

from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import Adam

class Model():
    """ Returns a fully-connected feed-forward neural network. """
    def __init__(self, action_space, state_space, hidden_nodes, layers, learning_rate):
        self.action_space = action_space
        self.state_space = state_space
        self.nodes = hidden_nodes
        self.layers = layers
        self.learning_rate = learning_rate

    def build(self):
        """ Builds the neural network model. """
        model = Sequential()
        # Input layer
        model.add(Dense(self.nodes, input_dim=self.state_space, activation=relu))
        # Hidden layers
        for _ in range(self.layers):
            model.add(Dense(self.nodes, activation=relu))
        # Output layer
        model.add(Dense(self.action_space, activation=linear))
        model.compile(
            optimizer=Adam(lr=self.learning_rate),
            loss='mse',
            metrics=['accuracy']
        )
        model.summary()
        return model

class Agent():
    """ Observes environment, selects actions and trains model. """
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.batch_size = BATCH_SIZE
        self.epsilon = 1.0
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.gamma = GAMMA
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = Model(
            action_space,
            state_space,
            HIDDEN_NODES,
            HIDDEN_LAYERS,
            LEARNING_RATE
        ).build()

    def act(self, state):
        """ Selects an action from the action space. """
        # Occasionally explore the state space via a random action
        if np.random.rand() <= self.epsilon:
            actions = range(self.action_space-1)
            action = np.random.choice(actions)
        # Normally choose predicted optimal action
        else:
            policy = self.model.predict(state)
            action = np.argmax(policy[0])
        return action

    def observe(self, environment, action):
        """ Gets new state and reward from environment. """
        return environment.step(action)

    def remember(self, observation):
        """ Stores observation in memory. """
        self.memory.append(observation)

    def learn(self):
        """ Retrains model using new observations. """
        if len(self.memory) > self.batch_size:
            # Get a batch of observations
            batch = random.sample(self.memory, self.batch_size)
            # Retrieve signals from batch
            states = np.array([i[0] for i in batch])
            actions = np.array([i[1] for i in batch])
            rewards = np.array([i[2] for i in batch])
            next_states = np.array([i[3] for i in batch])
            dones = np.array([i[4] for i in batch])
            # Remove single-dimensional entries from array shape
            states = np.squeeze(states)
            next_states = np.squeeze(next_states)
            # Predict action-state value function (Q-function) for current state
            action_state_value = self.model.predict_on_batch(states)
            # Predict optimal action-state value function for next state (Bellman equation)
            next_action_state_value = rewards + self.gamma*np.amax(self.model.predict_on_batch(next_states), axis=1)*(1-dones)
            # Create index from batch size
            indices = np.array([i for i in range(self.batch_size)])
            # Map actions to optimal action-state value function
            action_state_value[[indices], [actions]] = next_action_state_value
            # Train model with states and predicted action-state value function
            self.model.fit(states, action_state_value, epochs=1, verbose=0)
            # Decrease likelihood of exploratory action
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    def load(self, directory):
        """ Loads model weights. """
        self.model.load_weights(directory)

    def save(self, directory):
        """ Saves model weights. """
        self.model.save_weights(directory)

def main():
    """ Orchestrates agent and environment interactions. """
    # Create environment
    environment = gym.make(ENVIRONMENT)
    if RECORD:
        environment = Monitor(
            env=environment,
            directory=VIDEO_DIRECTORY,
            video_callable=lambda episode_id: True,
            force=True
        )
    # Set random seeds
    environment.seed(0)
    np.random.seed(0)
    # Get action and state space sizes
    action_space = environment.action_space.n
    state_space = environment.observation_space.shape[0]
    # Instantiate agent
    agent = Agent(action_space, state_space)
    # Load model weights
    if path.exists(CHECKPOINT_DIRECTORY):
        agent.load(CHECKPOINT_DIRECTORY)
    # Initialise list of all rewards
    rewards = []
    for episode in range(EPISODES):
        # Get initial state
        state = environment.reset()
        state = np.reshape(state, (1, state_space))
        # Reset score for this episode
        score = 0
        for _ in range(STEPS):
            if RENDER:
                environment.render()
            # Agent selects action from state
            action = agent.act(state)
            # Agent performs action and makes an observation of the environment
            next_state, reward, done, _ = agent.observe(environment, action)
            next_state = np.reshape(next_state, (1, state_space))
            observation = (state, action, reward, next_state, done)
            # Agent remembers parameters of this time step
            agent.remember(observation)
            state = next_state
            # Agent retrains model
            agent.learn()
            score += reward
            if done:
                print("Episode: {}/{}. Reward: {:.2f}".format(episode+1, EPISODES, score))
                break
        rewards.append(score)
        # Average reward over the last 100 episodes
        average_reward = np.mean(rewards[-100:])
        print("Average reward: {:.2f}\n".format(average_reward))
    # Terminate environment
    environment.close()
    # Save model
    agent.save(CHECKPOINT_DIRECTORY)
    # Display performance over time
    #summary(rewards)

if __name__ == "__main__":
    main()