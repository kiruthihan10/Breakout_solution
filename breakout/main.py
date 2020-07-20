import tensorflow as tf

from tensorflow import keras

import gym

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.animation as animation

from collections import deque

from gym import wrappers

from tensorflow.keras import backend as K

def record_epochs(index):
    #return True
    return index%50==0

class obs_wrapper(gym.ObservationWrapper):
    def __init__(self,env):
        self.image_shape = (env.reset().shape[0],env.reset().shape[1],3)
        self.mobile = keras.applications.Xception(input_shape=self.image_shape,include_top=False,pooling="avg")
        desired_shape = self.mobile.predict(np.array([np.zeros(self.image_shape)]))[0]
        desired_shape = desired_shape.shape[0]
        super().__init__(env)
        que_length = 2
        self.outs = list(np.zeros((que_length,desired_shape)))
        del desired_shape
        self.outs = deque(self.outs,maxlen=que_length)
        
    def state_preprocess(self,state):
        state = state/255
        #zeros = np.zeros(self.image_shape)
        #zeros[self.image_size//2-state.shape[0]//2:self.image_size//2+state.shape[0]//2,self.image_size//2-state.shape[1]//2:self.image_size//2+state.shape[1]//2,:]=state
        #print(zeros.shape)
        state = self.mobile.predict(np.array([state]))[0]
        self.outs.append(np.array(state))
        return np.array(list(self.outs))

    def observation(self,obs):
        obs = self.state_preprocess(obs)
        return obs

class action_wrapper(gym.ActionWrapper):
    def __init__(self,env):
        super().__init__(env)

    def action(self,act):
        act+=1
        return act
    
class reward_wrapper(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self.env = env
        self.step_count = 0
    
    def step(self,action):
        next_state, reward, done, info = self.env.step(action)
        reward += self.step_count/1000
        self.step_count+=1
        if done:
            self.step_count = 0
        return next_state, reward, done, info  

      

#env = wrappers.Monitor(reward_wrapper(obs_wrapper(action_wrapper(gym.make("Breakout-v0")))),"video",record_epochs,mode="training")
env = obs_wrapper(action_wrapper(gym.make("Breakout-v0")))
a=env.reset()
print(a.shape)

model = keras.Sequential([
    keras.layers.Input((a.shape[0],a.shape[1])),
    keras.layers.SimpleRNN(100,unroll=True,return_sequences=True,activation="relu"),
    keras.layers.SimpleRNN(10,unroll=True,activation="relu"),
    keras.layers.Dense(3)
])

print(model.summary())

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(3)
    else:
        #print(state.shape)
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

"""We will also need a replay memory. It will contain the agent's experiences, in the form of tuples: `(obs, action, reward, next_obs, done)`. We can use the `deque` class for that:"""

replay_memory = deque(maxlen=1000)

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

batch_size = 128
discount_rate = 0.99
lr=1e-4
optimizer = keras.optimizers.Nadam(lr=lr)
loss_fn = keras.losses.logcosh

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states,use_multiprocessing=True,workers=4)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards +
                       (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, 3)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    #print(loss)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

env.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

rewards = [] 
best_score = -np.inf
best = False
global_step = 0
for episode in range(1000):
    episode_reward = 0
    obs = env.reset()    
    step = -1
    env.step(1)
    epsilon = max(1 - episode / 300, 0.001)
    while True:
        global_step += 1
        step += 1
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        episode_reward+=reward
        if global_step >batch_size and global_step%4==0:
            training_step(batch_size)
        if done:
            break
    rewards.append(episode_reward) # Not shown in the book
    if episode_reward>best_score:
        best = True
        best_score = episode_reward
        best_weights = model.get_weights() # Not shown
    
    print("\rEpisode: {}, Current Score: {:.3f}, eps: {:.3f}, Best Score: {:.3f}".format(episode, episode_reward, epsilon,best_score), end="") # Not shown
    

        

model.set_weights(best_weights)

plt.figure(figsize=(8, 4))
plt.plot(rewards)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Sum of rewards", fontsize=14)
plt.show()

env.seed(42)
state = env.reset()

frames = []

while True:
    env.step(1)
    action = epsilon_greedy_policy(state)
    state, reward, done, info = env.step(action)
    if done:
        break
    img = env.render(mode="rgb_array")
    frames.append(img)

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim
   
plot_animation(frames)