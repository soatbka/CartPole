
import numpy as np
import gymnasium as gym

# create envir. variable
env = gym.make('CartPole-v1')#,render_mode='human')
env.reset()

# define for q_index
def q_index(state):
  # define x as a cart's position
  x = state[0]
  if -2.4<= x <-.8:
    x_index = 0
  elif -.8<= x <=.8:
    x_index = 3
  elif .8< x <= 2.4:
    x_index = 6
  # define x_dot as a cart's velocity
  x_dot = state[1]
  if x_dot <-.5:
    x_dot_index = 0
  elif -.5<= x_dot <=.5:
    x_dot_index = 1
  elif x_dot >.5:
    x_dot_index = 2
  # define theta is the pole's angle
  theta = state[2]
  if -12<= theta <-6:
    theta_index = 0
  elif -6<= theta <-1:
    theta_index = 3
  elif -1<= theta <0:
    theta_index = 6
  elif 0<= theta <1:
    theta_index = 9
  elif 1<= theta <6:
    theta_index = 12
  elif 6<= theta <=12:
    theta_index = 15
  # define theta_dot is the pole's rotation angle
  theta_dot = state[3]
  if theta_dot <-50:
    theta_dot_index = 0
  elif -50<= theta_dot <=50:
    theta_dot_index = 1
  elif theta_dot >50:
    theta_dot_index = 2
  state_index = ((x_index+x_dot_index),(theta_index+theta_dot_index))
  return state_index

# define q_table
q_table_size = [9,18] # x*x_dot = 9; theta*theta_dot = 18
q_table = np.random.uniform(low=-1, high=0, size=(q_table_size + [env.action_space.n])) # check print(q_table.shape)
c_learning_rate = .6
c_discounted_factor = .99
ep_reward_max = 0
action_list = []
action_list_max = []
current_index = []

for ep in range(1000):
  current_state, infor = env.reset()
  current_index = q_index(current_state)
  ep_reward = 0
  action_list = []
  terminate = False
  truncate = False
  while not (terminate or truncate):
    action = np.argmax(q_table[current_index])
    next_state, reward, terminate, truncate, infor = env.step(action)
    ep_reward += reward
    action_list.append(action)
    # update q_table
    current_q_value = q_table[current_index + (action,)]
    next_index = q_index(next_state)
    new_q_value = (1 - c_learning_rate) * current_q_value + c_learning_rate * (reward + c_discounted_factor * np.max(q_table[next_index]))
    q_table[current_index + (action,)] = new_q_value
    current_index = next_index

  if ep_reward > ep_reward_max:
    ep_reward_max = ep_reward
    action_list_max = action_list

print('The highest reward: ',ep_reward_max)
print('The maximum action list: ',action_list_max)
