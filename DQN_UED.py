#%% 
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

from maze_env import Maze
#%% Show the maze
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
HEIGHT = 7
WIDTH = 7
WALLS = 8
TRAPS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size =2, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.reshape(x,(-1,1,HEIGHT,WIDTH)).float()
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

BATCH_SIZE = 128
GAMMA = 0.999

EPS_START = 0.9
EPS_START_ORIGINAL = 0.9
EPS_START_FINAL = 0.2

EPS_END = 0.05

EPS_DECAY = 200
EPS_DECAY_ORIGINAL = 200

TARGET_UPDATE = 10

n_actions = 4
# Instantiating the DQN model useed for this task
policy_net = DQN(HEIGHT,WIDTH, n_actions).to(device).float()
target_net = DQN(HEIGHT,WIDTH, n_actions).to(device).float()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())

steps_done = 0

# This function will select the action to perform given a current state
def compute_eps_threshold():
    # Return the eps treshold at a given time
    return EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
        
def select_action(state,best = False):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold or best:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []
eps_threshold = []

def plot_durations_live():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    """
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    """
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def plot_durations(maze_number):
    fig, ax = plt.subplots()
    plt.xlabel("Episode for Maze "+str(maze_number))
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    ax.plot(durations_t.numpy(), color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Duration in blue and EPS in Green")

    ax2 = ax.twinx()
    eps_thresholds = torch.tensor(eps_threshold, dtype=torch.float)
    ax2.plot(eps_thresholds.numpy(), color='green')
    ax2.tick_params(axis='y', labelcolor='green')

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

#%% Training on n different ued mazes
# We will save the mazes and order them by difficulty of solving
difficulty_ordered = []
NUMBER_OF_MAZES = 10

for mazes in range(NUMBER_OF_MAZES):
    episode_durations = []
    eps_threshold = []
    
    # Instantiating the DQN model useed for this task
    policy_net = DQN(HEIGHT,WIDTH, n_actions).to(device).float()
    target_net = DQN(HEIGHT,WIDTH, n_actions).to(device).float()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    
    steps_done = 0

    memory = ReplayMemory(10000)
    env = Maze(height =HEIGHT-2,width = WIDTH-2, walls = WALLS,traps = TRAPS)
    print("Training")
    num_episodes =50
    mean_duration = 0
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset(True)
        observation_np= env.maze_state()
        observation = torch.tensor(observation_np)
        state = observation
        eps_threshold.append(compute_eps_threshold())

        for t in count():
            # Select and perform an action
            action = select_action(state,best = False)
            observation_np, reward, done = env.make_move(action.item())
            observation = torch.tensor(observation_np)
            reward = torch.tensor([reward], device=device)

            # Observe new state
            if not done:
                next_state = observation
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                mean_duration += t
                #plot_durations_live()
                break
        
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    plot_durations(mazes)

    mean_duration = mean_duration/num_episodes
    print("Mean duration for that maze:" + str(mean_duration))
    difficulty_ordered.append([mean_duration,env.original_maze])


#%% We will now train the maze agent on mazes of increasing difficulty
print(difficulty_ordered)
ordered_mazes = np.array(difficulty_ordered)
ordered_mazes = ordered_mazes[ordered_mazes[:, 0].argsort()]
print(ordered_mazes)
print('Training Complete on first mazes')
print("Ordering the mazes by difficulty")
print("Training by order of difficulty")
# Instantiating the DQN model useed for this task
policy_net = DQN(HEIGHT,WIDTH, n_actions).to(device).float()
target_net = DQN(HEIGHT,WIDTH, n_actions).to(device).float()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
memory = ReplayMemory(10000)
optimizer = optim.RMSprop(policy_net.parameters())

env.reset(True)


for mazes in range(len(ordered_mazes)):
    episode_durations = []
    eps_threshold = []
    steps_done = 0
    EPS_START = EPS_START_ORIGINAL - ((EPS_START_ORIGINAL- EPS_START_FINAL)/len(ordered_mazes))*mazes


    print("Steps Done:"+str(steps_done))
    print("EPS threshold:"+str(compute_eps_threshold()))
    print("Training on maze number:"+str(mazes))
    env.reset(True)
    env.set_maze(ordered_mazes[mazes,1])
    env.render_maze()
    print("Training")
    num_episodes =100
    mean_duration = 0
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        observation_np= env.maze_state()
        observation = torch.tensor(observation_np)
        state = observation
        for t in count():
            # Select and perform an action
            action = select_action(state,best = False)
            observation_np, reward, done = env.make_move(action.item())
            observation = torch.tensor(observation_np)
            reward = torch.tensor([reward], device=device)

            # Observe new state
            if not done:
                next_state = observation
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                eps_threshold.append(compute_eps_threshold())
                mean_duration += t
                break
        
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    plot_durations(mazes)
#%%
# Visualize the trained Maze solver
print("Testing on the maze:")
env.reset()
difficult_maze = [[2.0,2.0,2.0,2.0,2.0,2.0,2.0],
                            [2.0,1.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,0.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,0.0,2.0,0.0,6.0,0.0,2.0],
                            [2.0,0.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,6.0,0.0,0.0,2.0,3.0,2.0],
                            [2.0,2.0,2.0,2.0,2.0,2.0,2.0]]
env.set_maze(np.array(difficult_maze))

env.render_maze()
observation = torch.tensor(observation_np)
state = observation
for t in count():
    # Select and perform an action
    action = select_action(state,best = True)
    observation_np, reward, done = env.make_move(action.item())
    observation = torch.tensor(observation_np)
    reward = torch.tensor([reward], device=device)
    # Observe new state
    if not done:
        state = observation
    else:
        print("Episode finished after {} timesteps".format(t+1))
        break

plt.ioff()
plt.show()

# %% Test to see the depression and anxiety levels of the agent

situation_1 = [[2.0,2.0,2.0,2.0,2.0,2.0,2.0],
                            [2.0,1.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,0.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,0.0,2.0,0.0,6.0,0.0,2.0],
                            [2.0,0.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,6.0,0.0,0.0,2.0,3.0,2.0],
                            [2.0,2.0,2.0,2.0,2.0,2.0,2.0]]
situation_2 = [[2.0,2.0,2.0,2.0,2.0,2.0,2.0],
                            [2.0,0.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,0.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,0.0,2.0,0.0,6.0,0.0,2.0],
                            [2.0,1.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,6.0,0.0,0.0,2.0,3.0,2.0],
                            [2.0,2.0,2.0,2.0,2.0,2.0,2.0]]
situation_3 = [[2.0,2.0,2.0,2.0,2.0,2.0,2.0],
                            [2.0,0.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,0.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,0.0,2.0,1.0,6.0,0.0,2.0],
                            [2.0,0.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,6.0,0.0,0.0,2.0,3.0,2.0],
                            [2.0,2.0,2.0,2.0,2.0,2.0,2.0]]
#%%
env.set_maze(situation_1)

observation_np= env.maze_state()
observation = torch.tensor(observation_np)
state = observation
with torch.no_grad():
    print(policy_net(state))

env.set_maze(situation_2)

observation_np= env.maze_state()
observation = torch.tensor(observation_np)
state = observation
with torch.no_grad():
    print(policy_net(state))

env.set_maze(situation_3)

observation_np= env.maze_state()
observation = torch.tensor(observation_np)
state = observation
with torch.no_grad():
    print(policy_net(state))
# %% Training on hard maze
new_maze = [[2.0,2.0,2.0,2.0,2.0,2.0,2.0],
                            [2.0,1.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,0.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,0.0,2.0,0.0,6.0,0.0,2.0],
                            [2.0,0.0,2.0,0.0,2.0,0.0,2.0],
                            [2.0,6.0,0.0,0.0,2.0,3.0,2.0],
                            [2.0,2.0,2.0,2.0,2.0,2.0,2.0]]
env.set_maze(np.array(new_maze))
print("Training")
num_episodes =100
eps_threshold = []
for i_episode in range(num_episodes):
    eps_threshold.append(compute_eps_threshold())
    # Initialize the environment and state
    env.reset()
    observation_np= env.maze_state()
    observation = torch.tensor(observation_np)
    state = observation
    for t in count():
        # Select and perform an action
        action = select_action(state,best = False)
        observation_np, reward, done = env.make_move(action.item())
        observation = torch.tensor(observation_np)
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = observation
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            #plot_durations_live()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
plot_durations(1)
# %%
