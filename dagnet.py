# < begin copyright > 
# Copyright Ryan Marcus 2017
# 
# This file is part of dagger.
# 
# dagger is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# dagger is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with dagger.  If not, see <http://www.gnu.org/licenses/>.
# 
# < end copyright > 
import numpy as np
from graph import DAG, dag_from_file
from sequence_processor import Scheduler
from keras.models import Sequential
from keras.layers import Dense, Activation

import random


def load_dag():
    #adj = np.array([[-1, 1, 2, -1], [-1, -1, -1, 5],
    #                [-1, -1, -1, 3], [-1, -1, -1, -1]])
    #w = np.array([[3, 2], [4, 1], [1, 1], [8, 7]])
    #d = DAG(w, adj, (1, 5))
    #return d
    return dag_from_file("sparselu1.txt")

def get_scheduler(dag, deadline):
    return Scheduler(dag, deadline, 2)


DEADLINE = 2000
dag = load_dag()
sched = get_scheduler(dag, DEADLINE)
model = Sequential()
total_inputs = sched.state_vector().shape[0]
total_outputs = len(sched.actions())

print("Input dimension:", total_inputs)
print("Output dimension:", total_outputs)

def generate_random_data(sched):
    # generate some data by moving randomly
    sched.reset()
    actions = []
    states = []
    while not sched.is_done():
        action = random.choice(range(total_outputs))
        actions.append(action)
        states.append(sched.state_vector())
        sched.actions()[action]()
    cost = sched.cost()

    toR = []
    for i in range(len(actions)-1):
        toR.append((states[i], actions[i], 0, states[i+1]))

    # add the last experience with the reward
    toR.append((states[-1], actions[-1], cost, sched.state_vector()))
    return toR



# build the net
model.add(Dense(total_inputs * 2, input_dim=total_inputs))
model.add(Dense(total_outputs, activation="relu"))
model.add(Activation("softmax"))

model.compile("RMSprop", "mse")



# first, generate some random data to get us going
# tuples: (before state, action, reward, after state)
print("Generating random experiences...")
experience = []
NUM_RAND_EXPERIENCES = 5
for i in range(NUM_RAND_EXPERIENCES):
    print(i, "/", NUM_RAND_EXPERIENCES)
    experience.extend(generate_random_data(sched))
 

print("Gathered", len(experience), "random experiences")

GAMMA = 0.5
EPSILON = 0.05
def train_on_minibatch():
    batch = random.choices(experience, k=32)

    # first, evaluate the output vectors for the initial state
    initial_states = np.array([x[0] for x in batch])
    next_states = np.array([x[3] for x in batch])
    current_Q = model.predict(initial_states)
    next_Q = model.predict(next_states)

    x = []
    y = []
    for exp, currQ, nxtQ in zip(batch, current_Q, next_Q):
        x.append(np.array(exp[0]))
        newQ = np.array(currQ)
        if exp[2] == 0:
            # not a terminal reward.
            newQ[exp[1]] = 0 + GAMMA * max(nxtQ)
        else:
            # terminal reward
            newQ[exp[1]] = exp[2]
        y.append(newQ)

    model.train_on_batch(np.array(x), np.array(y))
    
# train on mini batches
for i in range(500):
    print(i, "/", 500)
    train_on_minibatch()

print("Trained the network on some random experiences")
    
def do_dqn_iteration():
    sched.reset()
    actions = []
    states = []
    
    while not sched.is_done():
        # with probability EPSILON, move randomly
        action = None
        if random.random() < EPSILON:
            action = random.choice(range(total_outputs))
        else:
            # select the action that maximizes Q
            Q = model.predict(np.array([sched.state_vector()]))
            action = np.argmin(Q[0])
            print(Q[0])

        print("selected:", action)
        actions.append(action)
        states.append(sched.state_vector())
        sched.actions()[action]()
        
    cost = sched.cost()

    for i in range(len(actions)-1):
        yield (states[i], actions[i], 0, states[i+1])

    # add the last experience with the reward
    print("DQN iteration got cost:", cost)
    yield (states[-1], actions[-1], cost, sched.state_vector())
    

for i in range(1000):
    experience.extend(list(do_dqn_iteration()))
    train_on_minibatch()
    print("Iteration", i, "complete, now have", len(experience), "experiences")


