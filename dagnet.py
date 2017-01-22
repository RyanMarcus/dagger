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
import pickle
import random
from keras.models import load_model


def load_dag():
    #adj = np.array([[-1, 1, 2, -1], [-1, -1, -1, 5],
    #                [-1, -1, -1, 3], [-1, -1, -1, -1]])
    #w = np.array([[3, 2], [4, 1], [1, 1], [8, 7]])
    #d = DAG(w, adj, (1, 5))
    #return d
    return dag_from_file("sparselu2.txt")

def get_scheduler(dag, deadline):
    return Scheduler(dag, deadline, 2)


DEADLINE = 3000
dag = load_dag()
all_on_one = [
    dag.cost_of([list(range(dag.num_vertices()))], [0], DEADLINE),
    dag.latency_of([list(range(dag.num_vertices()))], [0])
]

all_alone = [
    dag.cost_of([], [0], DEADLINE),
    dag.latency_of([], [])
]
print("all on one cost, latency:\t", all_on_one)
print("all alone cost, latency:\t", all_alone)

sched = get_scheduler(dag, DEADLINE)

while not sched.is_done():
    if sched.highest_edge_weight_child() == None:
        sched.split()
        sched.least_slack()

print("simple heur cost, latency:\t", [sched.cost(), sched.latency()])

simple_heuristic_cost = min(sched.cost(), all_on_one[0], all_alone[0])
sched.reset()


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
    rewards = []
    while not sched.is_done():
        action = random.choice(range(total_outputs))
        actions.append(action)
        states.append(sched.state_vector())
        rewards.append(sched.actions()[action]())

    toR = []
    for i in range(len(actions)-1):
        toR.append((states[i], actions[i], rewards[i], states[i+1], False))
    # add the terminal
    toR.append((states[-1], actions[-1], rewards[-1], sched.state_vector(), True))
        
    return toR



# build the net
model.add(Dense(total_inputs * 2, input_dim=total_inputs))
model.add(Dense(total_inputs * 2, activation="relu"))
model.add(Dense(total_inputs * 2, activation="relu"))
model.add(Dense(total_outputs))
model.compile("RMSprop", "mse")

#model = load_model("dqn_keras.h5")

#from keras.utils.visualize_util import plot
#plot(model, to_file='model.png')


# first, generate some random data to get us going
# tuples: (before state, action, reward, after state)
print("Generating random experiences...")
experience = []
NUM_RAND_EXPERIENCES = 0
for i in range(NUM_RAND_EXPERIENCES):
    print(i, "/", NUM_RAND_EXPERIENCES)
    experience.extend(generate_random_data(sched))
 

print("Gathered", len(experience), "random experiences")

GAMMA = 0.8
#EPSILON = 1.0
def train_on_minibatch(data=None):
    #with python3.6: batch = random.choices(experience, k=32)
    batch = data
    if data == None:
        batch = [random.choice(experience) for x in range(32)]

    # first, evaluate the output vectors for the initial state
    initial_states = np.array([x[0] for x in batch])
    current_Q = model.predict(initial_states)

    x = []
    y = []
    for exp, currQ in zip(batch, current_Q):
        x.append(np.array(exp[0]))
        newQ = np.array(currQ)
        newQ[exp[1]] = exp[2]
        y.append(newQ)

    model.train_on_batch(np.array(x), np.array(y))
    
# train on mini batches
if NUM_RAND_EXPERIENCES != 0:
    for i in range(50):
        if i % 10 == 0:
            print(i, "/", 50)
        train_on_minibatch()

print("Trained the network on some random experiences")

compare = []
mcompare = []

def select_index(prob):
    v = random.random()
    for idx, p in enumerate(prob):
        v -= p
        if v <= 0.0:
            return idx
    return len(prob) - 1

def do_dqn_iteration():
    sched.reset()
    actions = []
    states = []
    rewards = []
    
    # randomly pick between the leading N policies
    policy_idx = random.randint(0, 0)
    
    while not sched.is_done():
        # treat the NN as our policy and move as it directs
        Q = model.predict(np.array([sched.state_vector()]))[0]
        possible_actions = sorted(range(len(sched.actions())),
                                  key=lambda x: Q[x],
                                  reverse=True)

        # "cycle" the list to put our picked policy first
        for i in range(policy_idx):
            possible_actions.append(possible_actions.pop(0))

        action = None
        for proposed in possible_actions:
            r = sched.actions()[proposed]()
            if r != None:
                rewards.append(r)
                action = proposed
                break
            
        actions.append(action)
        states.append(sched.state_vector())
        
    print(actions)
    print("Final cost:", sched.cost(),
          "( latency:", sched.latency(), "),",
          "played policy:", policy_idx,
          "% of heuristic:", 100*(sched.cost() / simple_heuristic_cost))
    compare.append(sched.cost() / simple_heuristic_cost)
    mcompare.append(min(mcompare[-1] if len(mcompare) != 0 else 1000, compare[-1]))

    # create new training points
    for i in range(len(actions)):
        a = actions[i]
        s = states[i]
        r = rewards[i] + GAMMA * sum(rewards[i:])
        yield (s, a, r)

import matplotlib.pyplot as plt
plt.ion()
plt.ylim([0, 8])


for i in range(1000):
    d = list(do_dqn_iteration())
    #for exp in do_dqn_iteration():
    #    experience.append(exp)
    train_on_minibatch(data=d)

    plt.plot(compare, 'b')
    plt.plot([1 for _ in compare], 'r')
    plt.plot(mcompare, 'g')
    plt.draw()
    plt.pause(0.01)
    model.save("dqn_keras.h5")
        
    print("Iteration", i, "complete, now have", len(experience), "experiences")



