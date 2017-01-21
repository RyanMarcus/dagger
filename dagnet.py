import numpy as np
from graph import DAG
from sequence_processor import Scheduler
from keras.models import Sequential
from keras.layers import Dense, Activation
import random

def load_dag():
    adj = np.array([[0, 1, 2, 0], [0, 0, 0, 5], [0, 0, 0, 3], [0, 0, 0, 0]])
    w = np.array([[3, 2], [4, 1], [1, 1], [8, 7]])
    d = DAG(w, adj, (1, 5))
    return d

def get_scheduler(dag, deadline):
    return Scheduler(dag, deadline, 2)


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
    print(cost)

    for i in range(len(actions)-1):
        yield (states[i], actions[i], 0, states[i+1])

    # add the last experience with the reward
    yield (states[-1], actions[-1], cost, sched.state_vector())


dag = load_dag()
sched = get_scheduler(dag, 50)
model = Sequential()
total_inputs = sched.state_vector().shape[0]
total_outputs = len(sched.actions())


# build the net
model.add(Dense(total_inputs * 2, input_dim=total_inputs))
model.add(Dense(total_inputs * 2, activation="relu"))
model.add(Dense(total_outputs))
model.add(Activation("softmax"))

model.compile("RMSprop", "mse")



# first, generate some random data to get us going
# tuples: (before state, action, reward, after state)
experience = []
for i in range(200):
    experience.extend(list(generate_random_data(sched)))
 

print("Gathered", len(experience), "random experiences")

GAMMA = 0.5
EPSILON = 0.02
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
    
# train on 5 mini batches
for i in range(5):
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
            action = np.argmax(Q[0])
            
        actions.append(action)
        states.append(sched.state_vector())
        sched.actions()[action]()
        
    cost = sched.cost()

    for i in range(len(actions)-1):
        yield (states[i], actions[i], 0, states[i+1])

    # add the last experience with the reward
    print("DQN iteration got cost:", cost)
    yield (states[-1], actions[-1], cost, sched.state_vector())
    

for i in range(10):
    experience.extend(list(do_dqn_iteration()))
    train_on_minibatch()
    print("Iteration", i, "complete, now have", len(experience), "experiences")


