import random
import numpy as np
from itertools import product as itertoolsProduct
from environment import Agent, Environment, TrafficLight
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    #An agent that learns to drive in the smartcab world.
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        self.netReward = 0                  # Count the netReward in each trial
        self.numberOfPenalties = 0          # Count the number of Penalities received in each trial #                 
        self.numberOfFailures = 0           # Count the number of failures in all trials
        self.numberOfPenaltiesList = []
        self.numberOfFailuresList = []
        self.previousState = None                      
        self.previousAction = None 
        self.previousReward = None

        # ================== Parameters for Q-Learning ================== #
        self.epsilon = 0.1
        self.lr = 0.9
        self.alpha = 0.2

        # ================== Initializing the Q-Table ================== #
        # Define the components of my state tuple (next_waypoint, light, oncoming, left, right) 
        # and the possible valuess for each element 
        state_components = [Environment.valid_actions, ['red', 'green'], Environment.valid_actions, Environment.valid_actions, Environment.valid_actions]
        
        # Generate all the posssible states (512) that can be formed with the components 
        self.valid_state = list(itertoolsProduct(*state_components))

        # Initialize the Q-Tables with ones, dimension = all possible states x  all possible actions   
        self.Q = np.ones((len(self.valid_state), len(Environment.valid_actions))) 
 
    # ========================================================================== #
    # This function is run after a trial ended and before the new trial started
    # ========================================================================== #
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # Print out some statistic to evaluate the performance of the strategy 
        print ("=================================================================================")
        print ("Number of failures: %3d, Net Reward: %3.3f, Penalties received: %3d" % (self.numberOfFailures, self.netReward, self.numberOfPenalties))
        print ("=================================================================================")
        self.numberOfPenaltiesList.append(self.numberOfPenalties)
        self.numberOfFailuresList.append(self.numberOfFailures)
        self.netReward = 0
        self.numberOfPenalties = 0
        self.previousState = None
        self.previousAction = None
        self.previousReward = None


    # ============================================================== #
    # Find the index of the state in the Q-table given a state tuple
    # ============================================================== #
    # [INPUT]  Tuple:   A state tuple
    # [OUTPUT] Integer: The index of the state in the Q-table
    # ============================================================== #
    def getStateNumber(self, state):
        return self.valid_state.index(state)

    # ============================================================== #
    # Find the index of the action in the Q-table given an action
    # ============================================================== #
    # [INPUT]  String:  a string of action 
    # [OUTPUT] Integer: The index of the action in the Q-table
    # ============================================================== #
    def getActionNumber(self, action):
        return Environment.valid_actions.index(action)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # ======================================================== #
        # STEP 1: Updating state 
        # ======================================================== #
        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])
        
        # ======================================================== #
        # STEP 2: Select action according to your policy
        # ======================================================== #
        # Get a random number, if less than epsilon, randomly choose an action #
        if (random.random() < self.epsilon):
            action = random.choice(Environment.valid_actions)
        # Otherwise, Choose the best action from the Q-Table #
        else:
            actionIndex = np.argmax(self.Q[self.getStateNumber(self.state)])
            action = Environment.valid_actions[actionIndex]

        # ======================================================== #
        # STEP 3: Execute action and get reward
        # ======================================================== #
        reward = self.env.act(self, action)
        self.netReward += reward                # Calculate the netReward as well
        if (reward < 0):
            self.numberOfPenalties += reward

        # ======================================================== #
        # STEP 4: Q-Learning: Updating the Q table
        # ======================================================== #
        # TODO: Learn policy based on state, action, reward
        stateIndex = self.getStateNumber(self.state)

        if self.previousState != None:
            previousStateIndex = self.getStateNumber(self.previousState)
            previousActionIndex = self.getActionNumber(self.previousAction)
            self.Q[previousStateIndex, previousActionIndex] = (1-self.lr) * self.Q[previousStateIndex, previousActionIndex] + \
                                                              (self.lr) * (self.previousReward + (self.alpha * np.max(self.Q[stateIndex])))

        # Save the current state, action and reward for the next update #
        self.previousState = self.state
        self.previousAction = action
        self.previousReward = reward

        if deadline == 0:
            self.numberOfFailures += 1
        #print "{} LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(self.netReward, deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.00001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    
    # Print summary #
    allPenalities = a.numberOfPenaltiesList
    allFailures = a.numberOfFailuresList
    numberOfTrials = float(len(allFailures))
    numberOfFailures = float(allFailures[-1])
    numberOfSuccess = numberOfTrials - numberOfFailures
    numberOfSuccessFirstHalf = ((numberOfTrials) / 2) - float(allFailures[len(allFailures)/2])
    numberOfSuccessSecondHalf = numberOfSuccess - numberOfSuccessFirstHalf
    print ("=================================================================================")
    print ("SUMMARY")
    print ("=================================================================================")
    print ("Total Penalities received = %3.2f" % (sum(allPenalities)))
    print ("\tPenalities received in the first half of trials  = %3.2f" % (sum(allPenalities[:len(allPenalities)/2])))
    print ("\tPenalities received in the second half of trials = %3.2f" % (sum(allPenalities[len(allPenalities)/2:])))
    print ("Success Rate: %3.2f%%" % (numberOfSuccess/numberOfTrials*100))
    print ("\tSuccess Rate of the first half : %3.2f%%" % (numberOfSuccessFirstHalf/(numberOfTrials/2)*100))
    print ("\tSuccess Rate of the second half: %3.2f%%" % (numberOfSuccessSecondHalf/(numberOfTrials/2)*100))
    
if __name__ == '__main__':
    run()
