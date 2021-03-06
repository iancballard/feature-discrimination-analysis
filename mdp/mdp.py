import collections, util, math, random
import numpy as np

############################################################

############################################################
# Problem 2a

class ValueIteration(util.MDPAlgorithm):

    # Implement value iteration.  First, compute V_opt using the methods 
    # discussed in class.  Once you have computed V_opt, compute the optimal 
    # policy pi.  Note that ValueIteration is an instance of util.MDPAlgrotithm, 
    # which means you will need to set pi and V (see util.py).
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        
        v_opt_old = dict.fromkeys(mdp.states, 0)
        self.pi = dict.fromkeys(mdp.states, None)
        
        max_change = float('inf')
        while max_change > epsilon: #while v_opt is changing, keep looping
            self.V = dict.fromkeys(mdp.states, -float('inf'))
 
            for state in mdp.states: #loop through states
                for action in mdp.actions(state): #loop though actions
    
                    q_opt = 0 #for adding up the possibilities from successor states
                    for succesor in mdp.succAndProbReward(state, action): #loop through sucessor states
                        q_opt += succesor[1] * (succesor[2] + mdp.discount()*v_opt_old[succesor[0]])    

                    if q_opt > self.V[state]: #take best action and set optimal policy
                        self.V[state] = q_opt
                        self.pi[state] = action                    
            
            #compute stopping criterion
            delta_dict = {key: abs(self.V[key] - v_opt_old.get(key, 0)) for key in self.V.keys()}
            max_change = delta_dict[max(delta_dict, key=delta_dict.get)]                     
            v_opt_old = self.V.copy() 
        

############################################################
class ser_fd(util.MDP):
    def __init__(self):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        """

    # Return the start state. First is a indicator of whether on upper or lower trajectory, second is step in the task
    def startState(self):
        return (0, 0)  

    # Return set of actions possible from |state|.
    def actions(self, state):
        A, stage = state
        if stage == 0 or stage == 1:
            return ['Next']
        else:
            return ['Go', 'Dont']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|. 
    def succAndProbReward(self, state, action):
        succ_prob_reward = []
        stim, stage = state
        
        #stage 0, A
        #stage 1, B which leads to dot
        #stage 2, B which doesnt lead to dot
        
        if stage is None:
            return []
        elif stage == 0:
            succ_prob_reward.append(((0,1),.5,0)) 
            succ_prob_reward.append(((1,1),.5,0))
        elif stage == 1 and stim == 0: #A trial
            succ_prob_reward.append(((0,2),.95,0)) 
            succ_prob_reward.append(((1,2),.05,0)) 
        elif stage == 1 and stim == 1:
            succ_prob_reward.append(((0,2),.05,0)) 
            succ_prob_reward.append(((1,2),.95,0))
        elif stage == 2 and stim ==  0 and action == 'Go': #B+ and respond
            succ_prob_reward.append(((None,None),.95,1)) 
            succ_prob_reward.append(((None,None),.05,-1)) 
        elif stage == 2 and stim ==  0 and action == 'Dont': #B+ and dont respond
            succ_prob_reward.append(((None,None),.95,-.1)) 
            succ_prob_reward.append(((None,None),.05,.1)) 
        elif stage == 2 and stim ==  1 and action == 'Go': #B- and respond
            succ_prob_reward.append(((None,None),.95,-1)) 
            succ_prob_reward.append(((None,None),.05,1)) 
        elif stage == 2 and stim ==  1 and action == 'Dont': #B- and dont respond
            succ_prob_reward.append(((None,None),.95,.1)) 
            succ_prob_reward.append(((None,None),.05,-.1)) 

        return succ_prob_reward

    def discount(self):
        return 1

############################################################
class ser_fd_features(util.MDP):
    def __init__(self):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        """

    # Return the start state. First is a indicator of whether on upper or lower trajectory, second is step in the task
    def startState(self):
        return (0,0,0)  

    # Return set of actions possible from |state|.
    def actions(self, state):
        A,B,stage = state
        if stage == 0:
            return ['Next']
        else:
            return ['Go', 'Dont']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|. 
    def succAndProbReward(self, state, action):
        succ_prob_reward = []
        A,B,stage = state
        
        if stage is None:
            return []
        elif stage == 0:
            succ_prob_reward.append(((1,100,1),.5,0)) 
            succ_prob_reward.append(((0,100,1),.5,0))
        elif stage == 1 and A == 1 and action == 'Dont': #A trial
            succ_prob_reward.append(((2,1,2),1,0)) 
        elif stage == 1 and A == 1 and action == 'Go': #A trial
            succ_prob_reward.append(((2,1,2),1,-1))
        elif stage == 1 and A == 0 and action == 'Dont':
            succ_prob_reward.append(((3,1,2),1,0)) 
        elif stage == 1 and A == 0 and action == 'Go':
            succ_prob_reward.append(((3,1,2),1,-1))
            
        # elif stage == 2 and A == 2 and action == 'Go': #A -> B+ and respond
        #     succ_prob_reward.append(((None,None,None),.95,1))
        #     succ_prob_reward.append(((None,None,None),.05,-1))
        # elif stage == 2 and A == 2 and action == 'Dont': #A -> B+ and dont respond
        #     succ_prob_reward.append(((None,None,None),.95,-.1))
        #     succ_prob_reward.append(((None,None,None),.05,.1))
        # elif stage == 2 and A == 3 and action == 'Go': #B- and respond
        #     succ_prob_reward.append(((None,None,None),.95,-1))
        #     succ_prob_reward.append(((None,None,None),.05,1))
        # elif stage == 2 and A == 3 and action == 'Dont': #B- and dont respond
        #     succ_prob_reward.append(((None,None,None),.95,.1))
        #     succ_prob_reward.append(((None,None,None),.05,-.1))

        elif stage == 2 and A == 2 and action == 'Go': #A -> B+ and respond
            succ_prob_reward.append(((None,None,None),.95,-1)) 
            succ_prob_reward.append(((None,None,None),.05,1)) 
        elif stage == 2 and A == 2 and action == 'Dont': #A -> B+ and dont respond
            succ_prob_reward.append(((None,None,None),.95,.1)) 
            succ_prob_reward.append(((None,None,None),.05,-.1)) 
        elif stage == 2 and A == 3 and action == 'Go': #B- and respond
            succ_prob_reward.append(((None,None,None),.95,1)) 
            succ_prob_reward.append(((None,None,None),.05,-1)) 
        elif stage == 2 and A == 3 and action == 'Dont': #B- and dont respond
            succ_prob_reward.append(((None,None,None),.95,-.1)) 
            succ_prob_reward.append(((None,None,None),.05,.1)) 
        return succ_prob_reward

    def discount(self):
        return 1
 
############################################################
class sim_fd_features(util.MDP):
    def __init__(self):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        """

    # Return the start state. First is a indicator of whether on upper or lower trajectory, second is step in the task
    def startState(self):
        return (0,0,0)  

    # Return set of actions possible from |state|.
    def actions(self, state):
        A,B,stage = state
        if stage == 0:
            return ['Next']
        else:
            return ['Go', 'Dont']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|. 
    def succAndProbReward(self, state, action):
        succ_prob_reward = []
        A,B,stage = state
        
        if stage is None:
            return []
        elif stage == 0:
            succ_prob_reward.append(((1,1,1),.5,0)) 
            succ_prob_reward.append(((0,1,1),.5,0))
        elif stage == 1 and A == 1 and action == 'Go': #AB+ and respond
            succ_prob_reward.append(((None,None,None),.95,1))
            succ_prob_reward.append(((None,None,None),.05,-1))
        elif stage == 1 and A == 1 and action == 'Dont': #A -> B+ and dont respond
            succ_prob_reward.append(((None,None,None),.95,-.1))
            succ_prob_reward.append(((None,None,None),.05,.1))
        elif stage == 1 and A == 0 and action == 'Go': #B- and respond
            succ_prob_reward.append(((None,None,None),.95,-1))
            succ_prob_reward.append(((None,None,None),.05,1))
        elif stage == 1 and A == 0 and action == 'Dont': #B- and dont respond
            succ_prob_reward.append(((None,None,None),.95,.1))
            succ_prob_reward.append(((None,None,None),.05,-.1))
        # elif stage == 1 and A == 1 and action == 'Go': #AB+ and respond
        #     succ_prob_reward.append(((None,None,None),.95,-1))
        #     succ_prob_reward.append(((None,None,None),.05,1))
        # elif stage == 1 and A == 1 and action == 'Dont': #A -> B+ and dont respond
        #     succ_prob_reward.append(((None,None,None),.95,.1))
        #     succ_prob_reward.append(((None,None,None),.05,-.1))
        # elif stage == 1 and A == 0 and action == 'Go': #B- and respond
        #     succ_prob_reward.append(((None,None,None),.95,1))
        #     succ_prob_reward.append(((None,None,None),.05,-1))
        # elif stage == 1 and A == 0 and action == 'Dont': #B- and dont respond
        #     succ_prob_reward.append(((None,None,None),.95,-.1))
        #     succ_prob_reward.append(((None,None,None),.05,.1))

        return succ_prob_reward

    def discount(self):
        return 1       

class phase_2_blocking(util.MDP):
    def __init__(self):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        """

    # Return the start state. First is a indicator of whether on upper or lower trajectory, second is step in the task
    def startState(self):
        return (0,0,0)  

    # Return set of actions possible from |state|.
    def actions(self, state):
        A,B,stage = state
        if stage == 0:
            return ['Next']
        else:
            return ['Go', 'Dont']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|. 
    def succAndProbReward(self, state, action):
        succ_prob_reward = []
        A,B,stage = state
        
        if stage is None:
            return []
        elif stage == 0:
            succ_prob_reward.append(((1,10,1),.5,0)) 
            succ_prob_reward.append(((11,12,1),.5,0))
        elif stage == 1 and A == 1 and action == 'Go': #AC+ and respond
            succ_prob_reward.append(((None,None,None),.95,1)) 
            succ_prob_reward.append(((None,None,None),.05,-1)) 
        elif stage == 1 and A == 1 and action == 'Dont': #AC+ and dont respond
            succ_prob_reward.append(((None,None,None),.95,-.1)) 
            succ_prob_reward.append(((None,None,None),.05,.1)) 
        elif stage == 1 and A == 11 and action == 'Go': #DE- and respond
            succ_prob_reward.append(((None,None,None),.95,-1)) 
            succ_prob_reward.append(((None,None,None),.05,1)) 
        elif stage == 1 and A == 11 and action == 'Dont': #DE- and dont respond
            succ_prob_reward.append(((None,None,None),.95,.1)) 
            succ_prob_reward.append(((None,None,None),.05,-.1)) 

        return succ_prob_reward
        
    def discount(self):
        return 1   

############################################################
# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, init_weights = collections.Counter(), explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = init_weights
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        if newState == None:
            maxQ = 0
        else:
            maxQ = max((self.getQ(newState, act), act) for act in self.actions(newState))[0]
        r = reward + self.discount*maxQ - self.getQ(state, action)
        
        for f, v in self.featureExtractor(state, action):
            self.weights[f] = self.weights[f] + self.getStepSize() * r * v

# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
def serFeatureExtractor(state,action):
    A,B,stage = state
    if stage == None:
        return []
    featureValue = 1 #indicator features
    features = []
    if A == 0:
        featureKey = ('noA',action)
    elif A == 1:
        featureKey = ('A',action)
    elif A == 2:
        featureKey = ('pastA',action)        
    elif A==3:
        featureKey = ('no_pastA',action)        
    elif A == 11:
        featureKey = ('D',action)        
        
    features.append((featureKey,featureValue))
    
    if B == 0:
        featureKey = ('noB',action)
    elif B==1:
        featureKey = ('B',action)
    elif B==10:
        featureKey = ('C',action)
    elif B==12:
        featureKey = ('E',action)
    features.append((featureKey,featureValue))
        
    #conjunctive features
    if A == 2 and B == 1:
        featureKey = ('pastA','B',action)  
    if A == 1 and B == 1:
        featureKey = ('A','B',action)
    if A == 1 and B == 10:
        featureKey = ('A','C',action)
    if A == 11 and B == 12:
        featureKey = ('D','E',action)
    features.append((featureKey,featureValue))
    
    return features
  
    
# # run algorithms
# mdp = sim_fd_features()
# QLearning = QLearningAlgorithm(mdp.actions,mdp.discount(),serFeatureExtractor)
# rew_q = util.simulate(mdp, QLearning,numTrials=1000, maxIterations=1000)
#
# valueIter = ValueIteration()
# valueIter.solve(mdp)
# print valueIter.V
# print QLearning.weights
#
#
# #run subsequent
# mdp_phase2 = phase_2_blocking()
# QLearning_phase2 =  QLearningAlgorithm(mdp.actions,mdp.discount(),serFeatureExtractor,init_weights = QLearning.weights)
# rew_q2= util.simulate(mdp_phase2, QLearning_phase2,numTrials=25, maxIterations=1000)
#
# #get state q values
# for state in valueIter.pi:
#     print state
#     # print valueIter.pi[state]
#
#     actions = QLearning.actions(state)
#     if len(actions) > 1:
#        for action in actions:
#            print action
#            print QLearning.getQ(state,action)
#     else:
#         print actions
#         print QLearning.getQ(state,actions[0])
#
# ##get feature values
# print QLearning_phase2.weights
# print QLearning_phase2.weights[('C','Go')]
# print QLearning_phase2.weights[('C','Dont')]


c_go_out = '/Users/ianballard/Dropbox/fd/c_go_ser_fn.txt'
c_dont_out = '/Users/ianballard/Dropbox/fd/c_dont_ser_fn.txt'
num_iters = 100
c_go_weights = []
c_dont_weights = []
for i in range(0,num_iters):
    mdp = ser_fd_features()
    QLearning = QLearningAlgorithm(mdp.actions,mdp.discount(),serFeatureExtractor)
    rew_q = util.simulate(mdp, QLearning,numTrials=1000, maxIterations=1000)

    #run subsequent
    mdp_phase2 = phase_2_blocking()
    QLearning_phase2 =  QLearningAlgorithm(mdp.actions,mdp.discount(),serFeatureExtractor,init_weights = QLearning.weights)
    rew_q2= util.simulate(mdp_phase2, QLearning_phase2,numTrials=25, maxIterations=1000)
    
    c_go_weights.append(QLearning_phase2.weights[('C','Go')])
    c_dont_weights.append(QLearning_phase2.weights[('C','Dont')])
    
np.savetxt(c_go_out,c_go_weights)    
np.savetxt(c_dont_out,c_dont_weights)    
