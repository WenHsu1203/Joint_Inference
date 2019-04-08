from ValueIteration import *
import numpy as np
class JointInference(object):
	def __init__(self, stateSet, actionSet, truck1Location, truck2Location, allWorlds, mostDesiredFood, middleDesiredFood, leastDesiredFood, actionCost, stayCost, convergenceTolerance, gamma, beta):
		self.stateSet = stateSet
		self.actionSet = actionSet
		self.truck1Location = truck1Location
		self.truck2Location =truck2Location
		self.allWorlds = allWorlds
		self.mostDesiredFood = mostDesiredFood
		self.middleDesiredFood = middleDesiredFood
		self.leastDesiredFood = leastDesiredFood
		self.desiredFoodTables = [mostDesiredFood, middleDesiredFood, leastDesiredFood]
		self.actionCost = actionCost
		self.stayCost = stayCost
		self.convergenceTolerance = convergenceTolerance
		self.gamma = gamma
		self.beta = beta

		self.yellowZone = [(0,0),(3,0),(0,2),(2,1),(1,0),(4,0),(0,1),(1,2),(3,1),(2,0),(4,1),(1,1)]
		self.blueZone = [(2,3),(3,3),(4,3),]
		self.greenZone = [(1,3),(0,3)]
		self.belief_states = [(.17,.17,.17,.17,.17,.17,),
							 (1,0,0,0,0,0),
							 (0,1,0,0,0,0),
							 (0,0,1,0,0,0),
							 (0,0,0,1,0,0),
							 (0,0,0,0,1,0),
							 (0,0,0,0,0,1),
							 (0.5,0.5,0,0,0,0),
							 (0,0,0.5,0.5,0,0),
							 (0,0,0,0,0.5,0.5),
							 (0.5,0,0,0,0,0.5),
							 (0,0.5,0,0.5,0,0),
							 (0,0,0.5,0,0.5,0)]

		self.preferences = [] # size should be 3! The order is K->L->M
		self.labels = ["KLM", "KML", "LKM", "LMK", "MKL", "MLK"]
		self.rewardTables = [] # size should be preferences.size() x allWorlds.size()
		self.beliefRewardTables = [] # size should be preferences.size()
		self.beliefTransitionTable = {} 
		self.policies = [] # size should be preferences.size()

		self.constructRewardTables()
		self.constructBeliefTransitionTables()
		self.constructBeliefRewardTables()
		self.constructPoliciesTables()

	def constructRewardTables(self):
		for i in range(len(self.desiredFoodTables)):
			for j in range (len(self.desiredFoodTables)):
				for k in range (len(self.desiredFoodTables)):
					if j!=i and k!= i and k != j:
						self.preferences.append([self.desiredFoodTables[i], self.desiredFoodTables[j], self.desiredFoodTables[k]])
		# Switch the element 3 and 4 to match the plotting label
		self.preferences[3], self.preferences[4] = self.preferences[4], self.preferences[3]

		for preference in self.preferences:
			for world in self.allWorlds:
				worldDic = {}
				for state in self.stateSet:
					actionDic = {}
					for action in self.actionSet:
						x = state[0] + action[0]
						y = state[1] + action[1]
						outOfBound = (x,y) not in self.stateSet
						if outOfBound:
							(x, y) = state
						actionDic[action] = 0
						if action == (0,0):
							actionDic[action] += self.stayCost
						else:
							actionDic[action] += self.actionCost

						if (x,y) == self.truck1Location:
							if world[0] == 'K':
								actionDic[action] += preference[0]
							elif world[0] == 'L':
								actionDic[action] += preference[1]
							elif world[0] == 'M':
								actionDic[action] += preference[2]
						elif (x,y) == self.truck2Location:
							if world[1] == 'K':
								actionDic[action] += preference[0]
							elif world[1] == 'L':
								actionDic[action] += preference[1]
							elif world[1] == 'M':
								actionDic[action] += preference[2]
					worldDic[state] = actionDic
				self.rewardTables.append(worldDic)


	def constructBeliefRewardTables(self):
		for i in range(len(self.preferences)):
			beliefRewardTable = {}
			for state in self.beliefTransitionTable:
				actionDic = {}
				for action in self.actionSet:
					nextStateDic = {}
					for next_state in self.beliefTransitionTable[state][action]:
						(x,y) = state[0]
						belief = state[1]
						R1 = self.rewardTables[6*i][(x,y)][action]
						R2 = self.rewardTables[6*i+1][(x,y)][action]
						R3 = self.rewardTables[6*i+2][(x,y)][action]
						R4 = self.rewardTables[6*i+3][(x,y)][action]
						R5 = self.rewardTables[6*i+4][(x,y)][action]
						R6 = self.rewardTables[6*i+5][(x,y)][action]
						p1 = belief[0]
						p2 = belief[1]
						p3 = belief[2]
						p4 = belief[3]
						p5 = belief[4]
						p6 = belief[5]
						rho = R1*p1 + R2*p2 + R3*p3 + R4*p4 + R5*p5 + R6*p6
						nextStateDic[next_state] = rho
					actionDic[action] = nextStateDic
				beliefRewardTable[state] = actionDic
			self.beliefRewardTables.append(beliefRewardTable)

	def constructPoliciesTables(self):
		for i in range (len(self.preferences)):
			valueTable = {state:0 for state in self.beliefTransitionTable.keys()}
			ValueIteration = BoltzmannValueIteration(self.beliefTransitionTable, self.beliefRewardTables[i], valueTable, self.convergenceTolerance, self.gamma, self.beta)
			_, policy = ValueIteration()
			self.policies.append(policy)

	def inference (self, initial_belief, world, trajectory):
		possible_belief_states = []
		if world == self.allWorlds[0]:
			possible_belief_states = [self.belief_states[1], self.belief_states[7],self.belief_states[10]]
		elif world == self.allWorlds[1]:
			possible_belief_states = [self.belief_states[2], self.belief_states[7],self.belief_states[11]]
		elif world == self.allWorlds[2]:
			possible_belief_states = [self.belief_states[3], self.belief_states[8],self.belief_states[12]]
		elif world == self.allWorlds[3]:
			possible_belief_states = [self.belief_states[4], self.belief_states[8],self.belief_states[11]]
		elif world == self.allWorlds[4]:
			possible_belief_states = [self.belief_states[5], self.belief_states[9],self.belief_states[12]]
		elif world == self.allWorlds[5]:
			possible_belief_states = [self.belief_states[6], self.belief_states[9],self.belief_states[10]]

		P = [ [] for i in range(len(self.preferences)) ]
		Beliefs = []
		sum_P  = [0] * len(trajectory)
		for preference in range(len(self.preferences)):
			for length in range(1,len(trajectory)+1):
				p, beliefs = self.probability(initial_belief, preference, trajectory[:length], possible_belief_states)
				if length == len(trajectory):
					Beliefs = beliefs
				P[preference].append(p)
				sum_P[length-1] += p
		for i in range(len(self.beliefRewardTables)):
			for j in range(len(sum_P)):
				P[i][j] = P[i][j]/ sum_P[j]
		return np.transpose(np.asarray(P)), np.transpose(np.array(Beliefs))


	def probability(self, initial_belief, preference, trajectory, possible_belief_states):
		belief = initial_belief
		beliefs = []
		P = 1
		for i in range(1,len(trajectory)):
			(x,y) = trajectory[i-1]
			state = ((x,y), belief)
			beliefs.append(belief)
			p = 0
			for action in self.beliefTransitionTable[state]:
				for next_state, prob in self.beliefTransitionTable[state][action].items():
					if (next_state[0] == trajectory[i] and next_state[1] in possible_belief_states):
						p += prob*self.policies[preference][state][action]
						belief = next_state[1]
			P *= p
		beliefs.append(belief)
		return P/len(self.preferences), beliefs

	def constructBeliefTransitionTables(self):
		for i in range(len(self.belief_states)):
			for state in self.stateSet:
				initial_belief = ((state), self.belief_states[i])
				actionDic = {}
				for action in self.actionSet:
					x = state[0] + action[0]
					y = state[1] + action[1]
					outOfBound = (x,y) not in self.stateSet
					if outOfBound:
						(x, y) = state
					nextInYellow = (x,y) in self.yellowZone
					nextInGreen = (x,y) in self.greenZone
					nextInBlue = (x,y) in self.blueZone
					if i == 0:
						if nextInYellow:
							actionDic[action] = {((x,y),self.belief_states[7]):.33, ((x,y),self.belief_states[8]):.33, ((x,y),self.belief_states[9]):.33}
						elif nextInBlue:
							actionDic[action] = {((x,y),self.belief_states[10]):.33, ((x,y),self.belief_states[11]):.33, ((x,y),self.belief_states[12]):.33}
						elif nextInGreen:
							actionDic[action] = {((x,y),self.belief_states[1]):.17, ((x,y),self.belief_states[2]):.17, ((x,y),self.belief_states[3]):.17, ((x,y),self.belief_states[4]):.17, ((x,y),self.belief_states[5]):.17, ((x,y),self.belief_states[6]):.17}
					elif i>0 and i <=6:
						actionDic[action] = {((x,y),self.belief_states[i]):1}
					elif i == 7:
						if nextInYellow:
							actionDic[action] = {((x,y),self.belief_states[i]):1}
						else:
							actionDic[action] = {((x,y),self.belief_states[1]):.5, ((x,y),self.belief_states[2]):.5}
					elif i == 8:
						if nextInYellow:
							actionDic[action] = {((x,y),self.belief_states[i]):1}
						else:
							actionDic[action] = {((x,y),self.belief_states[3]):.5, ((x,y),self.belief_states[4]):.5}
					elif i == 9:
						if nextInYellow:
							actionDic[action] = {((x,y),self.belief_states[i]):1}
						else:
							actionDic[action] = {((x,y),self.belief_states[5]):.5, ((x,y),self.belief_states[6]):.5}
					elif i == 10:
						if nextInBlue:
							actionDic[action] = {((x,y),self.belief_states[i]):1}
						else:
							actionDic[action] = {((x,y),self.belief_states[1]):.5, ((x,y),self.belief_states[6]):.5}
					elif i == 11:
						if nextInBlue:
							actionDic[action] = {((x,y),self.belief_states[i]):1}
						else:
							actionDic[action] = {((x,y),self.belief_states[2]):.5, ((x,y),self.belief_states[4]):.5}
					elif i == 12:
						if nextInBlue:
							actionDic[action] = {((x,y),self.belief_states[i]):1}
						else:
							actionDic[action] = {((x,y),self.belief_states[3]):.5, ((x,y),self.belief_states[5]):.5}


				self.beliefTransitionTable[initial_belief] = actionDic

