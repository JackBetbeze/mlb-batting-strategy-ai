import csv
import random
import copy


class Pitcher:
    def __init__(self, data_file, train_season, test_season):

        self.train_season = str(train_season)
        self.test_season = str(test_season)

        # The predetermined rewards that are given when reaching each state
        self.rewards = {
            '00': 0,
            '01': 0,
            '02': 0,
            '10': 0,
            '11': 0,
            '12': 0,
            '20': 0,
            '21': 0,
            '22': 0,
            '30': 0,
            '31': 0,
            '32': 0,
            'O': 0,
            'W': 1,
            'S': 2,
            'D': 3,
            'T': 4,
            'HR': 5
        }

        # The best action to take in each state
        # Refined using Value Iteration
        self.policy = {
            '00': None,
            '01': None,
            '02': None,
            '10': None,
            '11': None,
            '12': None,
            '20': None,
            '21': None,
            '22': None,
            '30': None,
            '31': None,
            '32': None
        }

        # The max reward that could be achieved while committing an action in each state
        # Refined using Value Iteration
        self.max_reward = {
            '00': 0.0,
            '01': 0.0,
            '02': 0.0,
            '10': 0.0,
            '11': 0.0,
            '12': 0.0,
            '20': 0.0,
            '21': 0.0,
            '22': 0.0,
            '30': 0.0,
            '31': 0.0,
            '32': 0.0
        }

        # The expected reward when committing an action recommended by the policy in each state
        # Refined using Policy Evaluation
        self.expected_reward = {
            '00': 0.0,
            '01': 0.0,
            '02': 0.0,
            '10': 0.0,
            '11': 0.0,
            '12': 0.0,
            '20': 0.0,
            '21': 0.0,
            '22': 0.0,
            '30': 0.0,
            '31': 0.0,
            '32': 0.0
        }

        # TRAINING PROBABILITIES
        # Probabilities of transitioning from one count to another count/terminal if the batter swings
        self.ptrain_swing = {
            '00': {'01': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '01': {'02': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '02': {'02': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '10': {'11': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '11': {'12': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '12': {'12': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '20': {'21': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '21': {'22': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '22': {'22': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '30': {'31': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '31': {'32': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '32': {'32': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0}
        }

        # Probabilities of transitioning from one count to another count/terminal if the batter stands
        self.ptrain_stand = {
            '00': {'10': 0.0, '01': 0.0},
            '01': {'11': 0.0, '02': 0.0},
            '02': {'12': 0.0, 'O': 0.0},
            '10': {'20': 0.0, '11': 0.0},
            '11': {'21': 0.0, '12': 0.0},
            '12': {'22': 0.0, 'O': 0.0},
            '20': {'30': 0.0, '21': 0.0},
            '21': {'31': 0.0, '22': 0.0},
            '22': {'32': 0.0, 'O': 0.0},
            '30': {'31': 0.0, 'W': 0.0},
            '31': {'32': 0.0, 'W': 0.0},
            '32': {'W': 0.0, 'O': 0.0}
        }

        # pitch counting banks to help calculate probabilities
        train_swing_count = {
            '00': {'01': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '01': {'02': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '02': {'02': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '10': {'11': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '11': {'12': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '12': {'12': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '20': {'21': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '21': {'22': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '22': {'22': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '30': {'31': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '31': {'32': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '32': {'32': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0}
        }

        train_stand_count = {
            '00': {'10': 0, '01': 0, 'W': 0},
            '01': {'11': 0, '02': 0, 'W': 0},
            '02': {'12': 0, 'O': 0, 'W': 0},
            '10': {'20': 0, '11': 0, 'W': 0},
            '11': {'21': 0, '12': 0, 'W': 0},
            '12': {'22': 0, 'O': 0, 'W': 0},
            '20': {'30': 0, '21': 0, 'W': 0},
            '21': {'31': 0, '22': 0, 'W': 0},
            '22': {'32': 0, 'O': 0, 'W': 0},
            '30': {'31': 0, 'W': 0},
            '31': {'32': 0, 'W': 0},
            '32': {'W': 0, 'O': 0}
        }

        # TESTING PROBABILITIES
        # Probabilities of transitioning from one count to another count/terminal if the batter swings
        self.ptest_swing = {
            '00': {'01': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '01': {'02': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '02': {'02': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '10': {'11': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '11': {'12': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '12': {'12': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '20': {'21': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '21': {'22': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '22': {'22': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '30': {'31': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '31': {'32': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0},
            '32': {'32': 0.0, 'O': 0.0, 'S': 0.0, 'D': 0.0, 'T': 0.0, 'HR': 0.0}
        }

        # Probabilities of transitioning from one count to another count/terminal if the batter stands
        self.ptest_stand = {
            '00': {'10': 0.0, '01': 0.0},
            '01': {'11': 0.0, '02': 0.0},
            '02': {'12': 0.0, 'O': 0.0},
            '10': {'20': 0.0, '11': 0.0},
            '11': {'21': 0.0, '12': 0.0},
            '12': {'22': 0.0, 'O': 0.0},
            '20': {'30': 0.0, '21': 0.0},
            '21': {'31': 0.0, '22': 0.0},
            '22': {'32': 0.0, 'O': 0.0},
            '30': {'31': 0.0, 'W': 0.0},
            '31': {'32': 0.0, 'W': 0.0},
            '32': {'W': 0.0, 'O': 0.0}
        }

        # pitch counting banks to help calculate probabilities
        test_swing_count = {
            '00': {'01': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '01': {'02': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '02': {'02': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '10': {'11': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '11': {'12': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '12': {'12': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '20': {'21': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '21': {'22': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '22': {'22': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '30': {'31': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '31': {'32': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0},
            '32': {'32': 0, 'O': 0, 'S': 0, 'D': 0, 'T': 0, 'HR': 0}
        }

        test_stand_count = {
            '00': {'10': 0, '01': 0, 'W': 0},
            '01': {'11': 0, '02': 0, 'W': 0},
            '02': {'12': 0, 'O': 0, 'W': 0},
            '10': {'20': 0, '11': 0, 'W': 0},
            '11': {'21': 0, '12': 0, 'W': 0},
            '12': {'22': 0, 'O': 0, 'W': 0},
            '20': {'30': 0, '21': 0, 'W': 0},
            '21': {'31': 0, '22': 0, 'W': 0},
            '22': {'32': 0, 'O': 0, 'W': 0},
            '30': {'31': 0, 'W': 0},
            '31': {'32': 0, 'W': 0},
            '32': {'W': 0, 'O': 0}
        }

        # Open player stats file, load training data for given season
        with open(data_file) as file:
            pre_reader = csv.reader(file)
            reader = []
            for entry in pre_reader:
                reader.append(entry)
            # Loop backwards(upwards) through file to iterate through pitches correctly
            for i in range(len(reader)-1, -1, -1):
                curr_batter = reader[i][6]
                curr_inning = reader[i][35]
                if reader[i][26] == self.train_season:
                    count = [reader[i][24], reader[i][25]]
                    count_hash = count[0] + count[1]
                    outcome = [reader[i - 1][24], reader[i - 1][25]]
                    outcome_hash = outcome[0] + outcome[1]
                    if reader[i - 1][6] != curr_batter:
                        if reader[i][21] == 'X':
                            if reader[i][8] == 'single':
                                train_swing_count[count_hash]['S'] += 1
                            elif reader[i][8] == 'double':
                                train_swing_count[count_hash]['D'] += 1
                            elif reader[i][8] == 'triple':
                                train_swing_count[count_hash]['T'] += 1
                            elif reader[i][8] == 'home_run':
                                train_swing_count[count_hash]['HR'] += 1
                            else:
                                train_swing_count[count_hash]['O'] += 1
                        elif reader[i][21] == 'S':
                            if 'called_strike' in reader[i][9]:
                                train_stand_count[count_hash]['O'] += 1
                            elif 'swinging_strike' in reader[i][9]:
                                train_swing_count[count_hash]['O'] += 1
                        elif reader[i][21] == 'B':
                            train_stand_count[count_hash]['W'] += 1
                    else:
                        if reader[i-1][35] != curr_inning:
                            pass
                        elif reader[i][21] == 'S':
                            if 'called_strike' in reader[i][9]:
                                train_stand_count[count_hash][outcome_hash] += 1
                            elif 'swinging_strike' in reader[i][9] or 'foul' == reader[i][9]:
                                train_swing_count[count_hash][outcome_hash] += 1
                        elif reader[i][21] == 'B':
                            train_stand_count[count_hash][outcome_hash] += 1
                elif reader[i][26] == self.test_season:
                    count = [reader[i][24], reader[i][25]]
                    count_hash = count[0] + count[1]
                    outcome = [reader[i - 1][24], reader[i - 1][25]]
                    outcome_hash = outcome[0] + outcome[1]
                    if reader[i - 1][6] != curr_batter:
                        if reader[i][21] == 'X':
                            if reader[i][8] == 'single':
                                test_swing_count[count_hash]['S'] += 1
                            elif reader[i][8] == 'double':
                                test_swing_count[count_hash]['D'] += 1
                            elif reader[i][8] == 'triple':
                                test_swing_count[count_hash]['T'] += 1
                            elif reader[i][8] == 'home_run':
                                test_swing_count[count_hash]['HR'] += 1
                            else:
                                test_swing_count[count_hash]['O'] += 1
                        elif reader[i][21] == 'S':
                            if 'called_strike' in reader[i][9]:
                                test_stand_count[count_hash]['O'] += 1
                            elif 'swinging_strike' in reader[i][9]:
                                test_swing_count[count_hash]['O'] += 1
                        elif reader[i][21] == 'B':
                            test_stand_count[count_hash]['W'] += 1
                    else:
                        if reader[i-1][35] != curr_inning:
                            pass
                        elif reader[i][21] == 'S':
                            if 'called_strike' in reader[i][9]:
                                test_stand_count[count_hash][outcome_hash] += 1
                            elif 'swinging_strike' in reader[i][9] or 'foul' == reader[i][9]:
                                test_swing_count[count_hash][outcome_hash] += 1
                        elif reader[i][21] == 'B':
                            test_stand_count[count_hash][outcome_hash] += 1

        # Compute probabilities
        for i in self.ptrain_swing:
            sum1 = 0
            for j in train_swing_count[i]:
                sum1 += train_swing_count[i][j]
            for j in self.ptrain_swing[i]:
                self.ptrain_swing[i][j] = (float(train_swing_count[i][j])/float(sum1))

        for i in self.ptrain_stand:
            sum2 = 0
            for j in train_stand_count[i]:
                sum2 += train_stand_count[i][j]
            for j in self.ptrain_stand[i]:
                self.ptrain_stand[i][j] = (float(train_stand_count[i][j])/float(sum2))

        for i in self.ptest_swing:
            sum3 = 0
            for j in test_swing_count[i]:
                sum3 += test_swing_count[i][j]
            for j in self.ptest_swing[i]:
                self.ptest_swing[i][j] = (float(test_swing_count[i][j]) / float(sum3))

        for i in self.ptest_stand:
            sum4 = 0
            for j in test_stand_count[i]:
                sum4 += test_stand_count[i][j]
            for j in self.ptest_stand[i]:
                self.ptest_stand[i][j] = (float(test_stand_count[i][j]) / float(sum4))

    # Use value iteration to find the max rewards for each state and the best action to take in each state,
    # Returns number of iterations until convergence
    def solveActions(self, epsilon):
        delta = 100
        iterations = 0
        while delta >= epsilon:
            iterations += 1
            delta = 0
            for state in self.max_reward:
                curr_reward = copy.deepcopy(self.max_reward)[state]
                reward_swing = 0.0
                reward_stand = 0.0
                for outcome in self.ptrain_swing[state]:
                    if outcome in self.max_reward:
                        reward_swing += (self.ptrain_swing[state][outcome]*(self.rewards[outcome]+self.max_reward[outcome]))
                    else:
                        reward_swing += (self.ptrain_swing[state][outcome] * self.rewards[outcome])
                for outcome in self.ptrain_stand[state]:
                    if outcome in self.max_reward:
                        reward_stand += (self.ptrain_stand[state][outcome]*(self.rewards[outcome]+self.max_reward[outcome]))
                    else:
                        reward_stand += (self.ptrain_stand[state][outcome] * self.rewards[outcome])
                if reward_swing > reward_stand:
                    self.max_reward[state] = reward_swing
                    self.policy[state] = 'swing'
                elif reward_swing < reward_stand:
                    self.max_reward[state] = reward_stand
                    self.policy[state] = 'stand'
                else:
                    actions = ['stand', 'swing']
                    action = random.choice(actions)
                    if action == 'swing':
                        self.max_reward[state] = reward_swing
                        self.policy[state] = 'swing'
                    else:
                        self.max_reward[state] = reward_stand
                        self.policy[state] = 'stand'
                diff = abs(curr_reward-self.max_reward[state])
                delta = max([delta, diff])
        return iterations

    def evaluatePolicy(self, epsilon):
        delta = 100
        iterations = 0
        while delta >= epsilon:
            iterations += 1
            delta = 0
            for state in self.expected_reward:
                curr_reward = copy.deepcopy(self.expected_reward)[state]
                reward = 0.0
                action = self.policy[state]
                if action == 'swing':
                    for outcome in self.ptest_swing[state]:
                        if outcome in self.expected_reward:
                            reward += (self.ptest_swing[state][outcome] * (self.rewards[outcome] + self.expected_reward[outcome]))
                        else:
                            reward += (self.ptest_swing[state][outcome] * self.rewards[outcome])
                elif action == 'stand':
                    for outcome in self.ptest_stand[state]:
                        if outcome in self.expected_reward:
                            reward += (self.ptest_stand[state][outcome] * (self.rewards[outcome] + self.expected_reward[outcome]))
                        else:
                            reward += (self.ptest_stand[state][outcome] * self.rewards[outcome])
                self.expected_reward[state] = reward
                diff = abs(curr_reward - self.expected_reward[state])
                delta = max([delta, diff])
        return iterations

