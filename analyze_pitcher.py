from pitcher import Pitcher
import numpy as np


def print_policy(policy, rewards):
    print("(Balls, Strikes) => (Action, Reward)")
    print("====================================")
    for state in policy:
        line = "(" + state[0] + ", " + state[1] + ") => (" + policy[state] + ", " + str(rewards[state]) + ")"
        print(line)

user_pitcher = ''
valid_pitchers = {
    'kershaw': {'name': 'Clayton Kershaw', '2017era': '2.31', '2018era': '2.73', 'file': 'clayton_kershaw.csv'},
    'foltynewicz': {'name': 'Mike Foltynewicz', '2017era': '4.79', '2018era': '2.85', 'file': 'mike_foltynewicz.csv'},
    'scherzer': {'name': 'Max Scherzer', '2017era': '2.51', '2018era': '2.53', 'file': 'max_scherzer.csv'}
}

while user_pitcher != 'exit':
    valid_entry = False
    while not valid_entry:
        print("Pitcher Options")
        print("===============")
        print("Clayton Kershaw => kershaw")
        print("Mike Foltynewicz => foltynewicz")
        print("Max Scherzer => scherzer")
        print("Exit program => exit")
        user_pitcher = input("Please select a pitcher: ")
        print()
        if user_pitcher not in valid_pitchers and user_pitcher != 'exit':
            print("Invalid input. Please try again.")
            print()
        else:
            valid_entry = True

    if user_pitcher != 'exit':
        # Initialize pitcher: takes in data file, training season, testing season
        curr_pitcher = Pitcher(("PlayerData/"+valid_pitchers[user_pitcher]['file']), 2017, 2018)

        epsilon = np.finfo(float).eps

        # Display pitcher stats
        print("Pitcher: " + valid_pitchers[user_pitcher]['name'])
        print("2017 ERA: " + valid_pitchers[user_pitcher]['2017era'] + ", 2018 ERA: " + valid_pitchers[user_pitcher][
            '2018era'])
        print()

        # Determine optimal policy by using value iteration over the training season
        step_1_its = curr_pitcher.solveActions(epsilon)
        print("Training on 2017 Data - Iterations: " + str(step_1_its))
        print_policy(curr_pitcher.policy, curr_pitcher.max_reward)
        print()

        # Evaluate the performance of the optimal policy on the testing season by using policy evaluation
        step_2_its = curr_pitcher.evaluatePolicy(epsilon)
        print("Evaluating on 2018 Data - Iterations: " + str(step_2_its))
        print_policy(curr_pitcher.policy, curr_pitcher.expected_reward)
        print()

        # Determine how accurate using 2017's optimal policy on 2018's data actually is
        score = 0.0
        for state in curr_pitcher.max_reward:
            score += abs(curr_pitcher.max_reward[state] - curr_pitcher.expected_reward[state])
        print("Total deviation score: "+str(score))

        # Determine the performance of using 2017's optimal policy on 2018's data
        score = 0.0
        for state in curr_pitcher.max_reward:
            score += (curr_pitcher.expected_reward[state]-curr_pitcher.max_reward[state])
        print("Total performance score: "+str(score))
        print()
        print()

