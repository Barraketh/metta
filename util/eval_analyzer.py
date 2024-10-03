import json
from tabulate import tabulate
from termcolor import colored
from util.stats_library import *

#================================================================================
# LOOKUP DICTIONARIES
#================================================================================
# Function lookup dictionary
function_lookup = {
    '1v1': 'mann_whitney_u_test',
    'elo_1v1': 'elo_test',
    'glicko2_1v1': 'glicko2_test',
    'multiplayer': 'kruskal_wallis_test',
}
# Stat category lookup dictionary. This approach deals with the situation where an episode doesn't have a stat, which happens if none of the agents have a finite score in the category.
stat_category_lookup = {
    'altar': ['action.use.energy.altar'],
    'all': [
        "action.rotate.energy",
        "action.attack",
        "action.attack.energy",
        "action.move.energy",
        "action.gift.energy",
        "r3.stolen",
        "action.rotate",
        "action.attack.altar",
        "action.use.altar",
        "r1.stolen",
        "action.attack.agent",
        "shield_damage",
        "damage.altar",
        "status.shield.ticks",
        "action.use.energy.altar",
        "action.attack.wall",
        "destroyed.wall",
        "action.shield.energy",
        "r2.gained",
        "r3.gained",
        "action.move",
        "action.use.energy",
        "r2.stolen",
        "status.frozen.ticks",
        "shield_upkeep",
        "attack.frozen",
        "r1.gained",
        "action.use",
        "damage.wall",
        "policy_name"
    ],
    'adversarial': [
        "action.attack",
        "action.attack.energy",
        "action.attack.altar",
        "action.attack.agent",
        "action.attack.wall",
        "damage.altar",
        "damage.wall",
        "shield_damage",
        "attack.frozen",
        "r1.stolen",
        "r2.stolen",
        "r3.stolen",
        "destroyed.wall"
    ],
    'shield': [
        "shield_damage",
        "status.shield.ticks",
        "action.shield.energy",
        "shield_upkeep"
    ],
}
#================================================================================
#END LOOKUP DICTIONARIES
#================================================================================

def print_policy_stats(data, eval_method, stat_category):
    """
    Process game statistics data and perform statistical tests based on the evaluation method and stat category.

    Parameters:
    data (list): The game data loaded from JSON.
    eval_method (str): The evaluation method to use ('1v1' or 'multiplayer').
    stat_category (str): The category of statistics to analyze.

    Output:
    Prints the statistical test results.
    """

    # Get the function and stats list based on the evaluation method and stat category
    test_func_name = function_lookup.get(eval_method)
    if test_func_name is None:
        raise ValueError(f"Unknown method: {eval_method}")
    test_func = globals()[test_func_name]

    categories_list = stat_category_lookup.get(stat_category)
    if categories_list is None:
        raise ValueError(f"Unknown stat category: {stat_category}")

    #---start extracting data---
    # Extract all policy names from the data
    policy_names = []
    for episode in data:
        for agent in episode:
            policy_name = agent.get('policy_name', None)
            if policy_name and policy_name not in policy_names:
                policy_names.append(policy_name)

    # Initialize stats dictionaries for each stat and policy
    stats = {}
    for stat_name in categories_list:
        # Create a dictionary for each stat with a dictionary of each policy,
        # the values of which are lists of None as long as the number of episodes
        stats[stat_name] = {policy_name: [None] * len(data) for policy_name in policy_names}

    # Extract stats per policy per episode
    for idx, episode in enumerate(data):
        for agent in episode:
            policy = agent.get('policy_name', None)
            if policy is None:
                continue
            # Loop through each stat and add to this policy's scores for the episode
            for stat_name in categories_list:
                stat_value = agent.get(stat_name, None)
                # Sum the stat values for the policy at the current episode index
                if stat_value is not None:
                    if stats[stat_name][policy][idx] is not None:
                        stats[stat_name][policy][idx] += stat_value
                    else:
                        stats[stat_name][policy][idx] = stat_value
                # Else, leave as None
    #---end extracting data---

    # Run statistical analysis and print results
    test_func(stats, policy_names, categories_list)


# If you're running on Windows and need ANSI color support
# import colorama
# colorama.init()
