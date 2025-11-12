# RION Optimisation

import pandas as pd
import random
import matplotlib.pyplot as plt

class RionOptimiser:
    """
    This is the main class for the RionOptimiser
    """
    def __init__(self, funcs, generations = 10, pop_size = 100, self_confidence = 1, cross_confidence = 1):
        self.__objectives = funcs
        self.__gens = generations
        self.__population = pop_size
        self.__pso_self_conf = self_confidence
        self.__pso_cross_conf = cross_confidence

    def optimise(self, df, optimiser_attrs, constraints):
        pass

    def _nsga_ii(self):
        pass

    def _crowding(self):
        pass

    def _constraint_violate(self):
        pass

    def _particle_swarm(self):
        pass




























# Constraint violation function
def constraint_violation(team, budget):
    total_salary = sum(player['wage_eur'] for player in team)
    return max(0, (total_salary - budget) / budget)

def dominates(ind1, ind2, budget):
    """Return True if ind1 dominates ind2 under constraints."""
    cv1 = constraint_violation(ind1, budget)
    cv2 = constraint_violation(ind2, budget)

    if cv1 < cv2:
        return True
    elif cv1 > cv2:
        return False
    else:
        # Compare objectives
        f1 = [
            phi_overall(ind1),
            lambda_attack(ind1),
            lambda_defence(ind1),
            lambda_goalkeeper(ind1),
            potential(ind1),
        ]
        f2 = [
            phi_overall(ind2),
            lambda_attack(ind2),
            lambda_defence(ind2),
            lambda_goalkeeper(ind2),
            potential(ind2),
        ]
        better_or_equal = all(a >= b for a, b in zip(f1, f2))
        strictly_better = any(a > b for a, b in zip(f1, f2))
        return better_or_equal and strictly_better

def non_dominated_sort(population, budget):
    S = [[] for _ in range(len(population))]
    n = [0 for _ in range(len(population))]
    rank = [0 for _ in range(len(population))]

    fronts = [[]]

    for p in range(len(population)):
        for q in range(len(population)):
            if dominates(population[p], population[q], budget):
                S[p].append(q)
            elif dominates(population[q], population[p], budget):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts

def crowding(front, num_obj, budget):
    objectives = []
    idx = 0
    distance = [0 for _ in range(len(front))]
    for team in front:
        objectives.append([
            idx,
            phi_overall(team),
            lambda_attack(team),
            lambda_defence(team),
            lambda_goalkeeper(team),
            potential(team)
        ])
        idx += 1

    for i in range(num_obj):
        sorted_obj = sorted(objectives, key=lambda x: x[i + 1])
        max_t = sorted_obj[-1][0]
        min_t = sorted_obj[0][0]
        distance[min_t] = float('inf')
        distance[max_t] = float('inf')
        for j in range(1, len(front) - 1):
            prev_val = sorted_obj[j - 1][i + 1]
            next_val = sorted_obj[j + 1][i + 1]
            max_val = sorted_obj[-1][i + 1]
            min_val = sorted_obj[0][i + 1]
            if max_val != min_val:
                distance[sorted_obj[j][0]] += (next_val - prev_val) / (max_val - min_val)

    return distance

def crowded_comparison(ind1, ind2, rank, distance):
    if rank[ind1] < rank[ind2]:
        return True
    elif rank[ind1] == rank[ind2]:
        return distance[ind1] > distance[ind2]
    return False

def make_offspring(population, df, team_size):
    """Simple random re-initialization for offspring (placeholder)."""
    offspring = []
    for _ in range(len(population)):
        team = random.sample(list(df.to_dict(orient="records")), team_size)
        offspring.append(team)
    return offspring

def nsga_ii(df, budget, population_size=50, generations=10, team_size=11):
    # Initialize random population
    population = [
        random.sample(list(df.to_dict(orient="records")), team_size)
        for _ in range(population_size)
    ]

    for gen in range(int(generations)):
        offspring = make_offspring(population, df, team_size)
        combined = population + offspring

        fronts = non_dominated_sort(combined, budget)

        new_population = []
        rank = {}
        distance = {}

        for i, front_indices in enumerate(fronts):
            front = [combined[idx] for idx in front_indices]
            dists = crowding(front, 5, budget)

            for j, idx in enumerate(front_indices):
                rank[idx] = i
                distance[idx] = dists[j]

            if len(new_population) + len(front) <= population_size:
                new_population.extend(front)
            else:
                sorted_front = sorted(
                    front_indices,
                    key=lambda x: (rank[x], -distance[x])
                )
                needed = population_size - len(new_population)
                new_population.extend([combined[idx] for idx in sorted_front[:needed]])
                break

        population = new_population

        # Plot frontier 0 for visualization
        if fronts[0]:
            f0 = [combined[idx] for idx in fronts[0]]
            x = [potential(team) for team in f0]
            y = [phi_overall(team) for team in f0]
            color_spots = "red" if (gen == int(generations) - 1) else "blue"
            plt.scatter(x, y, color = color_spots)

    plt.xlabel("Potential")
    plt.ylabel("Overall")
    plt.legend()
    plt.show()

    return population

def main():
    df = pd.read_csv('./FTP/filteredplayer20.csv')
    num_generations = int(input("Enter the number of generations: "))
    population_size = int(input("Enter the population size: "))
    budget = float(input("Enter the total budget in euros (in millions): "))
    if (num_generations <= 0) or (population_size <= 0) or (budget <= 0):
        print("Invalid input parameters!")
        return
    nsga_ii(df, budget * 1000000, population_size, num_generations)

if __name__ == "__main__":
    main()
