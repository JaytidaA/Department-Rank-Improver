# RION Optimisation

from typing import Any, Callable, List
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RionOptimiser:
    """This is the main class for the RionOptimiser

    ---

    Parameters
    ----------
    * generations:
        the total number of generations to run the genetic algorithm for (default 10)
    
    * pop_size:
        the population size for the optimisation (default 100)

    * inertia:
        the inertia of the particles in the particle swarm optimisation (default 1.0)
    
    * self_confidence:
        the scaling factor of going towards the personal best during PSO (default 1.0)

    * cross_confidence:
        the scaling factor of going towards the global best during PSO (default 1.0)

    ---
    """

    def __init__(self, generations: int = 10, pop_size: int = 100, inertia: float = 1.0, self_confidence: float = 1.0, cross_confidence: float = 1.0):
        self.__gens           = generations
        self.__pop_size       = pop_size
        self.__pso_inertia    = inertia
        self.__pso_self_conf  = self_confidence
        self.__pso_cross_conf = cross_confidence

    def optimise(
        self, df: pd.DataFrame, optimiser_funcs: List[Callable[[tuple[Any, ...]], float]],
        constraints: List[List[int]]
    ) -> None:
        """Initialise values and call modified NSGA-2 optimisation

        ---
        Parameters
        ----------
        * df:
            the dataset containing the initial individuals to be modified

        * optimiser_funcs:
            a list of functions that take in an individual (row) and return a "fitness" for that individual

        * constraints:
            constraints on the attributes of the dataset passed in, expected form {`min_i`, `max_i`}

        ---
        Returns
        -------
        None

        ---
        """
        self.data            = df
        self.optimiser_funcs = optimiser_funcs
        self.constraints     = constraints

        # perform the optimisation
        population = self._modified_nsga_ii()

    def _modified_nsga_ii(self) -> pd.DataFrame:
        """Modified non-dominated sorting genetic algorithm 2

        ---
        Parameters
        ----------
        * None

        ---
        Returns
        -------
        Population after performing the required optimisations

        ---
        """
        population = self._generate_population(self.__pop_size)
        fronts, ranks = None, None

        for i in range(self.__gens):
            if i == 0:
                offspring = self._generate_offspring(population, self.__pop_size)
            else:
                offspring = self._generate_offspring(population, self.__pop_size, ranks)

            # combine the population and offspring
            combined = pd.concat([population, offspring], ignore_index = True)

            # get the pareto fronts and a list of each elements pareto index
            fronts, ranks = self._fast_non_dominated_sort(combined)

            # empty dataset with the same columns and attributes as population
            next_generation = population.iloc[0:0]

            front_idx = 0
            # while the total length does not exceed population size
            while len(next_generation) + len(fronts[front_idx]) <= self.__pop_size:
                for j in range(len(fronts[front_idx])):
                    next_generation.iloc[len(next_generation)] = \
                        combined.iloc[fronts[front_idx][j]]
                front_idx += 1

            fronts[front_idx] = self._crowding(combined, fronts[front_idx])
            remaining_spots = self.__pop_size - len(next_generation)
            for j in range(remaining_spots):
                next_generation.iloc[len(next_generation)] = combined.iloc[fronts[front_idx][j]]

            # new population is the next generation
            population = next_generation

        return population
            

    def _generate_population(self, psize: int) -> pd.DataFrame:
        """generate a random population of size 'psize' from the given dataset

        ---
        Parameters
        ----------
        * psize:
            number of individuals to sample from the dataset

        ---
        Returns
        -------
        Population of size 'psize' sampled randomly with replacement

        ---
        Raises
        ------
        * RuntimeError: If the dataset has not been initialised beforehand

        ---
        """
        if self.data is None:
            raise RuntimeError("generating population without data!")

        return self.data.sample(n = psize, replace = True)

    def _generate_offspring(self, population: pd.DataFrame, osize: int, ranks: List[int] = None) -> pd.DataFrame:
        """generate 'psize' offspring from the given population using tournament selection

        ---
        Parameters
        ----------
        * population:
            parent population to generate children from

        * osize:
            number of offspring to generate (usually same as len(population))

        * ranks:
            pareto ranks of each element in the population (default None)

        ---
        Returns
        -------
        Offspring of size 'osize' generated after performing crossover and PSO mutation

        ---
        """

        # empty dataset with the same columns and attributes as population
        offspring = population.iloc[0:0]

        # threshold for crossover
        cross_threshold = 0.7

        for i in range(osize / 2):
            if rank is not None:
                # select 4 random individuals from the population
                possible = random.sample(range(population.shape[0]), 4)
                parents = []

                # select the better one from each pair                
                parents.append(possible[0 if ranks[possible[0]] > ranks[possible[1]] else 1])
                parents.append(possible[2 if ranks[possible[2]] > ranks[possible[3]] else 3])

                children = population.iloc[parents]

                # crossover
                for col in children.columns:
                    if random.random() > cross_threshold:
                        # swap values between the two rows
                        children.loc[0, col], children.loc[1, col] = (
                            children.loc[1, col],
                            children.loc[0, col],
                        )

                # add to offspring
                offspring = pd.concat([offspring, children], ignore_index = True)
            else:
                children = population.sample(n = 2, replace = False)

                # crossover
                for col in chilren.columns:
                    if random.random() > cross_threshold:
                        # swap values between the two rows
                        children.loc[0, col], children.loc[1, col] = (
                            children.loc[1, col],
                            children.loc[0, col],
                        )

                # add to offspring
                offspring = pd.concat([offspring, children], ignore_index = True)

        # TODO: Integrate PSO library for mutation

        return offspring

    def _fast_non_dominated_sort(self, population: pd.DataFrame) -> List[int]:
        """sort the elements in the population into pareto frontiers

        ---
        Parameters
        ----------
        * population:
            individual population to optimise

        ---
        Returns
        -------
        Pareto fronts containing the indices of the elements in the population dataframe

        ---
        """

        # number of individual a specific ind. is dominated by
        domcount = [0 for _ in range(population.shape[0])]

        # individuals which a specific ind. dominates
        dominates = [[] for _ in range(population.shape[0])]

        # bookeeping of frontier for each individual
        rank = [0 for _ in range(population.shape[0])]

        fronts = [[]]

        for i in range(population.shape[0]):
            for j in range(population.shape[0]):
                if self._dominates(population.iloc[i], population.iloc[j]):
                    # if (population[i] > population[j])
                    dominates[i].append(j)
                    domcount[j] += 1
                elif self._dominates(population.iloc[j], population.iloc[i]):
                    dominates[j].append(i)
                    domcount[i] += 1
            
            if domcount[i] == 0:
                rank[i] = 0
                fronts[0].append(i)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominates[p]:
                    domcount[q] -= 1
                    if domcount[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        fronts.pop()
        return fronts, rank

    def _dominates(self, p: Tuple[Any, ...], q: Tuple[Any, ...]) -> bool:
        """returns true if individual p dominates individual q (it is assumed that each function should be minimised)

        ---
        Parameters
        ----------
        * p, q:
            Elements to be compared

        Returns
        -------
        Boolean representing if p dominates q or not

        ---
        """

        and_part = True
        or_part = False

        cv1 = self._constraint_violate(p)
        cv2 = self._constraint_violate(q)

        if cv1 < cv2:
            return True
        elif cv1 > cv2:
            return False
        else:
            for i in range(len(self.optimiser_funcs)):
                f = self.optimiser_funcs[i]
                # strictly better
                and_part = (and_part and (f(p) < f(q)))
                if and_part == False:
                    return False
                # better or equal
                or_part = (or_part or (f(p) <= f(q)))
            
            return or_part and and_part

    def _crowding(self, population: pd.DataFrame, front: List[int]) -> List[int]:
        """sorts the elements in the front by crowding distance and returns a new list
        
        ---
        Parameters
        ----------
        * population:
            Dataset containing the population of individuals

        * front:
            The actual front containing the indices referring to the individual in population which are part of that front

        Returns
        -------
        The input `front` sorted according to the crowding distance values

        ---
        """

        distances = {idx: 0.0 for idx in front}
        num_objectives = len(self.optimiser_funcs)

        front_list = list(front)

        for m in range(num_objectives):
            # Compute objective values for each individual in the front
            values = {idx: self.optimiser_funcs[m](population.loc[idx]) for idx in front_list}

            # Sort the front based on the m-th objective
            sorted_front = sorted(front_list, key=lambda idx: values[idx])

            # Find min and max values for normalization
            f_min = values[sorted_front[0]]
            f_max = values[sorted_front[-1]]

            # Assign infinite distance to boundary solutions
            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')

            # Compute crowding distance for each middle solution
            if f_max > f_min:  # avoid division by zero
                for i in range(1, len(sorted_front) - 1):
                    prev_val = values[sorted_front[i - 1]]
                    next_val = values[sorted_front[i + 1]]
                    distances[sorted_front[i]] += (next_val - prev_val) / (f_max - f_min)

        # Sort by descending crowding distance (larger distance = higher priority)
        sorted_by_distance = sorted(front_list, key=lambda idx: distances[idx], reverse=True)
        return sorted_by_distance

    def _constraint_violate(self, p: Tuple[Any, ...]) -> float:
        """returns a value representing the constraint violation of the individual p

        ---
        Parameters
        ----------
        * p:
            The individual which is to be checked for constraint violation

        Returns
        -------
        A value representing the 
        ---

        """
        sum = 0.0
        for constraint in self.constraints:
            c = max(0, (p[i] - max) / max, (min - p[i]) / min)
            sum += c

        return (sum / len(self.constraints))


    def _particle_swarm(self):
        pass




























# # Constraint violation function
# def constraint_violation(team, budget):
#     total_salary = sum(player['wage_eur'] for player in team)
#     return max(0, (total_salary - budget) / budget)

# def dominates(ind1, ind2, budget):
#     """Return True if ind1 dominates ind2 under constraints."""
#     cv1 = constraint_violation(ind1, budget)
#     cv2 = constraint_violation(ind2, budget)

#     if cv1 < cv2:
#         return True
#     elif cv1 > cv2:
#         return False
#     else:
#         # Compare objectives
#         f1 = [
#             phi_overall(ind1),
#             lambda_attack(ind1),
#             lambda_defence(ind1),
#             lambda_goalkeeper(ind1),
#             potential(ind1),
#         ]
#         f2 = [
#             phi_overall(ind2),
#             lambda_attack(ind2),
#             lambda_defence(ind2),
#             lambda_goalkeeper(ind2),
#             potential(ind2),
#         ]
#         better_or_equal = all(a >= b for a, b in zip(f1, f2))
#         strictly_better = any(a > b for a, b in zip(f1, f2))
#         return better_or_equal and strictly_better

# def non_dominated_sort(population, budget):
#     S = [[] for _ in range(len(population))]
#     n = [0 for _ in range(len(population))]
#     rank = [0 for _ in range(len(population))]

#     fronts = [[]]

#     for p in range(len(population)):
#         for q in range(len(population)):
#             if dominates(population[p], population[q], budget):
#                 S[p].append(q)
#             elif dominates(population[q], population[p], budget):
#                 n[p] += 1
#         if n[p] == 0:
#             rank[p] = 0
#             fronts[0].append(p)

#     i = 0
#     while fronts[i]:
#         next_front = []
#         for p in fronts[i]:
#             for q in S[p]:
#                 n[q] -= 1
#                 if n[q] == 0:
#                     rank[q] = i + 1
#                     next_front.append(q)
#         i += 1
#         fronts.append(next_front)
#     fronts.pop()
#     return fronts

# def crowding(front, num_obj, budget):
#     objectives = []
#     idx = 0
#     distance = [0 for _ in range(len(front))]
#     for team in front:
#         objectives.append([
#             idx,
#             phi_overall(team),
#             lambda_attack(team),
#             lambda_defence(team),
#             lambda_goalkeeper(team),
#             potential(team)
#         ])
#         idx += 1

#     for i in range(num_obj):
#         sorted_obj = sorted(objectives, key=lambda x: x[i + 1])
#         max_t = sorted_obj[-1][0]
#         min_t = sorted_obj[0][0]
#         distance[min_t] = float('inf')
#         distance[max_t] = float('inf')
#         for j in range(1, len(front) - 1):
#             prev_val = sorted_obj[j - 1][i + 1]
#             next_val = sorted_obj[j + 1][i + 1]
#             max_val = sorted_obj[-1][i + 1]
#             min_val = sorted_obj[0][i + 1]
#             if max_val != min_val:
#                 distance[sorted_obj[j][0]] += (next_val - prev_val) / (max_val - min_val)

#     return distance

# def crowded_comparison(ind1, ind2, rank, distance):
#     if rank[ind1] < rank[ind2]:
#         return True
#     elif rank[ind1] == rank[ind2]:
#         return distance[ind1] > distance[ind2]
#     return False

# def make_offspring(population, df, team_size):
#     """Simple random re-initialization for offspring (placeholder)."""
#     offspring = []
#     for _ in range(len(population)):
#         team = random.sample(list(df.to_dict(orient="records")), team_size)
#         offspring.append(team)
#     return offspring

# def nsga_ii(df, budget, population_size=50, generations=10, team_size=11):
#     # Initialize random population
#     population = [
#         random.sample(list(df.to_dict(orient="records")), team_size)
#         for _ in range(population_size)
#     ]

#     for gen in range(int(generations)):
#         offspring = make_offspring(population, df, team_size)
#         combined = population + offspring

#         fronts = non_dominated_sort(combined, budget)

#         new_population = []
#         rank = {}
#         distance = {}

#         for i, front_indices in enumerate(fronts):
#             front = [combined[idx] for idx in front_indices]
#             dists = crowding(front, 5, budget)

#             for j, idx in enumerate(front_indices):
#                 rank[idx] = i
#                 distance[idx] = dists[j]

#             if len(new_population) + len(front) <= population_size:
#                 new_population.extend(front)
#             else:
#                 sorted_front = sorted(
#                     front_indices,
#                     key=lambda x: (rank[x], -distance[x])
#                 )
#                 needed = population_size - len(new_population)
#                 new_population.extend([combined[idx] for idx in sorted_front[:needed]])
#                 break

#         population = new_population

#         # Plot frontier 0 for visualization
#         if fronts[0]:
#             f0 = [combined[idx] for idx in fronts[0]]
#             x = [potential(team) for team in f0]
#             y = [phi_overall(team) for team in f0]
#             color_spots = "red" if (gen == int(generations) - 1) else "blue"
#             plt.scatter(x, y, color = color_spots)

#     plt.xlabel("Potential")
#     plt.ylabel("Overall")
#     plt.legend()
#     plt.show()

#     return population

# def main():
#     df = pd.read_csv('./FTP/filteredplayer20.csv')
#     num_generations = int(input("Enter the number of generations: "))
#     population_size = int(input("Enter the population size: "))
#     budget = float(input("Enter the total budget in euros (in millions): "))
#     if (num_generations <= 0) or (population_size <= 0) or (budget <= 0):
#         print("Invalid input parameters!")
#         return
#     nsga_ii(df, budget * 1000000, population_size, num_generations)

# if __name__ == "__main__":
#     main()
