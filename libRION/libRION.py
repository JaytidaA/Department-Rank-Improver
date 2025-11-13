# RION Optimisation

from typing import Any, Callable, List, Tuple
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from libRION.libPSO import PSO

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

    * pso_iterations:
        the number of iterations for PSO mutation (default 10)

    ---
    """

    def __init__(self, generations: int = 10, pop_size: int = 100, inertia: float = 1.0, 
                 self_confidence: float = 1.0, cross_confidence: float = 1.0, pso_iterations: int = 10):
        self.__gens           = generations
        self.__pop_size       = pop_size
        self.__pso_inertia    = inertia
        self.__pso_self_conf  = self_confidence
        self.__pso_cross_conf = cross_confidence
        self.__pso_iterations = pso_iterations

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
        return population

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
                    next_generation = pd.concat([next_generation, combined.iloc[[fronts[front_idx][j]]]], ignore_index=True)
                front_idx += 1

            fronts[front_idx] = self._crowding(combined, fronts[front_idx])
            remaining_spots = self.__pop_size - len(next_generation)
            for j in range(remaining_spots):
                next_generation = pd.concat([next_generation, combined.iloc[[fronts[front_idx][j]]]], ignore_index=True)

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

        return self.data.sample(n = psize, replace = True).reset_index(drop=True)

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
        offspring = population.iloc[0:0].copy()

        # threshold for crossover
        cross_threshold = 0.7

        for i in range(osize // 2):
            if ranks is not None:
                # select 4 random individuals from the population
                possible = random.sample(range(population.shape[0]), 4)
                parents = []

                # select the better one from each pair                
                parents.append(possible[0 if ranks[possible[0]] < ranks[possible[1]] else 1])
                parents.append(possible[2 if ranks[possible[2]] < ranks[possible[3]] else 3])

                children = population.iloc[parents].copy()
                children = children.reset_index(drop=True)

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
                children = population.sample(n = 2, replace = False).copy()
                children = children.reset_index(drop=True)

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

        # PSO Mutation: Optimize a single random attribute
        offspring = self._pso_mutation(offspring)

        return offspring

    def _pso_mutation(self, offspring: pd.DataFrame) -> pd.DataFrame:
        """Apply PSO mutation to a single random attribute of the offspring population

        ---
        Parameters
        ----------
        * offspring:
            The offspring population to mutate

        ---
        Returns
        -------
        Mutated offspring with one attribute optimized using PSO

        ---
        """
        if offspring.shape[0] == 0:
            return offspring

        # Select a random attribute/column to optimize
        random_col = random.choice(offspring.columns.tolist())
        col_index = offspring.columns.get_loc(random_col)

        # Extract the column data as a 2D numpy array for PSO
        positions = offspring[[random_col]].values

        # Define objective function for PSO
        # This function evaluates the sum of all optimizer functions for an individual
        def pso_objective(x):
            # Create a temporary individual by replacing the selected attribute
            temp_individual = offspring.iloc[0].copy()
            temp_individual.iloc[col_index] = x[0]
            
            # Calculate aggregate fitness (sum of all objective functions)
            total_fitness = sum(func(temp_individual) for func in self.optimiser_funcs)
            
            # Add constraint penalty
            constraint_penalty = self._constraint_violate(temp_individual) * 1000
            
            return total_fitness + constraint_penalty

        # Initialize and run PSO
        pso = PSO(
            obj_func=pso_objective,
            data=positions,
            num_particles=offspring.shape[0],
            max_iter=self.__pso_iterations,
            inertia_weight=self.__pso_inertia,
            cognitive_coeff=self.__pso_self_conf,
            social_coeff=self.__pso_cross_conf
        )

        # Run PSO optimization (suppress output)
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        best_position, best_score = pso.optimize()
        
        sys.stdout = old_stdout

        # Update offspring with optimized positions
        offspring[random_col] = pso.positions[:, 0]

        return offspring

    def _fast_non_dominated_sort(self, population: pd.DataFrame) -> Tuple[List[List[int]], List[int]]:
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
                elif self._dominates(population.iloc[j], population.iloc[i]):
                    dominates[j].append(i)
            
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

    def _dominates(self, p: pd.Series, q: pd.Series) -> bool:
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

    def _constraint_violate(self, p: pd.Series) -> float:
        """returns a value representing the constraint violation of the individual p

        ---
        Parameters
        ----------
        * p:
            The individual which is to be checked for constraint violation

        Returns
        -------
        A value representing the constraint violation
        ---

        """
        if not self.constraints:
            return 0.0
            
        total = 0.0
        for i, constraint in enumerate(self.constraints):
            min_val, max_val = constraint[0], constraint[1]
            value = p.iloc[i]
            
            # Calculate normalized constraint violation
            if max_val != 0:
                c1 = max(0, (value - max_val) / abs(max_val))
            else:
                c1 = max(0, value - max_val)
                
            if min_val != 0:
                c2 = max(0, (min_val - value) / abs(min_val))
            else:
                c2 = max(0, min_val - value)
            
            total += c1 + c2

        return total / len(self.constraints) if self.constraints else 0.0
