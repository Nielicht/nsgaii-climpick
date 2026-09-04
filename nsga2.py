import numpy as np
import xarray as xr
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from solution import Solution

class Nsga2:

    def __init__(self, population: list[Solution] = None, generator_population_size: int = 100, generations: int = 100, seed: int = None, max_flips: int = 10, mutation_prob: float = 0.5, max_active_vars: int = 50, cross_type: int = 1):
        self.population = population
        self.generator_population_size = generator_population_size
        self.generations = generations
        self.max_flips = max_flips
        self.mutation_prob = mutation_prob
        self.max_active_vars = max_active_vars
        self.cross_type = cross_type
        self.rng = np.random.default_rng(seed)
        self.pareto_history = []

        if population is None:
            self.generation_count = -1
        else:
            self.generation_count = 0

        predictoras_tmp = xr.open_dataset('data/preprocessed/predictoras.nc')
        z500 = predictoras_tmp['z500'].values
        rh = predictoras_tmp['rh'].values
        predictoras = np.stack([z500, rh], axis=-1).reshape(z500.shape[0], -1)
        del predictoras_tmp, z500, rh
        etiquetas = pd.read_parquet('data/preprocessed/etiquetas.parquet')

        no_test_mask = etiquetas['time'].dt.year <= 2011
        predictoras = predictoras[no_test_mask.values]
        etiquetas = etiquetas[no_test_mask]
        
        train_mask = etiquetas['time'].dt.year <= 2009
        val_mask = etiquetas['time'].dt.year > 2009
        self.X_train = predictoras[train_mask.values]
        self.X_val = predictoras[val_mask.values]
        self.y_train = etiquetas[train_mask].reset_index(drop=True)
        self.y_val = etiquetas[val_mask].reset_index(drop=True)

    def run(self):
        if self.population is None:
            self.population = self.generate_population()
            for solution in self.population:
                if solution.fitness is None:
                    solution.fitness = self.get_fitness(solution)
            self.pareto(self.population, trim_count=len(self.population))
        
        for i in range(self.generations):
            self.iterate()

    def generate_population(self, population_size: int = None):
        if population_size is None:
            population_size = self.generator_population_size
        
        population = []

        for i in range(population_size):
            num_vars = self.rng.integers(1, self.max_active_vars + 1)
            gnome = np.zeros(self.X_train.shape[1], dtype=int)
            vars_seleccionadas = self.rng.choice(self.X_train.shape[1], size=num_vars, replace=False)
            gnome[vars_seleccionadas] = 1
            population.append(Solution(gnome))

        return population

    def limit_vars(self, gnome: np.ndarray):
        final_gnome = gnome.copy()
        num_vars = gnome.sum()
        
        if num_vars > self.max_active_vars:
            diff = gnome.sum() - self.max_active_vars
            possible_vars = np.where(gnome == 1)[0]
            vars_to_remove = self.rng.choice(possible_vars, size=diff, replace=False)
            final_gnome[vars_to_remove] = 0
        elif num_vars < 1:
            var_to_add = self.rng.choice(self.X_train.shape[1], size=1, replace=False)
            final_gnome[var_to_add] = 1

        return final_gnome
        
    def iterate(self):
        combined_population = self.combinePopulation()
        for solution in combined_population:
            if solution.fitness is None:
                solution.fitness = self.get_fitness(solution)
        self.pareto(combined_population)

    def get_fitness(self, solution: Solution):
        random_state=96
        
        X_train = self.X_train[:, solution.gnome == 1]
        X_val = self.X_val[:, solution.gnome == 1]
        y_hw_train = self.y_train['hw']
        y_hw_val = self.y_val['hw']
        y_dr_train = self.y_train['dr']
        y_dr_val = self.y_val['dr']

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)    

        modelo_hw = LogisticRegression(max_iter=5000, random_state=random_state)
        modelo_hw.fit(X_train, y_hw_train)
        
        modelo_dr = LogisticRegression(max_iter=5000, random_state=random_state)
        modelo_dr.fit(X_train, y_dr_train)

        y_hw_pred = modelo_hw.predict(X_val)
        y_dr_pred = modelo_dr.predict(X_val)

        f1_hw = f1_score(y_hw_val, y_hw_pred)
        f1_dr = f1_score(y_dr_val, y_dr_pred)

        return (f1_dr, f1_hw, X_train.shape[1])
        

    def pareto(self, population: list[Solution], trim_count: int = None):
        if trim_count is None:
            trim_count = len(population) // 2
        
        for solution in population:
            for against_solution in population:
                if solution is against_solution: 
                    continue
                elif solution.dominates(against_solution):
                    solution.dominated_solutions.append(against_solution)
                    against_solution.dominated_by += 1
        
        rank = 0
        aux_population: list[Solution] = population.copy()
        solutions_by_rank_list: list[Solution] = []
        while len(aux_population) > len(population) - trim_count:
            solutions_to_touch_list = []
            solutions_in_current_rank_list = []
            for solution in aux_population:
                if solution.dominated_by == 0:
                    solutions_to_touch_list.append(solution)
            for solution in solutions_to_touch_list:
                for dominated_solution in solution.dominated_solutions:
                    dominated_solution.dominated_by -= 1
                solution.dominated_solutions = []
                solution.rank = rank
                solutions_in_current_rank_list.append(solution)

            solutions_to_touch_set = set(solutions_to_touch_list)
            new_aux_population = []
            for solution in aux_population:
                if solution not in solutions_to_touch_set:
                    new_aux_population.append(solution)
            aux_population = new_aux_population
            solutions_by_rank_list.append(solutions_in_current_rank_list)
            rank += 1
        
        
        for rank_list in solutions_by_rank_list:
            for solution in rank_list:
                solution.crowding_distance = 0
            for solution_dimension in range(3):
                rank_list.sort(key=lambda solution: solution.fitness[solution_dimension])
                min_value = rank_list[0].fitness[solution_dimension]
                max_value = rank_list[-1].fitness[solution_dimension]
                if max_value - min_value == 0:
                    continue
                rank_list[0].crowding_distance = float('inf')
                rank_list[-1].crowding_distance = float('inf')
                for before_solution, solution, after_solution in zip(rank_list, rank_list[1:], rank_list[2:]):
                    solution.crowding_distance += (after_solution.fitness[solution_dimension] - before_solution.fitness[solution_dimension]) / (max_value - min_value)

        total_size = 0
        for rank_list in solutions_by_rank_list:
            total_size += len(rank_list)
        
        population_to_trim = total_size - trim_count

        if population_to_trim > 0:
            last_rank = solutions_by_rank_list[-1]
            last_rank.sort(reverse=True, key=lambda solution: solution.crowding_distance)
            for i in range(population_to_trim):
                last_rank.pop()

        final_population = []
        for rank_list in solutions_by_rank_list:
            final_population.extend(rank_list)

        self.population = final_population

        rank0 = solutions_by_rank_list[0]
        positions_fitness_list = []
        for solution in rank0:
            positions = np.where(solution.gnome == 1)[0]
            positions_fitness_list.append( (positions, solution.fitness) )

        self.pareto_history.append(positions_fitness_list)
        self.generation_count += 1

    def combinePopulation(self) -> list[Solution]:
        number_of_crossings = len(self.population) // 2
        population_R = self.population.copy()

        for i in range(number_of_crossings):
            gnome1 = self.rng.choice(self.population)
            gnome2 = self.rng.choice(self.population)
            gnome3 = self.rng.choice(self.population)
            gnome4 = self.rng.choice(self.population)

            if gnome1.better_than(gnome2):
                cross_sol_1 = gnome1
            else:
                cross_sol_1 = gnome2
            if gnome3.better_than(gnome4):
                cross_sol_2 = gnome3
            else:
                cross_sol_2 = gnome4

            child1, child2 = self.crossover(cross_sol_1, cross_sol_2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            population_R.append(child1)
            population_R.append(child2)

        return population_R

    def mutate(self, solution: Solution) -> Solution:
        total_elements = solution.gnome.size

        positions_to_flip = self.rng.choice(
            total_elements,
            size=self.max_flips,
            replace=False
        )
        should_flip = self.rng.random(self.max_flips) < self.mutation_prob
        positions_to_flip = positions_to_flip[should_flip]

        mask = np.zeros(total_elements, dtype=int)
        mask[positions_to_flip] = 1
        mask = mask.reshape(solution.gnome.shape)

        solution.gnome = self.limit_vars(solution.gnome ^ mask)
        return solution
    
    def crossover(self, parent1: Solution, parent2: Solution) -> tuple[Solution, Solution]:
        solution = None
        
        if self.cross_type == 1:
            solution = self.crossover_multipoint(parent1, parent2)
        else:
            solution = self.crossover_xor(parent1, parent2)
        
        return solution
    
    def crossover_xor(self, parent1: Solution, parent2: Solution) -> tuple[Solution, Solution]:
        xor_mask = parent1.gnome ^ parent2.gnome
        positions = self.rng.integers(0, 2, size=xor_mask.shape)
        final_list = (xor_mask * positions) == 1
        child_gnome_1 = np.where(final_list, parent2.gnome, parent1.gnome)
        child_gnome_2 = np.where(final_list, parent1.gnome, parent2.gnome)

        child_gnome_1 = self.limit_vars(child_gnome_1)
        child_gnome_2 = self.limit_vars(child_gnome_2)

        return Solution(child_gnome_1), Solution(child_gnome_2)
    
    def crossover_multipoint(self, parent1: Solution, parent2: Solution) -> tuple[Solution, Solution]:
        punto_corte = self.rng.integers(1, parent1.gnome.shape[0])

        child_gnome_1 = np.concatenate([
            parent1.gnome[:punto_corte],
            parent2.gnome[punto_corte:]
        ])
        child_gnome_2 = np.concatenate([
            parent2.gnome[:punto_corte],
            parent1.gnome[punto_corte:]
        ])

        child_gnome_1 = self.limit_vars(child_gnome_1)
        child_gnome_2 = self.limit_vars(child_gnome_2)

        return Solution(child_gnome_1), Solution(child_gnome_2)

    def get_pareto_front(self, rank: int = 0):
        solutions_in_pareto_front = []
        for solution in self.population:
            if solution.rank == rank:
                solutions_in_pareto_front.append(solution)
        return solutions_in_pareto_front