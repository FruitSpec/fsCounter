import numpy as np
import random
from typing import List, Tuple, Callable
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from MHS.models import models
from sklearn.pipeline import Pipeline
import joblib
from sklearn.preprocessing import (MinMaxScaler, StandardScaler, PowerTransformer,
                                   OneHotEncoder, RobustScaler, QuantileTransformer)
import os
from vision.misc.help_func import validate_output_path
from MHS.F_model_models_dict import F_model_models
from MHS.F_model_training import (read_data, read_f_df, get_rel_cols, get_full_name, clean_the_df,
                                  add_fs, add_features, process_fruit_type)
from omegaconf import OmegaConf
from tqdm import tqdm
import json


global scalers
scalers = {"Standard": StandardScaler(),
           "MinMax": MinMaxScaler(),
           "Power": PowerTransformer(),
           "Quantile": QuantileTransformer(),
           "Robust": RobustScaler()}


class GeneticAlgorithm:
    """
    A Genetic Algorithm class for feature selection with tournament selection, elitism and penalization.

    Attributes:
        model: A machine learning model that has a fit and predict method.
        X: A 2D numpy array of features.
        y: A 1D numpy array of target variable.
        population_size: An integer representing the number of individuals in the population.
        mutation_rate: A float representing the probability of mutation.
        generations: An integer representing the number of generations.
        maximize: A boolean representing whether the score should be maximized or minimized.
        cv: An integer representing the number of cross-validation folds.
        scoring: A callable object representing the scoring function.
        tournament_size: An integer representing the size of the tournament in tournament selection.
        elitism: A float representing the fraction of the best individuals to keep.
        fitness_cache: A dictionary for caching the fitness of individuals.
    """
    def __init__(self, model: Callable, X: np.ndarray, y: np.ndarray,
                 population_size: int = 200, mutation_rate: float = 0.2, generations: int = 10,
                 maximize: bool = False, cv: int = 5,
                 scoring: Callable = mean_absolute_percentage_error, tournament_size: int = 10, elitism: float = 0.1,
                 n_features_penalty: float = 0, njobs: int = 5):
        self.model = model
        self.X = X
        self.y = y
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.maximize = maximize
        self.cv = cv
        self.scoring = scoring
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.feature_size = X.shape[1]
        self.fitness_cache = {}
        self.n_features_penalty = n_features_penalty
        self.njobs = njobs

        # Initialize population
        self.population = self._initialize_population(self.population_size)

    def _initialize_population(self, population_size: int) -> List[List[int]]:
        """Initialize the population with binary chromosomes"""
        population = [self._create_individual() for _ in range(population_size // 2)]
        # Add complementary individuals to increase diversity
        population.extend([self._create_complementary_individual(individual) for individual in population])
        return population

    def _create_individual(self) -> List[int]:
        """Create an individual with a binary chromosome"""
        return [random.randint(0, 1) for _ in range(self.feature_size)]

    def _create_complementary_individual(self, individual: List[int]) -> List[int]:
        """Create a complementary individual with a binary chromosome"""
        return [1 - gene for gene in individual]

    def _calculate_fitness(self, individual: List[int]) -> float:
        """Calculate the fitness of an individual with a penalty for more features"""
        if not sum(individual):
            return -np.inf if self.maximize else np.inf
        individual_str = ''.join(map(str, individual))
        if individual_str in self.fitness_cache:
            return self.fitness_cache[individual_str]

        mask = np.array(individual, dtype=bool)
        X_subset = self.X.loc[:, mask]
        n_features = np.sum(mask)
        scores = cross_val_score(self.model, X_subset, self.y, cv=self.cv,
                                 scoring=make_scorer(self.scoring), n_jobs=self.njobs)
        fitness = np.mean(scores) + self.n_features_penalty * n_features  # add a penalty for more features

        self.fitness_cache[individual_str] = fitness
        return fitness

    def _tournament_selection(self) -> List[List[int]]:
        """Select individuals for reproduction based on their fitness in a tournament"""
        selected_individuals = []
        print("running tournament_selection")
        for _ in tqdm(range(self.population_size)):
            participants = random.sample(list(enumerate(self.population)), self.tournament_size)
            winner = max(participants, key=lambda x: self._calculate_fitness(x[1])) if self.maximize else\
                min(participants, key=lambda x: self._calculate_fitness(x[1]))
            selected_individuals.append(winner[1])
        return selected_individuals

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Perform uniform crossover"""
        child1 = []
        child2 = []
        for gene1, gene2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child1.append(gene1)
                child2.append(gene2)
            else:
                child1.append(1 - gene1)
                child2.append(1 - gene2)
        return child1, child2

    def _mutation(self, individual: List[int]) -> List[int]:
        """Perform mutation"""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    def _get_next_generation(self, selected_individuals: List[List[int]]) -> List[List[int]]:
        """Generate the next generation of individuals"""
        # Elitism: keep the top performers from the current population
        n_elite = int(self.elitism * self.population_size)
        elite = sorted(selected_individuals, key=self._calculate_fitness, reverse=self.maximize)[:n_elite]
        next_generation = elite
        while len(next_generation) < self.population_size:
            parent1, parent2 = random.sample(selected_individuals, 2)
            child1, child2 = self._crossover(parent1, parent2)
            next_generation.append(self._mutation(child1))
            if len(next_generation) < self.population_size:
                next_generation.append(self._mutation(child2))
        return next_generation

    def run(self) -> List[int]:
        """Run the genetic algorithm"""
        generation = 0
        while generation < self.generations:
            selected_individuals = self._tournament_selection()
            self.population = self._get_next_generation(selected_individuals)
            best_individual = max(self.population, key=self._calculate_fitness) if self.maximize\
                else min(self.population, key=self._calculate_fitness)
            best_fitness = self._calculate_fitness(best_individual)
            print(f"Generation {generation+1}: Best fitness = {best_fitness}")
            generation += 1
            print("current best features:")
            print(self.X.loc[:, np.array(best_individual, dtype=bool)].columns)
        return self.X.loc[:, np.array(best_individual, dtype=bool)].columns, best_fitness


def custom_make_regression(n_samples=100, n_features=1, n_informative=1, noise=0.0, random_state=None, loc=20, scale=5):
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal(size=(n_samples, n_features))

    # Generating informative features
    informative_indices = rng.choice(n_features, size=n_informative, replace=False)
    informative_features = rng.standard_normal(size=(n_samples, n_informative))

    # Combining informative features with noise features
    X[:, informative_indices] = informative_features

    # Generating target variable
    coef = rng.normal(loc=loc, scale=scale, size=n_informative)
    y = X[:, informative_indices].dot(coef) + \
        + noise * rng.standard_normal(size=n_samples)

    return X, y, informative_indices


def run_exmaple():
    # Example usage:
    n_samples = 500
    n_features = 20
    n_informative = 6
    noise = 0.2
    random_state = 43

    X, y, informative_indices = custom_make_regression(n_samples=n_samples, n_features=n_features,
                                                       n_informative=n_informative, noise=noise, random_state=random_state)
    print(sorted(informative_indices))
    # Run the genetic algorithm with a linear regression model
    ga = GeneticAlgorithm(LinearRegression(), X, y, generations=20, population_size=200,
                          mutation_rate=0.1, tournament_size=20, elitism=0.2)
    best_features, best_fitness = ga.run()

    print(f"Best features: {np.where(best_features)}")


def run_gfs(model_name, model_args, features_df, generations=50, population_size=500,
                          mutation_rate=0.3, tournament_size=10, elitism=0.2):

    model = models[model_name.split("_")[0]]
    model_params = model_args["model_params"]
    if isinstance(model_params, str):
        study = joblib.load(model_params)
        model_params = study.best_params
    gfs_train_cols = model_args["gfs_train_cols"]
    if isinstance(gfs_train_cols, str):
        print("no cols for:", model_name)
        return
    if not gfs_train_cols:
        print("no cols for:", model_name)
        return
    X_train = features_df[model_args["gfs_train_cols"]]
    y_train = features_df["F"]
    model.set_params(**model_params)
    if "scaler" in model_args.keys():
        scaler = scalers[model_args["scaler"]]
        model = Pipeline([("scaler", scaler),
                          ("final_estimator", model)])
    save_name = model_args["output_path"]
    validate_output_path(os.path.dirname(save_name))
    ga = GeneticAlgorithm(model, X_train, y_train, generations=generations, population_size=population_size,
                          mutation_rate=mutation_rate, tournament_size=tournament_size, elitism=elitism)
    best_features, best_fitness = ga.run()
    out_json = {"features": list(best_features), "fitness": best_fitness}
    output_path = os.path.join(os.path.dirname(model_args["output_path"]), f"{model_name}_ga_features.json")
    with open(output_path, 'w') as file:
        json.dump(out_json, file)
    print(model_name)
    print(best_features)


if __name__ == "__main__":
    use_best_study = True
    cfg = OmegaConf.load("model_config.yaml")
    f_df = read_f_df(cfg)
    features_df = read_data(cfg)
    final_cols, drop_final = get_rel_cols(cfg)
    f_df = read_f_df(cfg)
    features_df = read_data(cfg)
    full_name_org = get_full_name(features_df)
    features_df_clean = clean_the_df(features_df, drop_final, cfg)
    features_df_w_f = add_fs(features_df_clean, f_df)
    skip_models = ["XGBRegressor", "XGBRegressor_local", "DecisionTreeRegressor", "RandomForestRegressor"]

    if "fruit_type" in features_df_w_f.columns:
        df = process_fruit_type(features_df_w_f, cfg.cleaning.fruits_exclude, cfg)
        df.drop("fruit_type", axis=1, inplace=True)
    features_df_w_f = add_features(features_df_w_f)

    for model_name, model_args in F_model_models.items():
        if model_name in skip_models:
            continue
        print("starting: ", model_name)
        run_gfs(model_name, model_args, features_df_w_f.reset_index(drop=True))
