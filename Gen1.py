import deap
import random
from deap import base, creator, tools

temp_env= 20
ph_Env= 7
optimal_nutrition_efficiency = 1

RANDOM_SEED = 42
random.seed(RANDOM_SEED)#no variance in testing

chromosome_length = 100
population_size = 1000

creator.create("FitnessMin", base.Fitness, weights=(-1,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("zeroOrOne", random.randint, 0, 1) #used in creating chromosome with 0 or 1

toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.zeroOrOne, n=chromosome_length)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    temp_fitness = abs(temp_env-sum(individual))/1000
    return (temp_fitness,)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)



toolbox.population(n=population_size)

