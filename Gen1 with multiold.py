
if not hasattr(creator, "FitnessMin"):
    creator.create(
        "FitnessMin",
        base.Fitness,
        weights=(-1.0,)
    )

if not hasattr(creator, "Individual"):
    creator.create(
        "Individual",
        list,
        fitness=creator.FitnessMin
    )


# --------------------------------
# UMGEBUNGSPARAMETER
# --------------------------------

# Diese Variablen werden auch von den Worker-Prozessen verwendet.
temp_env = None
ph_Env = None
optimal_nutrition_efficiency = None
salt_concentration = None


def init_worker(environment):
    """
    Wird einmal beim Start jedes Worker-Prozesses aufgerufen.
    Überträgt die Umgebungswerte in den jeweiligen Prozess.
    """
    global temp_env
    global ph_Env
    global optimal_nutrition_efficiency
    global salt_concentration

    temp_env = environment["temperature"]
    ph_Env = environment["ph"]
    optimal_nutrition_efficiency = environment["nutrition"]
    salt_concentration = environment["salt"]


# --------------------------------
# FITNESS-FUNKTION
# --------------------------------

def evaluate_details(individual):

    temp_fitness = abs(
        (temp_env + 273) / 573
        - sum(individual[0:100]) / 100
    )

    ph_fitness = abs(
        ph_Env / 14
        - sum(individual[100:200]) / 100
    )

    nut_fitness = abs(
        optimal_nutrition_efficiency
        - sum(individual[200:300]) / 100
    )

    salt_fitness = abs(
        salt_concentration
        - sum(individual[300:400]) / 100
    )

    return (
        temp_fitness,
        ph_fitness,
        nut_fitness,
        salt_fitness
    )


def evaluate(individual):

    details = evaluate_details(individual)
    general_fitness = sum(details)
    return (general_fitness,)


# --------------------------------
# TOOLBOX
# --------------------------------

def create_toolbox():

    toolbox = base.Toolbox()

    toolbox.register("zeroOrOne", random.randint, 0, 1)

    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.zeroOrOne,
        n=chromosome_length
    )

    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual,
        n=popSize
    )

    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    return toolbox


# --------------------------------
# HAUPTPROGRAMM
# --------------------------------

def main():

    random.seed(RANDOM_SEED)

    # Umgebung auswählen
    environments = {
        "1": "enviroment1.json",
        "2": "enviroment2.json",
        "3": "enviroment3.json"
    }

    print("Select Environment (1, 2 or 3):")
    choice = input().strip()

    if choice not in environments:
        raise ValueError("Invalid environment selected!")

    with open(environments[choice], "r") as f:
        environment = json.load(f)

    # Umgebungswerte auch im Hauptprozess setzen
    init_worker(environment)

    # Toolbox erstellen
    toolbox = create_toolbox()

    # Statistiken
    bestTempFitness = []
    meanTempFitness = []

    bestPhFitness = []
    meanPhFitness = []

    bestNutFitness = []
    meanNutFitness = []

    bestSaltFitness = []
    meanSaltFitness = []

    bestFitness = []
    meanFitness = []

    start_time = time.perf_counter()

    # Population erstellen
    print("Creating population...")
    pop = toolbox.population()

    print(f"Starting multiprocessing with {workers} workers...")
    print(f"Population size: {popSize}")
    print(f"Chunksize: {CHUNKSIZE}")

    # --------------------------------
    # MULTIPROCESSING
    # --------------------------------

    with multiprocessing.Pool(
            processes=workers,
            initializer=init_worker,
            initargs=(environment,)
    ) as pool:

        # DEAP verwendet ab jetzt pool.map statt der normalen map-Funktion.
        toolbox.register(
            "map",
            pool.map,
            chunksize=CHUNKSIZE
        )

        # --------------------------------

