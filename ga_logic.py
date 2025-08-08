import pandas as pd
import numpy as np
import random
import copy
from sklearn.metrics import r2_score

# ==============================================================================
# DATA CONSTANTS
# ==============================================================================

AKG_BASE = {
    '19-29': {
        'Energi (kkal)': 2250, 'Protein (g)': 60, 'Lemak Total (g)': 65,
        'Karbohidrat (g)': 360, 'Serat (g)': 32, 'Air (ml)': 2350,
        'Vit B1 (mg)': 1.1, 'Vit B2 (mg)': 1.1, 'Vit C (mg)': 75,
        'Kalsium (mg)': 1000, 'Fosfor (mg)': 700, 'Besi (mg)': 18,
        'Seng (mg)': 8, 'Kalium (mg)': 4700, 'Natrium (mg)': 1500,
        'Tembaga (mcg)': 900
    },
    '30-49': {
        'Energi (kkal)': 2150, 'Protein (g)': 60, 'Lemak Total (g)': 60,
        'Karbohidrat (g)': 340, 'Serat (g)': 30, 'Air (ml)': 2350,
        'Vit B1 (mg)': 1.1, 'Vit B2 (mg)': 1.1, 'Vit C (mg)': 75,
        'Kalsium (mg)': 1000, 'Fosfor (mg)': 700, 'Besi (mg)': 18,
        'Seng (mg)': 8, 'Kalium (mg)': 4700, 'Natrium (mg)': 1500,
        'Tembaga (mcg)': 900
    }
}

TRIMESTER_ADDITIONS = {
    1: {
        'Energi (kkal)': 180, 'Protein (g)': 1, 'Lemak Total (g)': 2.3,
        'Karbohidrat (g)': 25, 'Serat (g)': 3, 'Air (ml)': 300,
        'Vit B1 (mg)': 0.3, 'Vit B2 (mg)': 0.3, 'Vit C (mg)': 10,
        'Kalsium (mg)': 200, 'Seng (mg)': 2, 'Tembaga (mcg)': 100
    },
    2: {
        'Energi (kkal)': 300, 'Protein (g)': 10, 'Lemak Total (g)': 2.3,
        'Karbohidrat (g)': 40, 'Serat (g)': 4, 'Air (ml)': 300,
        'Vit B1 (mg)': 0.3, 'Vit B2 (mg)': 0.3, 'Vit C (mg)': 10,
        'Kalsium (mg)': 200, 'Besi (mg)': 9, 'Seng (mg)': 4,
        'Tembaga (mcg)': 100
    },
    3: {
        'Energi (kkal)': 300, 'Protein (g)': 30, 'Lemak Total (g)': 2.3,
        'Karbohidrat (g)': 40, 'Serat (g)': 4, 'Air (ml)': 300,
        'Vit B1 (mg)': 0.3, 'Vit B2 (mg)': 0.3, 'Vit C (mg)': 10,
        'Kalsium (mg)': 200, 'Besi (mg)': 9, 'Seng (mg)': 4,
        'Tembaga (mcg)': 100
    }
}

# ==============================================================================
# GA CONFIGURATION
# ==============================================================================
GA_CONFIG = {
    'POPULATION_SIZE': 100,
    'NUM_GENERATIONS': 50,
    'TOURNAMENT_SIZE': 5,
    'CROSSOVER_PROBABILITY': 0.6,
    'MUTATION_PROBABILITY': 0.1,
    'ELITISM_COUNT': 2,
    'DISTRIBUTION_PENALTY_WEIGHT': 0.3,
    'MEAL_SLOTS': {
        'Breakfast': 4,
        'Snack 1': 2,
        'Lunch': 4,
        'Snack 2': 2,
        'Dinner': 4
    },
    'PORTION_SIZES': [0.5, 1.0, 1.5],
    'CALORIE_ALLOCATION': {
        'Breakfast': 0.22,
        'Snack 1': 0.06,
        'Lunch': 0.31,
        'Snack 2': 0.06,
        'Dinner': 0.35
    }
}


# ==============================================================================
# DATA LOADING FUNCTION
# ==============================================================================
def load_food_data(path="static/food_dataset.csv"):
    """Loads, cleans, and prepares the food nutrition dataset."""
    try:
        df_raw = pd.read_csv(path, na_values=['-', 'n/a', ''])
    except FileNotFoundError:
        print(f"Error: The file was not found at the path: {path}")
        return None

    df_raw.columns = df_raw.columns.str.strip().str.replace('\\u00a0', ' ', regex=True)
    column_mapping = {
        'Air(g)': 'Air (g)', 'Energi(Kal)': 'Energi (kkal)', 'Protein(g)': 'Protein (g)',
        'Lemak(g)': 'Lemak (g)', 'Karbohidrat(g)': 'Karbohidrat (g)', 'Serat(g)': 'Serat (g)',
        'Kalsium(mg)': 'Kalsium (mg)', 'Fosfor(mg)': 'Fosfor (mg)', 'Besi(mg)': 'Besi (mg)',
        'Natrium(mg)': 'Natrium (mg)', 'Kalium(mg)': 'Kalium (mg)', 'Tembaga(mg)': 'Tembaga (mg)',
        'Seng(mg)': 'Seng (mg)', 'Vit B1(mg)': 'Vit B1 (mg)', 'Vit B2(mg)': 'Vit B2 (mg)',
        'Vitamin C(mg)': 'Vit C (mg)'
    }
    nutrient_columns = list(column_mapping.values())
    df_renamed = df_raw.rename(columns=column_mapping)

    for col in nutrient_columns:
        if col in df_renamed.columns:
            df_renamed[col] = pd.to_numeric(df_renamed[col], errors='coerce')

    final_columns = ['Makanan', 'Utama/Snack'] + [col for col in nutrient_columns if col != 'Air (g)']
    food_df = df_renamed[final_columns].fillna(0).set_index('Makanan')

    for col in nutrient_columns:
        if col in food_df.columns and col != 'Air (g)':
            food_df[col] = food_df[col].astype(float)
            
    return food_df


# ==============================================================================
# NUTRITIONAL CALCULATION FUNCTIONS
# ==============================================================================
def _calculate_tee(age: int, weight_kg: float, height_cm: float) -> float:
    if age <= 29:
        tee = ((10 * weight_kg) + (6.25 * height_cm) - 281) * 1.78
    else:
        tee = ((10 * weight_kg) + (6.25 * height_cm) - 361) * 1.81
    return tee

def calculate_final_needs(age: int, weight_kg: float, height_cm: float, trimester: int, medical_conditions: dict) -> dict:
    """Calculates the final daily nutritional needs based on user profile."""
    if age <= 29:
        needs = copy.deepcopy(AKG_BASE['19-29'])
    else:
        needs = copy.deepcopy(AKG_BASE['30-49'])
        
    tee = _calculate_tee(age, weight_kg, height_cm)
    needs['Energi (kkal)'] = tee
    
    trimester_add = TRIMESTER_ADDITIONS.get(trimester, {})
    for nutrient, value in trimester_add.items():
        needs[nutrient] = needs.get(nutrient, 0) + value
        
    if medical_conditions.get('hypertension'): needs['Natrium (mg)'] -= 300
    if medical_conditions.get('pre_eclampsia'): needs['Kalsium (mg)'] += 600
    if medical_conditions.get('anemia'):
        needs['Besi (mg)'] += 40 if trimester == 1 else 30
        needs['Vit C (mg)'] += 50
    if medical_conditions.get('diabetes'):
        new_carb_grams = max(175.0, (needs['Energi (kkal)'] * 0.40) / 4)
        needs['Energi (kkal)'] = max(1700.0, needs['Energi (kkal)'] - ((needs.get('Karbohidrat (g)', new_carb_grams) - new_carb_grams) * 4))
        needs['Karbohidrat (g)'] = new_carb_grams
        
    if 'Tembaga (mcg)' in needs: needs['Tembaga (mg)'] = needs.pop('Tembaga (mcg)') / 1000.0
    if 'Lemak Total (g)' in needs: needs['Lemak (g)'] = needs.pop('Lemak Total (g)')
    if 'Air (ml)' in needs: needs.pop('Air (ml)')
            
    final_nutrient_keys = [
        'Energi (kkal)', 'Protein (g)', 'Lemak (g)', 'Karbohidrat (g)', 'Serat (g)', 
        'Kalsium (mg)', 'Fosfor (mg)', 'Besi (mg)', 'Natrium (mg)', 'Kalium (mg)', 
        'Tembaga (mg)', 'Seng (mg)', 'Vit B1 (mg)', 'Vit B2 (mg)', 'Vit C (mg)'
    ]
    return {key: needs[key] for key in final_nutrient_keys if key in needs}

# ==============================================================================
# METRIC CALCULATION FUNCTIONS
# ==============================================================================
def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100

def calculate_smape(y_true, y_pred):
    """Calculates Symmetric Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_pred - y_true) / (denominator + 1e-6)) * 100

def calculate_r2(y_true, y_pred):
    """Calculates Coefficient of Determination (R^2)."""
    return r2_score(y_true, y_pred)

# ==============================================================================
# BASE GENETIC ALGORITHM CLASS
# ==============================================================================
class GeneticAlgorithm:
    def __init__(self, food_df, config):
        self.food_df = food_df
        self.config = config
        self.nutrient_keys = [col for col in food_df.columns if col != 'Utama/Snack']
        self.food_indices = {
            'Utama': self.food_df[self.food_df['Utama/Snack'] == 'Utama'].index,
            'Snack': self.food_df[self.food_df['Utama/Snack'] == 'Snack'].index,
            'Semua': self.food_df[self.food_df['Utama/Snack'] == 'Semua'].index
        }
        self.all_food_indices = self.food_df.index

    def _create_individual(self):
        individual = []
        for meal, slots in self.config['MEAL_SLOTS'].items():
            for _ in range(slots):
                if 'Snack' in meal:
                    food_source = self.food_indices['Snack'].union(self.food_indices['Semua'])
                else:
                    food_source = self.food_indices['Utama'].union(self.food_indices['Semua'])
                random_food = random.choice(food_source.tolist())
                random_portion = random.choice(self.config['PORTION_SIZES'])
                individual.append((random_food, random_portion))
        return individual

    def _calculate_fitness(self, individual, target_needs):
        total_nutrients = {key: 0.0 for key in self.nutrient_keys}
        segment_calories = {meal: 0.0 for meal in self.config['MEAL_SLOTS']}
        current_slot = 0
        for meal, slots in self.config['MEAL_SLOTS'].items():
            for i in range(slots):
                food_name, portion = individual[current_slot + i]
                food_data = self.food_df.loc[food_name]
                for key in self.nutrient_keys:
                    total_nutrients[key] += food_data[key] * portion
                segment_calories[meal] += food_data['Energi (kkal)'] * portion
            current_slot += slots
        errors = [np.abs(total_nutrients.get(k, 0) - t) / (t + 1e-6) for k, t in target_needs.items()]
        nutrient_mape = np.mean(errors)
        total_actual_calories = total_nutrients['Energi (kkal)']
        dist_errors = [np.abs(segment_calories[m] / (total_actual_calories + 1e-6) - p) for m, p in self.config['CALORIE_ALLOCATION'].items()]
        distribution_penalty = np.mean(dist_errors)
        w = self.config['DISTRIBUTION_PENALTY_WEIGHT']
        total_error = (1 - w) * nutrient_mape + w * distribution_penalty
        fitness = 1 / (1 + total_error)
        return fitness, total_nutrients

    def _selection(self, population_with_fitness):
        sorted_population = sorted(population_with_fitness, key=lambda x: x[1][0], reverse=True)
        next_generation_parents = []
        for _ in range(self.config['POPULATION_SIZE']):
            tournament = random.sample(sorted_population, self.config['TOURNAMENT_SIZE'])
            winner = max(tournament, key=lambda x: x[1][0])
            next_generation_parents.append(winner[0])
        return next_generation_parents

    def _crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutate(self, individual, mutation_rate):
        mutated_individual = list(individual)
        for i in range(len(mutated_individual)):
            if random.random() < mutation_rate:
                 if random.random() < 0.5:
                    mutated_individual[i] = (random.choice(self.all_food_indices.tolist()), mutated_individual[i][1])
                 else:
                    mutated_individual[i] = (mutated_individual[i][0], random.choice(self.config['PORTION_SIZES']))
        return mutated_individual

    def run(self, user_profile):
        target_needs = calculate_final_needs(**user_profile)
        population = [self._create_individual() for _ in range(self.config['POPULATION_SIZE'])]
        
        for gen in range(self.config['NUM_GENERATIONS']):
            pop_with_fitness = [(ind, self._calculate_fitness(ind, target_needs)) for ind in population]
            sorted_pop = sorted(pop_with_fitness, key=lambda x: x[1][0], reverse=True)
            next_population = [sorted_pop[i][0] for i in range(self.config['ELITISM_COUNT'])]
            parents_for_breeding = self._selection(pop_with_fitness)
            
            while len(next_population) < self.config['POPULATION_SIZE']:
                p1, p2 = random.choices(parents_for_breeding, k=2)
                if random.random() < self.config['CROSSOVER_PROBABILITY']:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1, p2
                
                mutation_rate = self.config['MUTATION_PROBABILITY']
                next_population.append(self._mutate(c1, mutation_rate))
                if len(next_population) < self.config['POPULATION_SIZE']:
                    next_population.append(self._mutate(c2, mutation_rate))

            population = next_population            
            if (gen + 1) % 10 == 0:
                print(f"GA Gen {gen+1}/{self.config['NUM_GENERATIONS']} -> Best Fitness: {sorted_pop[0][1][0]:.4f}")
    
        final_pop_with_fitness = [(ind, self._calculate_fitness(ind, target_needs)) for ind in population]
        best_individual, (best_fitness, final_nutrients) = max(final_pop_with_fitness, key=lambda x: x[1][0])
        
        return best_individual, final_nutrients, target_needs, best_fitness

# ==============================================================================
# ADAPTIVE GENETIC ALGORITHM CLASS
# ==============================================================================
class AGAScenario1(GeneticAlgorithm):
    def _calculate_adaptive_probabilities(self, fitness, f_avg, f_min):
        pc_max, pc_min = 0.9, 0.6
        pm_max, pm_min = 0.15, 0.1

        if fitness < f_avg:
            denominator = f_avg - f_min + 1e-6
            factor = (f_avg - fitness) / denominator
            pc = pc_min + (pc_max - pc_min) * factor
            pm = pm_min + (pm_max - pm_min) * factor
        else:
            pc = pc_min
            pm = pm_min
        return pc, pm

    def run(self, user_profile):
        target_needs = calculate_final_needs(**user_profile)
        population = [self._create_individual() for _ in range(self.config['POPULATION_SIZE'])]
        
        for gen in range(self.config['NUM_GENERATIONS']):
            pop_with_fitness = [(ind, self._calculate_fitness(ind, target_needs)) for ind in population]
            fitness_scores = [f[1][0] for f in pop_with_fitness]
            f_max, f_avg, f_min = np.max(fitness_scores), np.mean(fitness_scores), np.min(fitness_scores)
            sorted_pop = sorted(pop_with_fitness, key=lambda x: x[1][0], reverse=True)
            next_population = [ind for ind, _ in sorted_pop[:self.config['ELITISM_COUNT']]]
            parents_for_breeding = self._selection(pop_with_fitness)
            parent_fitness_lookup = {tuple(ind): fit[0] for ind, fit in pop_with_fitness}
            
            while len(next_population) < self.config['POPULATION_SIZE']:
                p1, p2 = random.choices(parents_for_breeding, k=2)
                p1_fitness = parent_fitness_lookup[tuple(p1)]
                pc, pm1 = self._calculate_adaptive_probabilities(p1_fitness, f_avg, f_min)
                if random.random() < pc:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1, p2
                next_population.append(self._mutate(c1, pm1))
                if len(next_population) < self.config['POPULATION_SIZE']:
                    p2_fitness = parent_fitness_lookup[tuple(p2)]
                    _, pm2 = self._calculate_adaptive_probabilities(p2_fitness, f_avg, f_min)
                    next_population.append(self._mutate(c2, pm2))
            
            population = next_population            
            if (gen + 1) % 10 == 0:
                print(f"AGA Gen {gen+1}/{self.config['NUM_GENERATIONS']} -> Best Fitness: {f_max:.4f}")
                
        final_pop_with_fitness = [(ind, self._calculate_fitness(ind, target_needs)) for ind in population]
        best_individual, (best_fitness, final_nutrients) = max(final_pop_with_fitness, key=lambda x: x[1][0])
        
        return best_individual, final_nutrients, target_needs, best_fitness
