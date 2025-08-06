# QuantumLeap
Lottery simulation using frequency-driven Monte Carlo evolution, adaptive weights, spatial heuristics, and feedback memory. Generates optimized picks by analyzing hit patterns, number distribution, and historical data performance.


import pandas as pd
import numpy as np
import random

# 50 fictional Powerball draws (1–69, Powerball 1–26)
draw_history = [
    [random.randint(1, 69) for _ in range(5)] + [random.randint(1, 26)] for _ in range(50)
]
for i in range(len(draw_history)):
    while len(set(draw_history[i][:5])) < 5:
        draw_history[i][:5] = random.sample(range(1, 70), 5)
    draw_history[i][:5] = sorted(draw_history[i][:5])

# Class for cycle memory management
class CycleMemory:
    def __init__(self, max_cycles=5):
        self.cycles = []
        self.max_cycles = max_cycles
    
    def save_cycle(self, numbers, powerball, hits_per_draw, score):
        if len(self.cycles) >= self.max_cycles:
            self.cycles.pop(0)
        self.cycles.append((numbers, powerball, hits_per_draw, score))
        print("Cycle saved to memory.")
    
    def get_memory(self):
        return self.cycles
    
    def clear_memory(self):
        self.cycles = []
        print("Cycle memory cleared.")

# Class for adaptive weights
class AdaptiveWeights:
    def __init__(self, main_freq, powerball_freq):
        self.main_weights = {num: 1.0 + (main_freq.get(num, 0) / max(main_freq.values, default=1)) for num in range(1, 70)}
        self.powerball_weights = {num: 1.0 + (powerball_freq.get(num, 0) / max(powerball_freq.values, default=1)) for num in range(1, 27)}
    
    def update_weights(self, cycles):
        for numbers, powerball, hits_per_draw, score in cycles:
            avg_hits = np.mean([hits[0] for _, hits in hits_per_draw])
            for num in numbers:
                self.main_weights[num] += (avg_hits + score) * 0.1
            if any(hits[1] > 0 for _, hits in hits_per_draw):
                self.powerball_weights[powerball] += 0.2
            for num in range(1, 70):
                if num not in numbers:
                    self.main_weights[num] *= 0.95
            for num in range(1, 27):
                if num != powerball:
                    self.powerball_weights[num] *= 0.95
        max_weight = max(self.main_weights.values())
        if max_weight > 0:
            for num in self.main_weights:
                self.main_weights[num] /= max_weight
    
    def choose_number(self, candidates):
        if not candidates:
            return random.randint(1, 69)
        weights = [self.main_weights[num] for num in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]
    
    def choose_powerball(self):
        weights = [self.powerball_weights[num] for num in range(1, 27)]
        return random.choices(list(range(1, 27)), weights=weights, k=1)[0]

# Function to load and validate data
def load_data(draws):
    df = pd.DataFrame(draws, columns=['N1', 'N2', 'N3', 'N4', 'N5', 'Powerball'])
    if df.empty:
        raise ValueError("No draws provided.")
    for idx, draw in df.iterrows():
        main_numbers = draw[:5]
        powerball = draw[5]
        if len(set(main_numbers)) != 5 or any(not 1 <= num <= 69 for num in main_numbers) or not 1 <= powerball <= 26:
            raise ValueError(f"Invalid draw {idx}.")
    print(f"Loaded history with {len(df)} Powerball draws")
    return df

# Function to analyze frequency
def analyze_frequency(df):
    main_numbers = df.iloc[:, :5].values.flatten()
    powerballs = df['Powerball'].values
    main_freq = pd.Series(main_numbers).value_counts().sort_index()
    powerball_freq = pd.Series(powerballs).value_counts().sort_index()
    
    print("\nMain numbers frequency (1–69):")
    for num, freq in main_freq.items():
        print(f" - Number {num:02d}: {freq} time(s)")
    print("\nPowerball frequency (1–26):")
    for num, freq in powerball_freq.items():
        print(f" - Powerball {num:02d}: {freq} time(s)")
    
    avg = main_freq.mean()
    std = main_freq.std()
    hot = main_freq[main_freq > avg + 0.5 * std].index.tolist()
    warm = main_freq[(main_freq >= avg - 0.5 * std) & (main_freq <= avg + 0.5 * std)].index.tolist()
    cold = main_freq[main_freq < avg - 0.5 * std].index.tolist()
    
    print(f"\nFrequency average (main): {avg:.2f}")
    print(f"Standard deviation (main): {std:.2f}")
    print(f"Hot ({len(hot)}): {sorted(hot)}")
    print(f"Warm ({len(warm)}): {sorted(warm)}")
    print(f"Cold ({len(cold)}): {sorted(cold)}\n")
    
    return main_freq, powerball_freq, hot, warm, cold

# Function to check spatial distribution
def check_spatial_distribution(numbers):
    ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 69)]
    covered_ranges = set()
    for num in numbers:
        for i, (start, end) in enumerate(ranges):
            if start <= num <= end:
                covered_ranges.add(i)
    prob = random.uniform(0, 1)
    return len(covered_ranges) >= 3 or prob > 0.8

# Function to calculate game score
def calculate_score(numbers, base_sum, hot, cold):
    numbers_sum = sum(numbers)
    covered_ranges = len(set([min((num - 1) // 10, 6) for num in numbers]))
    hot_in_game = len([num for num in numbers if num in hot])
    cold_in_game = len([num for num in numbers if num in cold])
    even = len([num for num in numbers if num % 2 == 0])
    
    sum_score = 1 - min(abs(numbers_sum - base_sum) / base_sum, 1)
    ranges_score = covered_ranges / 5
    hot_score = 1 if hot_in_game > cold_in_game else 0.5
    even_score = 1 if 2 <= even <= 3 else 0.5
    
    return 0.35 * sum_score + 0.25 * ranges_score + 0.25 * hot_score + 0.15 * even_score

# Function to generate game with Monte Carlo mutation
def generate_game_with_mutation(hot, warm, cold, weights, base_sum, previous_avg_hits=0):
    max_attempts = 100
    attempt = 0
    best_game = None
    best_powerball = None
    best_score = 0
    
    # Dynamically adjust sum weight based on previous hits
    sum_weight = 0.35 if previous_avg_hits < 1 else 0.45
    
    while attempt < max_attempts:
        game = []
        game += random.sample(hot, min(2, len(hot)))
        game += random.sample(warm, min(2, len(warm)))
        game += [weights.choose_number(cold)]
        
        while len(game) < 5:
            number = weights.choose_number(list(range(1, 70)))
            if number not in game:
                game.append(number)
        
        powerball = weights.choose_powerball()
        game_sum = sum(game)
        
        consecutive = sum(1 for i in range(len(game) - 1) if game[i + 1] == game[i] + 1)
        ranges_ok = check_spatial_distribution(game)
        hot_in_game = len([num for num in game if num in hot])
        cold_in_game = len([num for num in game if num in cold])
        even = len([num for num in game if num % 2 == 0])
        
        score = calculate_score(game, base_sum, hot, cold)
        
        # Dynamic criteria with and/or
        if (abs(game_sum - base_sum) <= 20 and ranges_ok and consecutive <= 2 and hot_in_game > cold_in_game and 2 <= even <= 3) or (score > 0.7):
            return sorted(game), powerball, True, score
        
        if score > best_score:
            best_game = game
            best_powerball = powerball
            best_score = score
        
        if cold and random.uniform(0, 1) < 0.5:
            idx = random.randint(0, len(game) - 1)
            new_cold = weights.choose_number(cold)
            if new_cold not in game:
                game[idx] = new_cold
        attempt += 1
    
    print("Main criteria not met, returning best game.")
    return sorted(best_game), best_powerball, False, best_score

# Function to count hits
def count_hits(numbers, powerball, draw, draw_powerball):
    main_hits = len(set(numbers).intersection(draw[:5]))
    powerball_hit = 1 if powerball == draw_powerball else 0
    return main_hits, powerball_hit

# Function to simulate and test games with evolution
def simulate_and_test_with_evolution(df, hot, warm, cold, main_freq, powerball_freq, num_games=5, save_csv=False):
    base_sum = int(df.iloc[:, :5].sum(axis=1).mean())
    print(f"\nBase sum for mutation: {base_sum}")
    
    memory = CycleMemory()
    weights = AdaptiveWeights(main_freq, powerball_freq)
    
    print(f"\nSimulating {num_games} games with Monte Carlo mutation...")
    results = []
    evolution = {
        "Game": [],
        "Avg_Main_Hits": [],
        "Avg_Powerball_Hits": [],
        "Score": [],
        "Criteria_Met": [],
        "Covered_Ranges": [],
        "Hot_in_Game": []
    }
    
    previous_avg_hits = 0
    
    for i in range(num_games):
        numbers, powerball, criteria_met, score = generate_game_with_mutation(
            hot, warm, cold, weights, base_sum, previous_avg_hits
        )
        hits_per_draw = [(idx, count_hits(numbers, powerball, draw.values, draw['Powerball'])) for idx, draw in df.iterrows()]
        
        avg_main_hits = np.mean([hits[0] for _, hits in hits_per_draw])
        avg_powerball_hits = np.mean([hits[1] for _, hits in hits_per_draw])
        covered_ranges = len(set([min((num - 1) // 10, 6) for num in numbers]))
        hot_in_game = len([num for num in numbers if num in hot])
        
        memory.save_cycle(numbers, powerball, hits_per_draw, score)
        weights.update_weights(memory.get_memory())
        
        if not criteria_met:
            memory.clear_memory()
        
        results.append((numbers, powerball, hits_per_draw, criteria_met, score))
        
        evolution["Game"].append(i + 1)
        evolution["Avg_Main_Hits"].append(avg_main_hits)
        evolution["Avg_Powerball_Hits"].append(avg_powerball_hits)
        evolution["Score"].append(score)
        evolution["Criteria_Met"].append(criteria_met)
        evolution["Covered_Ranges"].append(covered_ranges)
        evolution["Hot_in_Game"].append(hot_in_game)
        
        previous_avg_hits = avg_main_hits
    
    # Configure DataFrame formatting
    pd.set_option('display.precision', 2)
    pd.set_option('display.colheader_justify', 'center')
    
    # Show summary results
    print("\nGame results:")
    print("-" * 50)
    for i, (numbers, powerball, hits_per_draw, criteria_met, score) in enumerate(results, 1):
        avg_main_hits = np.mean([hits[0] for _, hits in hits_per_draw])
        avg_powerball_hits = np.mean([hits[1] for _, hits in hits_per_draw])
        print(f"Game {i}: {numbers} + Powerball {powerball:02d} (Score: {score:.2f}, {'Criteria met' if criteria_met else 'Fallback'})")
        print(f"  Average main hits: {avg_main_hits:.2f}")
        print(f"  Average Powerball hits: {avg_powerball_hits:.2f}\n")
    
    # Show evolution table
    df_evolution = pd.DataFrame(evolution)
    print("\nGame performance evolution:\n")
    print(df_evolution.to_string(index=False))
    
    # Additional statistics
    print("\nGeneral statistics:")
    print(f" - Average main hits: {np.mean(evolution['Avg_Main_Hits']):.2f}")
    print(f" - Maximum main hits: {max([np.mean([hits[0] for _, hits in res[2]]) for res in results]):.2f}")
    print(f" - Average Powerball hits: {np.mean(evolution['Avg_Powerball_Hits']):.2f}")
    print(f" - Best score: {max(evolution['Score']):.2f}")
    print(f" - Best game: {results[np.argmax(evolution['Avg_Main_Hits'])][0]} + Powerball {results[np.argmax(evolution['Avg_Main_Hits'])][1]:02d}")
    print("-" * 50)
    
    # Save to CSV if requested
    if save_csv:
        df_evolution.to_csv("powerball_performance_evolution.csv", index=False)
        print("Evolution saved to 'powerball_performance_evolution.csv'.")

# Main function
def main():
    df = load_data(draw_history)
    main_freq, powerball_freq, hot, warm, cold = analyze_frequency(df)
    simulate_and_test_with_evolution(df, hot, warm, cold, main_freq, powerball_freq, num_games=5, save_csv=False)
    print("\nSimulation completed.")

if __name__ == "__main__":
    main()
