import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Define attack types
NORMAL = 0
SIDE_CHANNEL = 1
FAULT_INJECTION = 2
BOTH_ATTACKS = 3

# Simulate multi-attack characteristics
def simulate_attack(round_num, stage):
    # Randomly determine if an attack occurs
    attack_probability = random.random()
    if attack_probability < 0.3:  # 30% chance of attack
        if random.random() < 0.5:
            return SIDE_CHANNEL  # Single side-channel attack
        else:
            return FAULT_INJECTION  # Single fault injection attack
    elif 0.3 <= attack_probability < 0.4:  # 10% chance of combined attacks
        return BOTH_ATTACKS
    return NORMAL  # No attack (60% chance)

# ML model to classify attacks
def train_attack_detection_model():
    # Mock data: Power profiles and labels
    features = np.random.rand(100, 10)  # 100 samples, 10 features
    labels = [NORMAL] * 50 + [SIDE_CHANNEL] * 25 + [FAULT_INJECTION] * 25
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)
    return model

attack_model = train_attack_detection_model()

# Function for chaotic clocking
def chaotic_clocking(attack_type, stage, round_num):
    if attack_type == SIDE_CHANNEL:
        print(f"Chaotic clock introduced for Side-channel attack in {stage} of round {round_num}.")
        return 100 + random.uniform(-5, 5)  # Small chaotic variations
    elif attack_type == FAULT_INJECTION:
        print(f"Chaotic clock introduced for Fault Injection attack in {stage} of round {round_num}.")
        return 100 + random.uniform(-20, 20)  # Large chaotic variations
    elif attack_type == BOTH_ATTACKS:
        print(f"Chaotic clock introduced for Combined attacks in {stage} of round {round_num}.")
        return 100 + random.uniform(-30, 30)  # Extreme chaotic variations
    else:
        return 100  # Normal frequency

# AES Functions
def sub_bytes(state, round_num):
    print("Applying SubBytes...")
    attack_type = simulate_attack(round_num, "SubBytes")
    clock_freq = chaotic_clocking(attack_type, "SubBytes", round_num)
    if attack_type in [SIDE_CHANNEL, BOTH_ATTACKS]:
        state = side_channel_attack(state)
    if attack_type in [FAULT_INJECTION, BOTH_ATTACKS]:
        state = fault_injection_attack(state)
    return (state + 1) % 256, clock_freq

def shift_rows(state, round_num):
    print("Applying ShiftRows...")
    return np.roll(state, -1, axis=1), 100  # No clock adjustment here

def mix_columns(state, round_num):
    print("Applying MixColumns...")
    attack_type = simulate_attack(round_num, "MixColumns")
    clock_freq = chaotic_clocking(attack_type, "MixColumns", round_num)
    if attack_type in [SIDE_CHANNEL, BOTH_ATTACKS]:
        state = side_channel_attack(state)
    if attack_type in [FAULT_INJECTION, BOTH_ATTACKS]:
        state = fault_injection_attack(state)
    return (state * 2) % 256, clock_freq

def add_round_key(state, round_key, round_num):
    print("Applying AddRoundKey...")
    attack_type = simulate_attack(round_num, "AddRoundKey")
    clock_freq = chaotic_clocking(attack_type, "AddRoundKey", round_num)
    if attack_type in [SIDE_CHANNEL, BOTH_ATTACKS]:
        state = side_channel_attack(state)
    if attack_type in [FAULT_INJECTION, BOTH_ATTACKS]:
        state = fault_injection_attack(state)
    return state ^ round_key, clock_freq

# Simulate side-channel and fault injection attacks
def side_channel_attack(state):
    print("Simulating Side-Channel Attack...")
    return np.mod(state + random.randint(0, 5), 256)

def fault_injection_attack(state):
    print("Simulating Fault Injection Attack...")
    fault_indices = random.sample(range(state.size), k=random.randint(1, 3))
    state = state.flatten()
    for idx in fault_indices:
        state[idx] = random.randint(0, 255)  # Inject random fault
    return state.reshape(4, 4)

# AES Encryption Process
def aes_encrypt(input_data, key):
    rounds = 10
    state = input_data.copy()
    results = []

    for round_num in range(1, rounds + 1):
        print(f"--- Round {round_num} ---")
        print("Initial State Matrix:")
        print(state)

        # SubBytes
        state, clock_freq_subbytes = sub_bytes(state, round_num)
        print("After SubBytes:")
        print(state)
        print(f"Clock frequency after SubBytes: {clock_freq_subbytes} MHz")

        # ShiftRows
        state, clock_freq_shiftrows = shift_rows(state, round_num)
        print("After ShiftRows:")
        print(state)
        print(f"Clock frequency after ShiftRows: {clock_freq_shiftrows} MHz")

        # MixColumns (except in the final round)
        if round_num < 10:
            state, clock_freq_mixcolumns = mix_columns(state, round_num)
            print("After MixColumns:")
            print(state)
            print(f"Clock frequency after MixColumns: {clock_freq_mixcolumns} MHz")

        # AddRoundKey
        round_key = np.random.randint(0, 256, (4, 4))  # Example round key
        state, clock_freq_addroundkey = add_round_key(state, round_key, round_num)
        print("After AddRoundKey:")
        print(state)
        print(f"Clock frequency after AddRoundKey: {clock_freq_addroundkey} MHz")

        # Store results for plotting
        results.append({
            "round": round_num,
            "state": state,
            "clock_frequencies": {
                "SubBytes": clock_freq_subbytes,
                "ShiftRows": clock_freq_shiftrows,
                "MixColumns": clock_freq_mixcolumns if round_num < 10 else 100,
                "AddRoundKey": clock_freq_addroundkey,
            }
        })

    return results

# Example usage
input_data = np.random.randint(0, 256, (4, 4))
key = np.random.randint(0, 256, (4, 4))
results = aes_encrypt(input_data, key)

# Plot clocking changes
rounds = [res["round"] for res in results]
subbytes_freq = [res["clock_frequencies"]["SubBytes"] for res in results]
mixcolumns_freq = [res["clock_frequencies"]["MixColumns"] for res in results]
addroundkey_freq = [res["clock_frequencies"]["AddRoundKey"] for res in results]

plt.figure(figsize=(10, 6))
plt.plot(rounds, subbytes_freq, marker='o', label='SubBytes')
plt.plot(rounds, mixcolumns_freq, marker='o', label='MixColumns')
plt.plot(rounds, addroundkey_freq, marker='o', label='AddRoundKey')
plt.axhline(100, color='gray', linestyle='--', label='Periodic Clock (100 MHz)')
plt.xlabel("Round Number")
plt.ylabel("Clock Frequency (MHz)")
plt.title("Chaotic Clocking Across AES Encryption Stages")
plt.legend()
plt.show()