import random
import requests
import re

# Lists of name components and races
first_names = ["Kingdom of", "Empire of", "Realm of", "Dominion of", "New", "Old", "Great"]
second_names = ["Oars", "Eldoria", "Valoria", "Narnia", "Avalon", "Celestia", "Aethoria", "Zephyria", "Lumeij"]

worldbox_races = ["Human", "Elf", "Orc", "Dwarf"]
additional_races = [
    "Goblin", "Fallen Angel", "Skeletonman", "Barbarian", "Quagoa", "Succubus", "Troll", "Ogre",
    "Snow Golem", "Halfling", "Li guei", "Yeti", "Bat demon", "Mud man", "Cyclops", "Devil"
]

all_races = worldbox_races + additional_races

def clean_name(name):
    # Remove any text after a period or newline
    name = re.split('[.\n]', name)[0]
    # Remove quotes and extra whitespace
    name = re.sub(r'["\']', '', name).strip()
    return name

def generate_name_for_race(race, use_koboldcpp=True):
    if use_koboldcpp:
        try:
            input_text = (
                f"Create a unique and creative fantasy kingdom name for a {race} civilization. " +
                "The name should be 1-4 words long, be imaginative, and avoid common fantasy tropes. " +
                "Do not include the race name in the kingdom name. " +
                "Provide only the name, without any explanation or additional text:"
            )
            response = requests.post("http://localhost:5001/api/v1/generate",
                                     json={"prompt": input_text, "max_length": 30})
            response.raise_for_status()
            response_data = response.json()

            if "results" in response_data and len(response_data["results"]) > 0:
                generated_name = clean_name(response_data["results"][0]["text"])
                if generated_name:
                    return generated_name
            print(f"Warning: Unexpected or empty response from KoboldCpp for {race}")
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to connect to KoboldCpp server for {race}. {e}")

    # Fallback to random generation
    return f"{random.choice(first_names)} {random.choice(second_names)}"

def generate_kingdom_info(race):
    name = generate_name_for_race(race)
    info = {
        "name": name,
        "primary_race": race,

    }
    return info


'''"population": random.randint(10000, 1000000),
"government": random.choice(["Monarchy", "Republic", "Oligarchy", "Theocracy", "Dictatorship"]),
"terrain": random.choice(["Mountainous", "Coastal", "Forest", "Desert", "Tundra", "Grassland"]),'''

# Generate and print kingdom information for each race
for race in all_races:
    kingdom_info = generate_kingdom_info(race)
    print(f"\nKingdom: {kingdom_info['name']}")
    print(f"Primary Race: {kingdom_info['primary_race']}")
'''    print(f"Population: {kingdom_info['population']:,}")
    print(f"Government: {kingdom_info['government']}")
    print(f"Terrain: {kingdom_info['terrain']}")'''