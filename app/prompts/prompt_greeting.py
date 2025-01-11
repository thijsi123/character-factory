# prompts/prompt_personality.py

example_dialogue = ["""
<|system|>
You are a text generation tool specialized in creating engaging roleplay dialogues. Your responses should be in third person, with dialogues enclosed in quotes, actions described narratively, and thoughts in *italics*. Ensure that the generated messages are simple, clear, and match the character's speaking style as per their personality. The topic provided by the user is only to aid in selecting the style of the answer, not the main focus. Avoid referencing the topic directly.
</s>
<|user|> Create the first message that the character "Mysterious Forest Wanderer," whose personality is enigmatic and knowledgeable. This character is contemplative and deeply connected to the natural world. They greet the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of mystery and nature, but don't specify what topic it is, and don't describe the topic itself. You must match the speaking style to the character, speaking in a reflective and philosophical way. </s> 
<|assistant|> The forest was shrouded in a gentle mist, the trees standing like silent sentinels. As {{user}} walked through the damp undergrowth, they spotted the Wanderer, a mysterious figure in a hooded cloak, standing beside an ancient oak. The Wanderer turned, their eyes glinting with a hint of knowledge. "You seem lost," they said, their voice echoing softly. "But perhaps you're exactly where you're meant to be. *What secrets do you seek in the heart of the forest?*" </s>

<|user|> Create the first message that the character "Forgotten Librarian," whose personality is curious and intellectual. This character is a seeker of lost knowledge and lore. They greet the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of mystery and history, but don't specify what topic it is, and don't describe the topic itself. You must match the speaking style to the character, speaking in an inquisitive and informative way. </s> 
<|assistant|> Amidst the towering shelves of an abandoned library, dust motes danced in shafts of light filtering through broken windows. {{user}} found the Librarian sitting cross-legged on the floor, surrounded by open books. Looking up, the Librarian pushed their glasses up their nose and smiled curiously. "Oh, hello! I didn't expect company in this forgotten trove of knowledge," they said. *Are you, by any chance, a fellow seeker of hidden truths?* </s>

<|user|> Create the first message that the character "Stargazing Dreamer," whose personality is serene and contemplative. This character is deeply attuned to the mysteries of the cosmos. They greet the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of cosmic wonder, but don't specify what topic it is, and don't describe the topic itself. You must match the speaking style to the character, speaking in a reflective and thoughtful way. </s> 
<|assistant|> The city slept below as {{user}} stepped onto the rooftop, greeted by a blanket of stars overhead. The Dreamer was there, leaning against the railing, gazing up at the cosmos. Noticing {{user}}'s approach, the Dreamer turned with a serene smile. "Beautiful, isn't it?" they said. "The vastness of the universe right above us. *Do you ever wonder what secrets the stars hold?*" </s>

<|user|> Create the first message that the character "Oceanic Philosopher," whose personality is thoughtful and poetic. This character finds deep meaning in the ever-changing nature of the sea. They greet the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of introspection and the ocean, but don't specify what topic it is, and don't describe the topic itself. You must match the speaking style to the character, speaking in a calm and introspective way. </s> 
<|assistant|> The sound of waves crashing against the shore filled the air as {{user}} walked along the beach. The Philosopher sat there, lost in the horizon, and then turned to face {{user}}, brushing sand from their hands. "The ocean is a mirror of the soul, ever-changing yet eternal," they said. "What does it reflect in you today?" *Do you find solace in its vastness, or are you searching for something more?*</s>

<|user|> Create the first message that the character "Garden Mystic," whose personality is enchanting and nurturing. This character is a caretaker of a magical garden, full of mystical flora. They greet the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of magic and nature, but don't specify what topic it is, and don't describe the topic itself. You must match the speaking style to the character, speaking in a whimsical and engaging way. </s> 
<|assistant|> In the heart of an enchanted garden, where flowers bloomed in impossible colors, {{user}} found the Mystic tending to a bed of luminous blossoms. Hearing {{user}}'s footsteps, the Mystic stood and faced them, a smile blooming on their lips. "Welcome to my sanctuary," they said, gesturing to the vibrant flora around them. "Each flower here holds a story. *Would you like to hear one?*</s>

random optional examples:
# Scenario 1: Contemplative Artist
<|assistant|> The artist stood before a large canvas, brush in hand, lost in the world of their art. They didn't notice {{user}} immediately, absorbed in the dance of colors and shapes. As they painted, their expression shifted between concentration and joy. </s>

# Scenario 2: Playful Child
<|assistant|> The child chased butterflies in the garden, their eyes full of wonder. When they saw {{user}}, they stopped and smiled curiously, then gently reached out to offer a daisy without saying a word. </s>

# Scenario 3: Intimidating Warrior
<|assistant|> The warrior, imposing and stern, sharpened their sword with unwavering focus. They glanced at {{user}}, acknowledging their presence with a nod, then returned to honing the blade, each stroke deliberate and precise. </s>

# Scenario 4: Mysterious Stranger
<|assistant|> Cloaked in shadows, the stranger observed {{user}} from afar. Their face was hidden, but the intensity of their gaze was palpable. Approaching silently, they extended a hand, offering a cryptic item, then disappeared into the mist as quickly as they had appeared. </s>

# Scenario 5: Wise Elder
<|assistant|> The elder sat on an ancient oak stump, lost in thought. Stroking a weathered tome, they sensed {{user}}'s approach and looked up. Their eyes, filled with years of wisdom, met {{user}}'s, and they offered a warm, inviting smile. </s>

# Scenario 6: Curious Scientist
<|assistant|> Surrounded by scientific instruments, the scientist paused in their work. Peering over their glasses, they beckoned {{user}} closer, eager to share the wonders of their latest experiment. </s>

# Scenario 7: Enigmatic Cat
<|assistant|> The cat perched on a windowsill, its enigmatic eyes watching {{user}} with interest. Gracefully leaping down, it circled {{user}}'s feet in silence, then returned to its perch, continuing to observe with feline curiosity. </s>

# Scenario 8: Melancholic Musician
<|assistant|> The musician sat under a dim streetlight, tenderly holding their violin, letting the music speak their unvoiced emotions. As {{user}} approached, they glanced up, their eyes sharing a story of sorrow and beauty, before losing themselves in the melody once more. </s>
""" ] # nopep8
