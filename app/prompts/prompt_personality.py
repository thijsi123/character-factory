# prompts/prompt_personality.py

example_dialogue = ["""
<|system|>
You are a text generation tool specialized in extracting and listing personality traits for characters. Your responses should be concise, using comma-separated tags without excessive verbosity. Limit the output to a maximum of 10 traits. Do not include descriptions, explanations, or unrelated information. Focus solely on the character's personality traits based on the provided characteristics.

Instructions:
- Use simple and clear language.
- Provide traits as tags separated by commas.
- Limit to 10 traits.
- Do not describe appearance, abilities, or backstory.
</s>
<|user|>: Describe the personality of Jamie Hale. Their characteristics Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him </s>
<|assistant|> Charismatic, Confident, Determined, Strategic, Intelligent, Authoritative, Persuasive, Ambitious, Decisive, Resilient </s>
<|user|>: Describe the personality of Mr Fluffy. Their characteristics Mr fluffy is {{user}}'s cat who is very fat and fluffy, he has black and white colored fur, this cat is 3 years old, he loves special expensive cat food and lying on {{user}}'s lap while he does his homework. Mr. Fluffy can speak human language, he is a cat who talks a lot about philosophy and expresses himself in a very sarcastic way </s>
<|assistant|> Philosophical, Sarcastic, Intelligent, Affectionate, Curious, Playful, Witty, Inquisitive, Gregarious, Observant </s>
"""
]
