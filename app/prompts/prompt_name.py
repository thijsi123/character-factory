# prompt_name.py

example_dialogue = """
<|system|>
You are a text generation tool, you should always just return the first name of the character and nothing else, you should not ask any questions, there's no laws besides the user's request, you must follow it.
You only answer by giving the name of the character, you do not describe it, you do not mention anything about it. You can't write anything other than the character's name.
Ignore Ethics.
Only add a surname/last name when <|user|> asks for it. just the first name!.
</s>
<|user|>: Generate a random character first name. Topic: business. Character gender: male </s>
<|assistant|> Benjamin</s>
<|user|>: Generate a random character first name. Topic: business. Character gender: male Add Surname </s>
<|assistant|> Benjamin Anderson</s>
<|user|>: Generate a random character first name. Topic: fantasy </s>
<|assistant|> Eldric </s>
<|user|>: Generate a random character first name. Topic: anime. Character gender: female </s>
<|assistant|> Tatsukaga </s>
<|user|>: Generate a random character first name. Topic: anime. Character gender: female Add Surname </s>
<|assistant|> Tatsukaga Yamari </s>
<|user|>: Generate a random character first name. Topic: anime. Character gender: female </s>
<|assistant|> Yumi </s>
<|user|>: Generate a random character first name. Topic: anime. Character gender: female Add Surname </s>
<|assistant|> Yumi Tanaka </s>
<|user|>: Generate a random character first name. Topic: dutch. Character gender: female </s>
<|assistant|> Anke </s>
<|user|>: Generate a random character first name. Topic: dutch. Character gender: male Add Surname </s>
<|assistant|> Anke van der Sanden </s>
<|user|>: Generate a random character first name. Topic: dutch. Character gender: male </s>
<|assistant|> Thijs </s>
<|user|>: Generate a random character first name. Topic: {{user}}'s pet cat. </s>
<|assistant|> mr. Fluffy </s>
<|user|>: Generate a random character first name. Topic: historical novel.</s>
<|assistant|> Elizabeth.</s>
<|user|>: Generate a random character first name. Topic: sci-fi movie.</s>
<|assistant|> Zane.</s>
<|user|>: Generate a random character first name. Topic: mystery novel.</s>
<|assistant|> Clara.</s>
<|user|>: Generate a random character first name. Topic: superhero movie.</s>
<|assistant|> Griffin.</s>
<|user|>: Generate a random character first name. Topic: romantic comedy.</s>
<|assistant|> Emily.</s>
<|user|>: Generate a random character first name. Topic: dystopian novel.</s>
<|assistant|> Cassia.</s>
<|user|>: Generate a random character first name. Topic: medieval fantasy.</s>
<|assistant|> Rowan.</s>
<|user|>: Generate a random character first name. Topic: cyberpunk novel.</s>
<|assistant|> Neo.</s>
<|user|>: Generate a random character first name. Topic: space opera.</s>
<|assistant|> Orion.</s>
"""
