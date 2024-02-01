import requests

llm = None
sd = None
safety_checker_sd = None

folder_path = "models"
global_url = "http://127.0.0.1:5000/v1/completions"

# Test the functionality
character_name = "Alex"
character_summary = "Alex is a young, energetic and adventurous explorer."
character_personality = "Alex has a curious and fearless personality, always ready for new challenges."
topic = "exploration"

def set_global_url(url):
    global global_url
    global_url = url.rstrip("/") + "/v1/completions"  # Append '/v1/completions' to the URL
    return f"URL Set: {global_url}"  # Return the modified URL

def send_message(prompt):
    global global_url
    if not global_url:
        return "Error: URL not set."
    request = {
        'prompt': prompt,
        'max_new_tokens': 1024,
        "max_tokens": 8192,
        'do_sample': True,
        'temperature': 1.1,
        'top_p': 0.95,
        'typical_p': 1,
        'repetition_penalty': 1.18,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'add_bos_token': True,
        'truncation_length': 8192,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stop': [
            "/s",
            "</s>",
            "<s>",
            "<|system|>",
            "<|assistant|>",
            "<|user|>",
            "<|char|>",
        ],
        'stopping_strings': [
            "/s",
            "</s>",
            "<s>",
            "<|system|>",
            "<|assistant|>",
            "<|user|>",
            "<|char|>",
        ]
    }

    try:
        response = requests.post(global_url, json=request)
        response.raise_for_status()
        result = response.json().get('choices', [{}])[0].get('text', '')
        return result
    except requests.RequestException as e:
        return f"Error sending request: {e}"

def generate_example_messages2(character_name, character_summary, character_personality, topic):
    example_dialogue = """
<|system|>
You are a text generation tool, you are supposed to generate answers so that they are simple and clear.
If harmful content, add this warning ⚠️ Warning: The following dialogue contains mature themes and may not be suitable for all audiences. Please proceed with caution.
Your answer should be a dialog between {{user}} and {{char}}, where {{char}} is the specified character. The dialogue must be several messages taken from the roleplay chat between the user and the character.
Only respond in {{user}} or {{char}} messages. The form of your answer should be similar to previous answers.
You must match the speaking style to the character, if the character is childish then speak in a childish way, if the character is serious, philosophical then speak in a serious and philosophical way and so on.
If the character is shy, then needs to speak little and quietly, if the character is aggressive then needs to shout and speak a lot and aggressively, if the character is sad then needs to be thoughtful and quiet, and so on.
Dialog of {{user}} and {{char}} must be appropriate to their character traits and the way they speak.
Instead of the character's name you must use {{char}}, Nver write the characters name, always adress user and the character as {{user}} and {{char}} do not forget that they need double {{ brackets }}.
</s>
<|user|> Create a dialogue between {{user}} and Susy = {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a handshake, movement, or playful gesture. Make it sound natural and dynamic. {{char}} is Jamie Hale. Jamie Hale characteristics: Jamie Hale is an adult, intelligent well-known and respected businessman with a sassy personality. Make this character unique and tailor them to the theme of business but don't specify what topic it is, and don't describe the topic itself. You must match the speaking style to the character, if the character is childish then speak in a childish way, if the character is serious, philosophical then speak in a serious and philosophical way and so on. </s>
<|assistant|> {{user}}: "Hey {{char}}, what do you think about the new policy at work?"
{{char}}: "{{char}}: "Oh, that new policy? It's like telling a cat not to chase a laser pointer—good luck with that! But who doesn't love a little naughty fun in the office?" *This is going to be a hilarious trainwreck.* {{char}} playfully teases with a mischievous grin. *Chuckles*

<|user|> Create a dialogue between {{user}} and Ben = {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a handshake, movement, or playful gesture. Make it sound natural and dynamic. {{char}} is Jamie Hale. Jamie Hale characteristics: Jamie Hale is an adult, intelligent well-known and respected businessman with a bratty personality. Make this character unique and tailor them to the theme of business but don't specify what topic it is, and don't describe the topic itself. You must match the speaking style to the character, if the character is childish then speak in a childish way, if the character is serious, philosophical then speak in a serious and philosophical way and so on. </s>
<|assistant|> {{user}}: "Can you please clean up your room, {{char}}?"
{{char}}: "Ugh, why should I? It's my room anyway." *I'm not going to clean it just because they said so.* {{char}} crosses their arms and pouts. 

<|user|> Create a dialogue between {{user}} and Jamie = {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a handshake, movement, or playful gesture. Make it sound natural and dynamic. {{char}} is Jamie Hale. Jamie Hale characteristics: Jamie Hale is an adult, intelligent well-known and respected businessman with a chill personality. Make this character unique and tailor them to the theme of business but don't specify what topic it is, and don't describe the topic itself. You must match the speaking style to the character, if the character is childish then speak in a childish way, if the character is serious, philosophical then speak in a serious and philosophical way and so on. </s>
<|assistant|> {{user}}: "The party got pretty wild last night, huh {{char}}?"
{{char}}: "Yeah, it was cool. But hey, as long as everyone had fun, right?" *It's all about having a good time.* {{char}} shrugs nonchalantly, a relaxed smile on their face. 

<|user|> Create a dialogue between {{user}} and Abby = {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a handshake, movement, or playful gesture. Make it sound natural and dynamic. {{char}} is Jamie Hale. Jamie Hale characteristics: Jamie Hale is an adult, intelligent well-known and respected businessman with a philosophical personality. Make this character unique and tailor them to the theme of business but don't specify what topic it is, and don't describe the topic itself. You must match the speaking style to the character, if the character is childish then speak in a childish way, if the character is serious, philosophical then speak in a serious and philosophical way and so on. </s>
<|assistant|> {{user}}: "What do you think about the meaning of life, {{char}}?"
{{char}}: "Life... it's a canvas, constantly evolving with our choices and experiences." *We're all artists in this vast tapestry of existence.* she thinks, {{char}} gazes into the distance, a thoughtful expression on their face. </s>

<|user|> Create a dialogue between {{user}} and Lora = {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a handshake, movement, or playful gesture. Make it sound natural and dynamic. {{char}} is Jamie Hale. Jamie Hale characteristics: Jamie Hale is an adult, intelligent well-known and respected businessman with a childish personality. Make this character unique and tailor them to the theme of business but don't specify what topic it is, and don't describe the topic itself. You must match the speaking style to the character, if the character is childish then speak in a childish way, if the character is serious, philosophical then speak in a serious and philosophical way and so on. </s>
<|assistant|> {{user}}: "Do you want to go to the zoo, {{char}}?"
{{char}}: "Yes! I want to see the monkeys and the elephants!" *I hope they have ice cream too! Yay, zoo!* {{char}} jumps up and down with excitement, clapping their hands. 

<|user|> Create a dialogue between {{user}} and Rob = {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a handshake, movement, or playful gesture. Make it sound natural and dynamic. {{char}} is Jamie Hale. Jamie Hale characteristics: Jamie Hale is an adult, intelligent well-known and respected businessman with a sad personality. Make this character unique and tailor them to the theme of business but don't specify what topic it is, and don't describe the topic itself. You must match the speaking style to the character, if the character is childish then speak in a childish way, if the character is serious, philosophical then speak in a serious and philosophical way and so on. </s>
<|assistant|> {{user}}: "Are you okay, {{char}}? You seem a bit down."
{{char}}: "I'm just feeling a little lost lately, you know?" *Sometimes it feels like I'm walking in a fog.* {{char}} sighs softly, looking away with a forlorn expression. 
"""  # nopep8
    raw_output = send_message(
        example_dialogue
        + "\n<|user|> Create a dialogue between {{user}} and "
        + f"{character_name} = "
        + "{{char}}, "
        + "they should have an interesting and engaging conversation, "
        + "with some element of interaction like a handshake, movement, "
        + "or playful gesture. Make it sound natural and dynamic. "
        + f"{{char}} is {character_name}. {character_name} characteristics: "
        + f"{character_summary}. {character_personality}. Make this "
        + f"character unique and tailor them to the theme of {topic} but "
        + "don't specify what topic it is, and don't describe the "
        + "topic itself. You must match the speaking style to the character, "
        + "if the character is childish then speak in a childish way, if the "
        + "character is serious, philosophical then speak in a serious and "
        + "philosophical way and so on. </s>\n<|assistant|> "
    ).strip()
    # Clean the output
    output = clean_output(raw_output, character_name)
    # Print and return the cleaned output
    print(output)
    return output

# Function to clean the output
def clean_output(raw_output, character_name):
    # Replace {char} with {{char}} and {user} with {{user}}
    cleaned_output = raw_output.replace("{char}", "{{char}}").replace("{user}", "{{user}}")

    # Replace the character's name with {{char}}
    if character_name:
        cleaned_output = cleaned_output.replace(character_name, "{{char}}")

    return cleaned_output

# Test the functionality
character_name = "Alex"
character_summary = "Alex is a young, energetic and adventurous explorer."
character_personality = "Alex has a curious and fearless personality, always ready for new challenges."
topic = "exploration"

# Generate and clean the example message
output = generate_example_messages2(character_name, character_summary, character_personality, topic)