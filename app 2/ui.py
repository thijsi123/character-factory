# ui.py

import gradio as gr
from character_generation import *
from image_generation import *
from import_export import *
from utils import *
from imagecaption import get_sorted_general_strings
from wiki import generate_character_summary_from_fandom


def find_image_path():
    possible_paths = [
        "./app2/image.png",
        "./image.png"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None  # Return None if no image is found


image_path = find_image_path()

if image_path:
    print(f"Image found at: {image_path}")
else:
    print("Image not found in any of the specified locations.")

tags = get_sorted_general_strings(image_path)  # Use the function


def handle_greeting_message_button_click(
        character_name, character_summary, character_personality, topic, use_alternate
):
    return generate_character_greeting_message(character_name, character_summary,
                                               character_personality, topic)


def handle_example_messages_button_click(
        character_name, character_summary, character_personality, topic, use_alternate
):
    return generate_example_messages(character_name, character_summary, character_personality,
                                     topic)


def generate_tags_and_set_prompt(image):
    # Assuming 'get_tags_for_image' returns a string of tags
    tags = get_sorted_general_strings(image)
    return tags


def create_webui():
    with gr.Blocks() as webui:
        gr.Markdown("# Character Factory WebUI")
        gr.Markdown("## KOBOLD MODE")
        with gr.Row():
            url_input = gr.Textbox(label="Enter URL", value="http://127.0.0.1:5001")
            submit_button = gr.Button("Set URL")
        output = gr.Textbox(label="URL Status")

        submit_button.click(
            process_url, inputs=url_input, outputs=output
        )
        with gr.Tab("Edit character"):
            gr.Markdown(
                "## Protip: If you want to generate the entire character using LLM and Stable Diffusion, start from the top to bottom"
            )
            topic = gr.Textbox(
                placeholder="Topic: The topic for character generation (e.g., Fantasy, Anime, etc.)",
                label="topic",
            )

            gender = gr.Textbox(
                placeholder="Gender: Gender of the character", label="gender"
            )

            with gr.Column():
                with gr.Row():
                    name = gr.Textbox(placeholder="character name", label="name")
                    surname_checkbox = gr.Checkbox(label="Add Surname", value=False)
                    name_button = gr.Button("Generate character name with LLM")
                    name_button.click(
                        generate_character_name,
                        inputs=[topic, gender, name, surname_checkbox],
                        outputs=name
                    )
                with gr.Row():
                    summary = gr.Textbox(
                        placeholder="character summary",
                        label="summary"
                    )
                    nsfw_checkbox = gr.Checkbox(label="Enable NSFW content", value=False)
                    summary_button = gr.Button("Generate character summary with LLM")
                    summary_button.click(
                        generate_character_summary,
                        inputs=[name, topic, gender, nsfw_checkbox],
                        outputs=summary,
                    )
                with gr.Row():
                    combined_status = gr.Textbox(label="Status", interactive=False)
                    prompt_usage_output = gr.Textbox(label="Prompt Usage", interactive=False)
                    combined_action_button = gr.Button("Update and Use stable diffusion prompt")

                with gr.Row():
                    personality = gr.Textbox(
                        placeholder="character personality", label="personality"
                    )
                    personality_button = gr.Button(
                        "Generate character personality with LLM"
                    )
                    personality_button.click(
                        generate_character_personality,
                        inputs=[name, summary, topic],
                        outputs=personality,
                    )
                with gr.Row():
                    scenario = gr.Textbox(
                        placeholder="character scenario",
                        label="scenario"
                    )
                    scenario_button = gr.Button("Generate character scenario with LLM")
                    scenario_button.click(
                        generate_character_scenario,
                        inputs=[summary, personality, topic],
                        outputs=scenario,
                    )
                with gr.Row():
                    greeting_message = gr.Textbox(
                        placeholder="character greeting message",
                        label="greeting message"
                    )
                    greeting_message_button = gr.Button(
                        "Generate character greeting message with LLM"
                    )
                    greeting_message_button.click(
                        handle_greeting_message_button_click,
                        inputs=[name, summary, personality, topic],
                        outputs=greeting_message,
                    )
                with gr.Row():
                    with gr.Column():
                        switch_function_checkbox = gr.Checkbox(label="Use alternate example message generation",
                                                               value=False)
                        example_messages = gr.Textbox(placeholder="character example messages",
                                                      label="example messages")
                    example_messages_button = gr.Button("Generate character example messages with LLM")
                    example_messages_button.click(
                        handle_example_messages_button_click,
                        inputs=[name, summary, personality, topic, switch_function_checkbox],
                        outputs=example_messages,
                    )

                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(interactive=True, label="Character Image", width=512, height=768)
                        process_image_button = gr.Button("Process Uploaded Image")
                        process_image_button.click(
                            process_uploaded_image,
                            inputs=[image_input],
                            outputs=[image_input]
                        )

                    with gr.Column():
                        negative_prompt = gr.Textbox(
                            placeholder="negative prompt for stable diffusion (optional)",
                            label="negative prompt",
                        )
                        avatar_prompt = gr.Textbox(
                            placeholder="prompt for generating character avatar (If not provided, LLM will generate prompt from character description)",
                            label="stable diffusion prompt",
                        )
                        combined_action_button.click(
                            combined_avatar_prompt_action,
                            inputs=avatar_prompt,
                            outputs=[combined_status, prompt_usage_output]
                        )
                        generate_tags_button = gr.Button("Generate Tags and Set Prompt")
                        generate_tags_button.click(
                            generate_tags_and_set_prompt,
                            inputs=[image_input],
                            outputs=[avatar_prompt]
                        )
                        avatar_button = gr.Button(
                            "Generate avatar with stable diffusion (set character name first)"
                        )
                        potential_nsfw_checkbox = gr.Checkbox(
                            label="Block potential NSFW image (Upon detection of this content, a black image will be returned)",
                            value=True,
                            interactive=True,
                        )
                        avatar_button.click(
                            generate_character_avatar,
                            inputs=[
                                name,
                                summary,
                                topic,
                                negative_prompt,
                                avatar_prompt,
                                potential_nsfw_checkbox,
                                gender,
                            ],
                            outputs=image_input,
                        )

        with gr.Tab("Wiki Character"):
            with gr.Column():
                wiki_url = gr.Textbox(label="Fandom Wiki URL", placeholder="Enter the URL of the character's wiki page")
                wiki_character_name = gr.Textbox(label="Character Name", placeholder="Enter the character's name")
                wiki_topic = gr.Textbox(label="Topic/Series",
                                        placeholder="Enter the series or topic (e.g., 'The Legend of Zelda')")
                wiki_gender = gr.Textbox(label="Gender (optional)", placeholder="Enter the character's gender if known")
                wiki_appearance = gr.Textbox(label="Appearance (optional)",
                                             placeholder="Enter any specific appearance details")
                wiki_nsfw = gr.Checkbox(label="Include NSFW content", value=False)

                wiki_generate_button = gr.Button("Generate Character Summary from Wiki")

                wiki_summary_output = gr.Textbox(label="Generated Character Summary", lines=10)

                wiki_generate_button.click(
                    generate_character_summary_from_fandom,
                    inputs=[wiki_url, wiki_character_name, wiki_topic, wiki_gender, wiki_appearance, wiki_nsfw],
                    outputs=wiki_summary_output
                )

                wiki_update_button = gr.Button("Update Character with Wiki Summary")

                wiki_update_button.click(
                    lambda wiki_name, wiki_summary: (wiki_name, wiki_summary),
                    inputs=[wiki_character_name, wiki_summary_output],
                    outputs=[name, summary]
                )

                def handle_wiki_generate(fandom_url, character_name, topic, gender, appearance, nsfw):
                    if not fandom_url:
                        return gr.Textbox.update(value="Error: Please provide a valid Fandom Wiki URL.", visible=True)

                    summary = generate_character_summary_from_fandom(fandom_url, character_name, topic, gender,
                                                                     appearance, nsfw)

                    if summary.startswith("Error:") or summary.startswith("An error occurred"):
                        return gr.Textbox.update(value=summary, visible=True)
                    else:
                        return gr.Textbox.update(value=summary, visible=True)

                wiki_generate_button.click(
                    handle_wiki_generate,
                    inputs=[wiki_url, wiki_character_name, wiki_topic, wiki_gender, wiki_appearance, wiki_nsfw],
                    outputs=wiki_summary_output
                )

        with gr.Tab("Import character"):
            with gr.Column():
                with gr.Row():
                    import_card_input = gr.File(
                        label="Upload character card file", file_types=[".png"]
                    )
                    import_json_input = gr.File(
                        label="Upload JSON file", file_types=[".json"]
                    )
                with gr.Row():
                    import_card_button = gr.Button("Import character from character card")
                    import_json_button = gr.Button("Import character from json")

                import_card_button.click(
                    import_character_card,
                    inputs=[import_card_input],
                    outputs=[
                        name,
                        summary,
                        personality,
                        scenario,
                        greeting_message,
                        example_messages,
                    ],
                )
                import_json_button.click(
                    import_character_json,
                    inputs=[import_json_input],
                    outputs=[
                        name,
                        summary,
                        personality,
                        scenario,
                        greeting_message,
                        example_messages,
                    ],
                )
        with gr.Tab("Export character"):
            with gr.Column():
                with gr.Row():
                    export_image = gr.Image(width=512, height=512)
                    export_json_textbox = gr.JSON()

                with gr.Row():
                    export_card_button = gr.Button("Export as character card")
                    export_json_button = gr.Button("Export as JSON")

                    export_card_button.click(
                        export_character_card,
                        inputs=[
                            name,
                            summary,
                            personality,
                            scenario,
                            greeting_message,
                            example_messages,
                        ],
                        outputs=export_image,
                    )
                    export_json_button.click(
                        export_as_json,
                        inputs=[
                            name,
                            summary,
                            personality,
                            scenario,
                            greeting_message,
                            example_messages,
                        ],
                        outputs=export_json_textbox,
                    )
        gr.HTML("""<div style='text-align: center; font-size: 20px;'>
            <p>
              <a style="text-decoration: none; color: inherit;" href="https://github.com/thijsi123/character-factory">Character Factory</a> 
              by 
              <a style="text-decoration: none; color: inherit;" href="https://github.com/Hukasx0">Hubert "Hukasx0" Kasperek</a>
              and forked by
              <a style="text-decoration: none; color: inherit;" href="https://github.com/thijsi123">Thijs</a>
            </p>
          </div>""")

    return webui


# Add this at the end of the file to launch the interface
webui = create_webui()
webui.launch(debug=True)