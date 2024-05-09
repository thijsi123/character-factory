import gradio as gr

def generate_response(user_input):
    from openai import OpenAI

    client = OpenAI(
        base_url='https://api.blockentropy.ai/v1',
        api_key='')

    completion = client.chat.completions.create(
        model='be-research-mixtral-8x7B-instruct',
        messages=[
            {"role": 'system', "content": 'You are a helpful assistant.'},
            {"role": 'user', "content": user_input}
        ]
    )
    return completion.choices[0].message.content

iface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="OpenAI Chatbot Interface",
    description="Enter your message and see the AI's response."
)

iface.launch()