from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session

app = Flask(__name__)

# Secret key for session management. Use a secure key in production.
app.secret_key = 'your_secret_key_here'

# Configure server-side session
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Define the steps in order
STEPS = [
    'style',
    'ethnicity',
    'age',
    'eye_color',
    'hair_style',
    'hair_color',
    'body_type',
    'breast_size',
    'butt_size',
    'voice',
    'occupation',
    'hobbies',
    'relationship',
    'clothing'
]

@app.route('/')
def index():
    # Clear any existing session data
    session.clear()
    return redirect(url_for('step', step_num=1))

@app.route('/step/<int:step_num>', methods=['GET', 'POST'])
def step(step_num):
    if step_num < 1 or step_num > len(STEPS) + 1:
        return redirect(url_for('summary'))

    if request.method == 'POST':
        # Determine if 'Previous' or 'Next' was clicked
        if 'previous' in request.form:
            return redirect(url_for('step', step_num=step_num - 1))
        elif 'next' in request.form:
            # Save the data from the current step
            step_key = STEPS[step_num - 1]
            if step_key == 'hobbies':
                selected_hobbies = request.form.getlist('hobbies')
                if len(selected_hobbies) > 3:
                    error = "You can choose up to 3 hobbies."
                    return render_template(f'step{step_num}.html', step_num=step_num, error=error, data=session)
                session[step_key] = selected_hobbies
            else:
                selected_value = request.form.get(step_key)
                if selected_value:
                    session[step_key] = selected_value
            return redirect(url_for('step', step_num=step_num + 1))

    if step_num == len(STEPS) + 1:
        return redirect(url_for('summary'))

    return render_template(f'step{step_num}.html', step_num=step_num, data=session)

@app.route('/summary')
def summary():
    return render_template('summary.html', data=session)

@app.route('/finalize', methods=['POST'])
def finalize():
    # Here you can handle the finalization process, such as saving to a database
    # For demonstration, we'll just clear the session and display a message
    session.clear()
    return "Your AI character has been created!"

if __name__ == '__main__':
    app.run(debug=True)
