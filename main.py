import flask
import pickle


with open(f"model/Stress.pkl", "rb") as f:
    model = pickle.load(f)

with open(f'model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))

    if flask.request.method == 'POST':
        inputtext = flask.request.form.get("inputtext")
        inputtext1=vectorizer.transform([inputtext]).toarray()
        prediction = model.predict(inputtext1)
        if prediction == ['Unstressed']:
            prediction = "Seems you have a good day you are unstressed"
        else:
            prediction = "Take a break you are stressed"
            

        prediction_text = "Hi:"
        
        return(flask.render_template('index.html',  prediction_text=prediction_text,  result=prediction))

if __name__ == '__main__':
    app.run(debug=True)

