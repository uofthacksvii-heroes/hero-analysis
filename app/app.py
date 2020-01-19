from flask import Flask, request, jsonify, render_template, redirect
import pickle
import requests

app = Flask(__name__)
model_logistic_regression = pickle.load(open('app/model_logistic_regression.pkl', 'rb'))
model_random_forest = pickle.load(open('app/model_random_forest.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('/home/index.html')

# Logistic Regression
@app.route('/predict/one',methods=['POST'])
def predict_one():
    data = request.get_json(force=True)
    no_aed = data["from_victim"]["distance"]
    aed = data["from_total"]["distance"]

    prob_no_aed = model_logistic_regression.predict_proba([[no_aed, 0]])[0][1]
    prob_aed = model_logistic_regression.predict_proba([[aed, 1]])[0][1]

    output = {
        "no aed": prob_no_aed,
        "aed": prob_aed,
    }

    # Testing the ML model using Postman
    return render_template('/home/index.html', message="Status: POST SUCCESS",
                           message_1_a="Responder goes straight to victim ",
                           message_1_b="Survival rate: " + str(round(prob_no_aed * 100, 2)) + "%",
                           message_2_a="Responder goes to AED, and then to victim ",
                           message_2_b="Survival rate: " + str(round(prob_aed * 100, 2)) + "%"
                           )

    # POST status to Hero Command (Andrea)
    status = requests.post("http://4e1b7b99.ngrok.io/percentage", output).status_code

    if status == 200:
        return render_template('/home/index.html', message="Status: POST SUCCESS",
                               message_1_a="Responder goes straight to victim ",
                               message_1_b="Survival rate: " + str(round(prob_no_aed * 100, 2)) + "%",
                               message_2_a="Responder goes to AED, and then to victim ",
                               message_2_b="Survival rate: " + str(round(prob_aed * 100, 2)) + "%"
                               )
    else:
        return render_template('/home/index.html', message="Status: POST FAIL")

# Random Forest
@app.route('/predict/multiple',methods=['POST'])
def predict_multiple():
    data = request.get_json(force=True)

    prob_person1_to_victim = model_random_forest.predict_proba([[200, 400, 200, 1000, 1, 2]])[0][1]
    prob_person2_to_victim = model_random_forest.predict_proba([[200, 400, 200, 1000, 2, 1]])[0][1]

    output = {
        "person1_to_victim": round(prob_person1_to_victim * 100, 2),
        "person2_to_victim": round(prob_person2_to_victim * 100, 2),
    }

    # # Testing the ML model using Postman
    # return render_template('/home/index.html', message="Status: POST SUCCESS",
    #                        message_1_a="Responder goes straight to victim ",
    #                        message_1_b="Survival rate: " + str(round(prob_no_aed * 100, 2)) + "%",
    #                        message_2_a="Responder goes to AED, and then to victim ",
    #                        message_2_b="Survival rate: " + str(round(prob_aed * 100, 2)) + "%"
    #                        )

    # POST status to Hero Command (Andrea)
    status = requests.post("http://4e1b7b99.ngrok.io/percentage", output).status_code
    if status == 200:
        return render_template('/home/index.html', message="Status: POST SUCCESS",
                               message_1_a="Responder 1 goes straight to victim. Responder 2 goes to AED, and then to victim",
                               message_1_b="Survival rate: " + str(round(prob_person1_to_victim * 100, 2)) + "%",
                               message_2_a="Responder 2 goes straight to victim. Responder 1 goes to AED, and then to victim",
                               message_2_b="Survival rate: " + str(round(prob_person2_to_victim * 100, 2)) + "%"
                               )
    else:
        return render_template('/home/index.html', message="POST FAIL")


if __name__ == "__main__":
    app.run(debug=True)