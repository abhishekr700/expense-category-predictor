from flask import Flask, request, jsonify
from fastai.text.all import load_learner

learn = load_learner('models/learner.pkl')

app = Flask(__name__)

@app.route('/get_expense', methods=['GET'])
def get_expense():
    expense_name = request.args.get('expenseName')
    if expense_name is not None:
        prediction = int(learn.predict(expense_name)[0])

        # Replace this with your actual logic to fetch expense details
        expense_details = {"expenseName": expense_name, "categoryId": prediction}
        return jsonify(expense_details)
    else:
        return jsonify({"error": "Expense name not provided"}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=4001,debug=not True)
