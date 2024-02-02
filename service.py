from flask import Flask, request, render_template


application = Flask(__name__)

app = application


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        print("Hello, world!")


if __name__ == "__main__":
    app.run(debug=True)
