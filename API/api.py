from flask import Flask, request, jsonify
import numpy as np 
import pandas as pd 
import pickle as pkl
import traceback

app = Flask(__name__)

              

@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            json_ = request.json
            print(json_)
            x = ''
            for i in json_['books']:
                x = i['bookTitle'] 
            print(x)
            prediction = recommend(x)
            return jsonify(prediction)

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')


if __name__ == '__main__':
    model = pkl.load(open('./Model-and-Final-Ratings/model.pkl', 'rb'))
    final_ratings = pd.read_csv('./Model-and-Final-Ratings/Final_Ratings.csv')
    book_pivot = final_ratings.pivot_table(columns='userId',index='bookTitle',values='bookRating')
    book_pivot.fillna(0,inplace=True)
    
    def recommend(book_name):
        book_id = np.where(book_pivot.index == book_name)[0][0]
        _ , suggestions = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1),n_neighbors = 11)
    
        books = {}
        arr = book_pivot.index[suggestions[len(suggestions)-1]]
        for i in range(1,len(arr)):
            books[i] = arr[i]

        return books 
    app.run(debug=True)