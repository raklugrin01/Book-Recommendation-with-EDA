<img src="https://user-images.githubusercontent.com/60045850/126037283-98579e7c-eb11-401c-8706-e0884520ab82.jpeg" alt="DisasterTweets" width="1500" height="450" />


[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/mohitnirgulkar/book-recommendation-data) [![Language](https://img.shields.io/badge/Lang-Python-brightgreen)](https://www.python.org/)  [![ML Library](https://img.shields.io/badge/ML-Scikit--learn-orange)](https://scikit-learn.org/) [![Scipy](https://img.shields.io/badge/Other-Scipy-red)](https://www.scipy.org/)

# Project Description
In this Project we analyse and preprocess the Book Crossing Dataset collected by Cai-Nicolas Ziegler
and apply Machine Learning to recommend different books from a book you previously read.
Whole code below is in Python using various libraries. Open source library Scipy is used for preprocessing and Scikit-Learn is used for creating the model.

- Total approach towards the project can be seen on kaggle

  - **Kaggle Notebook** : https://www.kaggle.com/mohitnirgulkar/book-recommendation-with-data-analysis
 
# Project Contents

  1. Exploratory Data Analysis
  2. Different ways of building Recommendation system 
  3. Model and flask Api 
  4. Refrences

# Resources Used
  
  - **Packages** : Pandas, Numpy, Matplotlib, Seaborn, Word-cloud, Scikit-Learn etc.
  - **Dataset**  : https://www.kaggle.com/mohitnirgulkar/book-recommendation-data

## 1. Exploratory Data Analysis

  - **Visualising Explicit Rating Counts (for 1-10 rating value)**

    <img src="https://www.kaggleusercontent.com/kf/68370032/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FAjg2pahzNZq-V0oRBRfIA.kCW2fMt_A8GmMQsF50T940HYuMn_PSmKjW5J9KSxk5g2fzu2FVV2txschB8b2UAzEMp44lZYiXSeEmHUj2J5Y4YYVh5Za8WmsZ6pEIQxSDaWuQ3nm2f1sL9rZHD-psZ4s99Ohm1ixz_NGaiEHEjKMacvTYtu7B06AzlrnPPj7NlJTBykxkJYFZcvAKryfyWKesENfzQ5iyFiiW92JYoHFcu_fJ0RSk6hE67GhXBA_zCNlVNZg9gbHaCr1bBYwzfE03ZP568FkhZU9MqO47edRu6ysxbcWOe2csz6rptlTntVHs58dLrH17pprHJTot4ttugrTjRPy3ra5cETBiysTLda7ozaUCNEZQXbufq-8W8Zxaay-q6nq0P-6oe6QQG7b8sMR1AzI4uWkVwrAwNPShGsoTJBfid3FWO4UnXXfBR7TXmoBO6oprzyp9cB9pT1jO_dXttOlb6rTm7zq-aTIi4xLPRKj8m9UBVP-mTbm81KocNdnght0bDOAlXXcTmae3SNEGvebyy17sfU-ESxNWqhHPjrZAwJB4EmlX8XCBkZf4smh2tX9g4IDaa8v4cDC2RY80psM2-UkEFgaihny_42wmw6cmq9-DycqTL-5McjQGQYg0M0DPvHQDZMwtpJr2iruorzvNTOITmdqIE0EN9WfYS3ycR8XKJHVwaOZWV7SZxpJenJk7QiDI19fz5n.o62nt0Ar66oYEXXKS9MKdA/__results___files/__results___75_0.png" alt="Target Variable" width="900" height="450" />

  - **Visualising top 30 most read books**

    <img src="https://www.kaggleusercontent.com/kf/68370032/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FAjg2pahzNZq-V0oRBRfIA.kCW2fMt_A8GmMQsF50T940HYuMn_PSmKjW5J9KSxk5g2fzu2FVV2txschB8b2UAzEMp44lZYiXSeEmHUj2J5Y4YYVh5Za8WmsZ6pEIQxSDaWuQ3nm2f1sL9rZHD-psZ4s99Ohm1ixz_NGaiEHEjKMacvTYtu7B06AzlrnPPj7NlJTBykxkJYFZcvAKryfyWKesENfzQ5iyFiiW92JYoHFcu_fJ0RSk6hE67GhXBA_zCNlVNZg9gbHaCr1bBYwzfE03ZP568FkhZU9MqO47edRu6ysxbcWOe2csz6rptlTntVHs58dLrH17pprHJTot4ttugrTjRPy3ra5cETBiysTLda7ozaUCNEZQXbufq-8W8Zxaay-q6nq0P-6oe6QQG7b8sMR1AzI4uWkVwrAwNPShGsoTJBfid3FWO4UnXXfBR7TXmoBO6oprzyp9cB9pT1jO_dXttOlb6rTm7zq-aTIi4xLPRKj8m9UBVP-mTbm81KocNdnght0bDOAlXXcTmae3SNEGvebyy17sfU-ESxNWqhHPjrZAwJB4EmlX8XCBkZf4smh2tX9g4IDaa8v4cDC2RY80psM2-UkEFgaihny_42wmw6cmq9-DycqTL-5McjQGQYg0M0DPvHQDZMwtpJr2iruorzvNTOITmdqIE0EN9WfYS3ycR8XKJHVwaOZWV7SZxpJenJk7QiDI19fz5n.o62nt0Ar66oYEXXKS9MKdA/__results___files/__results___86_0.png" alt="Target Variable" width="900" height="475" />

  - **Visualising top 30 most read books with there average ratings**

    <img src="https://www.kaggleusercontent.com/kf/68370032/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FAjg2pahzNZq-V0oRBRfIA.kCW2fMt_A8GmMQsF50T940HYuMn_PSmKjW5J9KSxk5g2fzu2FVV2txschB8b2UAzEMp44lZYiXSeEmHUj2J5Y4YYVh5Za8WmsZ6pEIQxSDaWuQ3nm2f1sL9rZHD-psZ4s99Ohm1ixz_NGaiEHEjKMacvTYtu7B06AzlrnPPj7NlJTBykxkJYFZcvAKryfyWKesENfzQ5iyFiiW92JYoHFcu_fJ0RSk6hE67GhXBA_zCNlVNZg9gbHaCr1bBYwzfE03ZP568FkhZU9MqO47edRu6ysxbcWOe2csz6rptlTntVHs58dLrH17pprHJTot4ttugrTjRPy3ra5cETBiysTLda7ozaUCNEZQXbufq-8W8Zxaay-q6nq0P-6oe6QQG7b8sMR1AzI4uWkVwrAwNPShGsoTJBfid3FWO4UnXXfBR7TXmoBO6oprzyp9cB9pT1jO_dXttOlb6rTm7zq-aTIi4xLPRKj8m9UBVP-mTbm81KocNdnght0bDOAlXXcTmae3SNEGvebyy17sfU-ESxNWqhHPjrZAwJB4EmlX8XCBkZf4smh2tX9g4IDaa8v4cDC2RY80psM2-UkEFgaihny_42wmw6cmq9-DycqTL-5McjQGQYg0M0DPvHQDZMwtpJr2iruorzvNTOITmdqIE0EN9WfYS3ycR8XKJHVwaOZWV7SZxpJenJk7QiDI19fz5n.o62nt0Ar66oYEXXKS9MKdA/__results___files/__results___93_0.png" alt="Target Variable" width="900" height="475" />

  - **Visualising top 30 years with most book being published**

    <img src="https://www.kaggleusercontent.com/kf/68370032/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FAjg2pahzNZq-V0oRBRfIA.kCW2fMt_A8GmMQsF50T940HYuMn_PSmKjW5J9KSxk5g2fzu2FVV2txschB8b2UAzEMp44lZYiXSeEmHUj2J5Y4YYVh5Za8WmsZ6pEIQxSDaWuQ3nm2f1sL9rZHD-psZ4s99Ohm1ixz_NGaiEHEjKMacvTYtu7B06AzlrnPPj7NlJTBykxkJYFZcvAKryfyWKesENfzQ5iyFiiW92JYoHFcu_fJ0RSk6hE67GhXBA_zCNlVNZg9gbHaCr1bBYwzfE03ZP568FkhZU9MqO47edRu6ysxbcWOe2csz6rptlTntVHs58dLrH17pprHJTot4ttugrTjRPy3ra5cETBiysTLda7ozaUCNEZQXbufq-8W8Zxaay-q6nq0P-6oe6QQG7b8sMR1AzI4uWkVwrAwNPShGsoTJBfid3FWO4UnXXfBR7TXmoBO6oprzyp9cB9pT1jO_dXttOlb6rTm7zq-aTIi4xLPRKj8m9UBVP-mTbm81KocNdnght0bDOAlXXcTmae3SNEGvebyy17sfU-ESxNWqhHPjrZAwJB4EmlX8XCBkZf4smh2tX9g4IDaa8v4cDC2RY80psM2-UkEFgaihny_42wmw6cmq9-DycqTL-5McjQGQYg0M0DPvHQDZMwtpJr2iruorzvNTOITmdqIE0EN9WfYS3ycR8XKJHVwaOZWV7SZxpJenJk7QiDI19fz5n.o62nt0Ar66oYEXXKS9MKdA/__results___files/__results___99_0.png" alt="Target Variable" width="900" height="475" />

  - **Visualising top 30 authors with most books**

    <img src="https://www.kaggleusercontent.com/kf/68370032/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FAjg2pahzNZq-V0oRBRfIA.kCW2fMt_A8GmMQsF50T940HYuMn_PSmKjW5J9KSxk5g2fzu2FVV2txschB8b2UAzEMp44lZYiXSeEmHUj2J5Y4YYVh5Za8WmsZ6pEIQxSDaWuQ3nm2f1sL9rZHD-psZ4s99Ohm1ixz_NGaiEHEjKMacvTYtu7B06AzlrnPPj7NlJTBykxkJYFZcvAKryfyWKesENfzQ5iyFiiW92JYoHFcu_fJ0RSk6hE67GhXBA_zCNlVNZg9gbHaCr1bBYwzfE03ZP568FkhZU9MqO47edRu6ysxbcWOe2csz6rptlTntVHs58dLrH17pprHJTot4ttugrTjRPy3ra5cETBiysTLda7ozaUCNEZQXbufq-8W8Zxaay-q6nq0P-6oe6QQG7b8sMR1AzI4uWkVwrAwNPShGsoTJBfid3FWO4UnXXfBR7TXmoBO6oprzyp9cB9pT1jO_dXttOlb6rTm7zq-aTIi4xLPRKj8m9UBVP-mTbm81KocNdnght0bDOAlXXcTmae3SNEGvebyy17sfU-ESxNWqhHPjrZAwJB4EmlX8XCBkZf4smh2tX9g4IDaa8v4cDC2RY80psM2-UkEFgaihny_42wmw6cmq9-DycqTL-5McjQGQYg0M0DPvHQDZMwtpJr2iruorzvNTOITmdqIE0EN9WfYS3ycR8XKJHVwaOZWV7SZxpJenJk7QiDI19fz5n.o62nt0Ar66oYEXXKS9MKdA/__results___files/__results___102_0.png" alt="Target Variable" width="900" height="475" />

  - **Visualising the age distribution of the users**

    <img src="https://www.kaggleusercontent.com/kf/68370032/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FAjg2pahzNZq-V0oRBRfIA.kCW2fMt_A8GmMQsF50T940HYuMn_PSmKjW5J9KSxk5g2fzu2FVV2txschB8b2UAzEMp44lZYiXSeEmHUj2J5Y4YYVh5Za8WmsZ6pEIQxSDaWuQ3nm2f1sL9rZHD-psZ4s99Ohm1ixz_NGaiEHEjKMacvTYtu7B06AzlrnPPj7NlJTBykxkJYFZcvAKryfyWKesENfzQ5iyFiiW92JYoHFcu_fJ0RSk6hE67GhXBA_zCNlVNZg9gbHaCr1bBYwzfE03ZP568FkhZU9MqO47edRu6ysxbcWOe2csz6rptlTntVHs58dLrH17pprHJTot4ttugrTjRPy3ra5cETBiysTLda7ozaUCNEZQXbufq-8W8Zxaay-q6nq0P-6oe6QQG7b8sMR1AzI4uWkVwrAwNPShGsoTJBfid3FWO4UnXXfBR7TXmoBO6oprzyp9cB9pT1jO_dXttOlb6rTm7zq-aTIi4xLPRKj8m9UBVP-mTbm81KocNdnght0bDOAlXXcTmae3SNEGvebyy17sfU-ESxNWqhHPjrZAwJB4EmlX8XCBkZf4smh2tX9g4IDaa8v4cDC2RY80psM2-UkEFgaihny_42wmw6cmq9-DycqTL-5McjQGQYg0M0DPvHQDZMwtpJr2iruorzvNTOITmdqIE0EN9WfYS3ycR8XKJHVwaOZWV7SZxpJenJk7QiDI19fz5n.o62nt0Ar66oYEXXKS9MKdA/__results___files/__results___107_0.png" alt="Target Variable" width="900" height="450" />

  - **Extra Analysis**
    - Some of the Plots and wordclouds which aren't present here can be found in Notebook

## 2. Different ways of building Recommendation system

  1. **Popularity-based**

      These simply recommend the most popular items to users. Popularity-based systems are simplest of all and have minimal computational requirements. However, as these systems       do not make personalized recommendations based on specific user’s likes & behaviors, they tend to be less accurate than content-based or collaborative filtering based
      systems. This type of recommendation is performed in the notebook, the output i.e. 10 most popular books is

      <img src="https://user-images.githubusercontent.com/60045850/126038344-966e91ae-8e62-4c11-9e63-3a0319bc6590.png" alt="Target Variable" width="900" height="475" />

  2. **Content-based**

      Content-based systems depend on external information for creating user and item profiles and this information might not be easily available. Also, these do not take users
      behavioral information into account and discount the fact that user interest and preferences may change over time.

  3. **Collaborative Filtering**

      - Memory-based/ Neighborhood-based
  
          Memory Based recommendation systems can again be divided into two categories i.e. [User Based](https://www.geeksforgeeks.org/user-based-collaborative-filtering/#:~:text=User%2DBased%20Collaborative%20Filtering%20is,for%20building%20their%20recommendation%20system.) and [Item Based](https://www.geeksforgeeks.org/item-to-item-based-collaborative-filtering/) which  can  easily be implemented using similarity measure like [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity), [Pearson similarity](https://neo4j.com/docs/graph-data-science/current/alpha-algorithms/pearson/) are used to find most similar items     according to the Data

      - Model-based/Matrix Factorization

          Model-based Collaborative Filtering approach employs dimensionality reduction techniques like matrix factorization (Singular Value Decomposition — SVD, Principal
          Component Analysis- PCA and Latent Factor models) to discover hidden concepts and their relationship with users and items.
  
      - Hybrid Approach

          Memory-based and model-based collaborative filtering approaches can be combined in practice to exploit the benefits each of the approaches provide. Also, content-based           and collaborative filtering approaches can be combined in various ways to achieve greater synergies between them.
    
## 3. Model and Flask Api

- **Model** :-

    Scikit-Learn's [Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors) model is
    build under collaborative filtering approach. Also we use the Scientific computing library for creating compressed sparse row matrix([csr matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)) from pivot table and is used for modelling with a brute algorithm and cosine as metric    

- **Flask Api** :- 
 
    1. Clone the Project and download [Book_names_with_urlM.csv](https://www.kaggle.com/mohitnirgulkar/book-recommendation-with-data-analysis) from the output section and put it
       in the directory containing model
    ```ruby
      
        git clone https://github.com/raklugrin01/Book-Recommendation-with-EDA
      
   ```  
    2. Install Flask
    
    ```ruby
      
        pip install flask
      
   ```  
    3. Run the python file
    ```ruby
      
        python api.py
      
   ``` 
 - **Testing result** :- 
 - 
    We can see that for a Book Title as input the api returned us 10 books as the recommendations
    
    <img src="https://user-images.githubusercontent.com/60045850/126040732-4036966a-47ce-4654-86f6-9766269c078e.png" alt="Target Variable" width="900" height="450" />
 
 ## 4. Refrences
 
 - [About Recommendation Systems](https://medium.com/@chhavi.saluja1401/recommendation-systems-made-simple-b5a79cac8862)
 - [Turning Model into an Api](https://www.datacamp.com/community/tutorials/machine-learning-models-api-python)
 - [Algorithms](https://www.researchgate.net/post/which_Algorithm_is_best_for_book_recommendation_system)
      
