# uofthacksvii-machine-learning
A "middleware" built using the Flask framework that accepts POST requests with JSON objects, and returns values estimated by trained machine learning models.

2 supervised, classification machine learning models were implemented:
    
    1. Logistic Regression
    Send a post request to /prediction/one
        
    2. Random Forest
    Send a post request to /prediction/multiple

Both POST routes will return a JSON object with the estimated survival rates of the cardiac arrest victims in different scenarios.