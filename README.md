Business needs:
    model predicts the price of phone in rupees

Requirements:
    python 3.7
    numpy==1.17.3
    pandas==1.1.5
    sklearn==1.0.0

Running:
    To run the demo, execute:
        python predict.py 

    After running the script in that folder will be generated <prediction_results.csv> 
    The file has 'price_pred' column with the result value.

    The input is expected csv file in the same folder with a name <test_data.csv>. The file should have all features columns. 

Training a Model:
    Before you run the training script for the first time, you will need to create dataset. The file <train_data.csv> should contain all features columns and target for prediction class.
    After running the script the "param_dict.pickle"  and "finalized_model.saw" will be created.
    Run the training script:
        python train.py

    The model accuracy is 93%
    There is no fraud check.