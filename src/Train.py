from Features import prepare_features
from Models import get_models
from Evaluation import evaluate_model
import pandas as pd

df = pd.read_csv(r'C:\Users\User\Desktop\Lung Cancer Prediction\data\preprocessed_lung_cancer_data.csv')
X_train,X_test,y_train,y_test = prepare_features(df)
models = get_models()

for name,model in models.items():
    print(f'\nTraining {name}')
    model.fit(X_train,y_train)
    evaluate_model(model,X_test,y_test)
    print('-'*50)

