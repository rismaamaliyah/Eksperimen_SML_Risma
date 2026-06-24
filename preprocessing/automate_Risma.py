import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(df):
    # Fill missing values with the mode for each column
    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Trim outliers for numerical columns using IQR method
    num_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Separate features (X) and target (y)
    X = df.drop(columns=['income', 'fnlwgt']) # 'fnlwgt' is dropped as it's not a feature for prediction
    y = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

    # Define numerical and categorical features
    num_features = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    cat_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

    # Create the preprocessor (ColumnTransformer)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

    return X, y, preprocessor

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('data/adult.csv')  # Load your dataset here
    X, y, preprocessor = preprocess_data(df)
    X.to_csv("adult_income_dataset_preprocessing.csv", index=False)
    print("Preprocessing complete. Features and target are ready for modeling.")