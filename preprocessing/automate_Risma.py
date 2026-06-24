import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
    
    df = pd.read_csv("adult_income_dataset_raw.csv",
                     header=None, names=columns)
    X, y, preprocessor = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    num_features = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    cat_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    ohe = preprocessor.named_transformers_['cat']
    ohe_features = ohe.get_feature_names_out(cat_features)
    all_features = np.concatenate([num_features, ohe_features])

    X_train_df = pd.DataFrame(X_train_processed.toarray(), columns=all_features)
    X_test_df = pd.DataFrame(X_test_processed.toarray(), columns=all_features)

    train_preprocessed_df = pd.concat([X_train_df, y_train.reset_index(drop=True)], axis=1)
    test_preprocessed_df = pd.concat([X_test_df, y_test.reset_index(drop=True)], axis=1)

    train_preprocessed_df.to_csv("preprocessing/adult_income_dataset_preprocessing/adult_income_dataset_preprocessing.csv", index=False)
    test_preprocessed_df.to_csv("preprocessing/adult_income_dataset_preprocessing/adult_income_dataset_preprocessing_test.csv", index=False)
    
    print("Preprocessing completed. Preprocessed datasets saved to 'preprocessing/adult_income_dataset_preprocessing/'.")