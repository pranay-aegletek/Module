import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
)
from datetime import datetime
import warnings
import os
import logging
import time

warnings.filterwarnings('ignore')

# --- Setup Logging ---
log_filename = 'fraud_detection.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler()
    ]
)

class HealthcareFraudDetector:
    """
    XGBoost-based Healthcare Fraud Detection Model
    """

    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.target_column = 'IsFraud'

    def load_data(self, csv_file_path):
        logging.info(f"Loading data from: {csv_file_path}")
        try:
            df = pd.read_csv(csv_file_path)
            logging.info(f"Data loaded successfully! Shape: {df.shape}")
            logging.info(f"Fraud rate: {df[self.target_column].mean() * 100:.2f}%")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return None

    def preprocess_data(self, df, is_training=True):
        logging.info("Starting data preprocessing...")
        data = df.copy()

        if 'DateOfBill' in data.columns:
            data['DateOfBill'] = pd.to_datetime(data['DateOfBill'])
            data['BillMonth'] = data['DateOfBill'].dt.month
            data['BillDayOfWeek'] = data['DateOfBill'].dt.dayofweek
            data['BillQuarter'] = data['DateOfBill'].dt.quarter
            data.drop('DateOfBill', axis=1, inplace=True)

        if 'FeesChargedByDoctor' in data.columns and 'AmountPaid' in data.columns:
            data['ReimbursementRate'] = data['AmountPaid'] / (data['FeesChargedByDoctor'] + 1e-8)
            data['UnpaidAmount'] = data['FeesChargedByDoctor'] - data['AmountPaid']

        if 'FeesChargedByDoctor' in data.columns and 'DaysInHospital' in data.columns:
            data['CostPerDay'] = data['FeesChargedByDoctor'] / (data['DaysInHospital'] + 1e-8)

        condition_cols = ['IsFever', 'HasFracture', 'NeedsLaparoscopySurgery']
        existing_condition_cols = [col for col in condition_cols if col in data.columns]
        if existing_condition_cols:
            data['ConditionComplexity'] = data[existing_condition_cols].sum(axis=1)

        X = data.drop([self.target_column], axis=1, errors='ignore')
        y = data[self.target_column] if self.target_column in data.columns else None

        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                le = self.label_encoders.get(col)
                if le:
                    known_labels = list(le.classes_)
                    X[col] = X[col].astype(str).apply(lambda x: x if x in known_labels else 'unseen')
                    if 'unseen' not in known_labels:
                        le.classes_ = np.append(le.classes_, 'unseen')
                    X[col] = le.transform(X[col])

        boolean_cols = X.select_dtypes(include=['bool']).columns
        X[boolean_cols] = X[boolean_cols].astype(int)

        if is_training:
            self.feature_names = X.columns.tolist()
        else:
            X = X[self.feature_names]

        logging.info("Preprocessing completed!")
        return X, y

    def train_model(self, X_train, y_train):
        logging.info("Starting XGBoost model training...")
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        logging.info("Model training completed!")

    def evaluate_model(self, X_test, y_test):
        logging.info("Evaluating model performance...")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        logging.info("\n" + "=" * 50 + "\nMODEL EVALUATION RESULTS\n" + "=" * 50)
        logging.info(f"Accuracy: {accuracy * 100:.2f}%")
        logging.info(f"F1 Score: {f1:.4f}")
        logging.info(f"AUC-ROC: {auc_roc:.4f}")
        logging.info("\nClassification Report:\n" + classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

        cm = confusion_matrix(y_test, y_pred)
        logging.info("\nConfusion Matrix:")
        logging.info(f"True Negatives: {cm[0, 0]}, False Positives: {cm[0, 1]}")
        logging.info(f"False Negatives: {cm[1, 0]}, True Positives: {cm[1, 1]}")

        self._display_feature_importance()

        return {'predictions': y_pred, 'probabilities': y_pred_proba}

    def _display_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            logging.info("\nFEATURE IMPORTANCE RANKING:\n" + "=" * 40 + "\n" + importance_df.head(10).to_string(index=False))

    def save_model(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fraud_detection_model_{timestamp}.joblib"
        import joblib
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filename)
        logging.info(f"Model saved as: {filename}")

def main():
    overall_start_time = time.time()
    logging.info("Healthcare Fraud Detection with XGBoost")
    logging.info("=" * 50)

    local_data_path = 'healthcare_claims_complete.csv'

    if not os.path.exists(local_data_path):
        logging.error(f"Data file not found at {local_data_path}")
        logging.error("Please run the `generate_data.py` script first.")
        return

    detector = HealthcareFraudDetector()
    df = detector.load_data(local_data_path)
    if df is None:
        return

    start_time = time.time()
    X, y = detector.preprocess_data(df, is_training=True)
    logging.info(f"Data preprocessing took: {time.time() - start_time:.2f} seconds")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    start_time = time.time()
    detector.train_model(X_train, y_train)
    logging.info(f"Model training took: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    eval_results = detector.evaluate_model(X_test, y_test)
    logging.info(f"Model evaluation took: {time.time() - start_time:.2f} seconds")
    
    detector.save_model()

    logging.info("\n" + "=" * 50 + "\nGENERATING RED FLAG REPORT\n" + "=" * 50)
    test_df_original = df.iloc[X_test.index].copy()
    test_df_original['PredictedFraud'] = eval_results['predictions']
    test_df_original['FraudProbability'] = eval_results['probabilities']

    red_flag_df = test_df_original[test_df_original['PredictedFraud'] == 1].copy()
    red_flag_df['FraudProbability'] = red_flag_df['FraudProbability'].apply(lambda x: f"{x:.2%}")
    red_flag_df.sort_values(by='FraudProbability', ascending=False, inplace=True)
    
    report_filename = "red_flag_report.csv"
    red_flag_df.to_csv(report_filename, index=False)
    logging.info(f"Red Flag Report saved as: {report_filename}")
    logging.info(f"Found {len(red_flag_df)} potential fraud cases.")
    
    logging.info(f"Total execution time: {time.time() - overall_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
