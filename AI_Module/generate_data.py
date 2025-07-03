import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Expanded multicultural name lists
first_names = [
    'John', 'Jane', 'Alice', 'Bob', 'Charlie', 'Diana', 'Eva', 'Frank', 'Grace', 'Henry',
    'Priya', 'Rahul', 'Arjun', 'Anjali', 'Ravi', 'Chen', 'Yuki', 'Hiroshi', 'Fatima', 'Ahmed',
    'Jean', 'Luca', 'Aisha', 'Omar', 'Zainab', 'Giulia', 'Kwame', 'Marta', 'Ren', 'Nina'
]

last_names = [
    'Doe', 'Smith', 'Johnson', 'Brown', 'Wilson', 'Davis', 'Garcia', 'Miller', 'Lee', 'Taylor',
    'Patel', 'Sharma', 'Mehta', 'Verma', 'Zhang', 'Tanaka', 'Sato', 'Khan', 'Noor', 'Rahman',
    'Laurent', 'Bianchi', 'Popov', 'Mensah', 'Choudhary', 'Rossi', 'Moore', 'Sanchez', 'Chen', 'Juma'
]
doctor_ids = [f"DOC{str(i).zfill(4)}" for i in range(100)]
hospital_ids = [f"HSP{str(i).zfill(3)}" for i in range(30)]


def random_patient_name():
    return f"{random.choice(first_names)} {random.choice(last_names)}"


used_names = set()


def unique_random_patient_name():
    while True:
        name = random_patient_name()
        if name not in used_names:
            used_names.add(name)
            return name


def generate_synthetic_claims_data(num_claims=10000, fraud_rate=0.2):
    headers = ['ClaimID', 'PatientName', 'PolicyNumber', 'DateOfBill', 'AmountPaid',
               'DiagnosisCode', 'FeesChargedByDoctor', 'DaysInHospital', 'IsFever',
               'HasFracture', 'NeedsLaparoscopySurgery', 'ClaimStatus', 'IsFraud',
               'DoctorID', 'HospitalID', 'ReimbursementRate', 'PreviousClaims',
               'HighRiskRegion', 'ChronicCondition']

    diagnosis_codes = ['A01', 'B02', 'C03', 'D04', 'E05', 'F06', 'G07', 'H08', 'I09', 'J10']
    claim_statuses = ['Approved', 'Pending', 'Rejected', 'Under Review']

    # --- NEW: Select high-risk doctors ---
    high_risk_doctor_ids = random.sample(doctor_ids, int(len(doctor_ids) * 0.1)) # 10% are high risk
    print(f"Identified {len(high_risk_doctor_ids)} high-risk doctors.")

    data = []
    for i in range(num_claims):
        claim_id = f"CLM{str(i + 1).zfill(6)}"
        patient_name = random_patient_name()
        policy_number = random.randint(1000000, 9999999)
        doctor_id = random.choice(doctor_ids) # Choose doctor first
        hospital_id = random.choice(hospital_ids)

        # --- NEW: Temporal bias for claim dates ---
        if random.random() < 0.20:  # 20% chance of being an end-of-quarter claim
            quarter = random.randint(1, 4)
            # Get a date in the last 10 days of a quarter
            quarter_end_month = 3 * quarter
            last_day = 30 if quarter_end_month in [4, 6, 9, 11] else 31
            if quarter_end_month == 2: last_day = 28
            start_day = last_day - 10
            random_date = datetime(2023, quarter_end_month, random.randint(start_day, last_day))
        else:
            random_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 364))
        date_of_bill = random_date.strftime("%Y-%m-%d")

        diagnosis_code = random.choice(diagnosis_codes)
        claim_status = random.choice(claim_statuses)

        is_fever = random.choice([True, False])
        has_fracture = random.choice([True, False])
        needs_laparoscopy = random.choice([True, False])

        base_fee = 1000
        if is_fever: base_fee += random.randint(500, 1500)
        if has_fracture: base_fee += random.randint(2000, 5000)
        if needs_laparoscopy: base_fee += random.randint(15000, 25000)

        days_in_hospital = 1
        if is_fever: days_in_hospital += random.randint(1, 3)
        if has_fracture: days_in_hospital += random.randint(3, 7)
        if needs_laparoscopy: days_in_hospital += random.randint(2, 5)

        # --- MODIFIED: is_fraud decision ---
        # Higher chance of fraud if doctor is high-risk
        fraud_probability = fraud_rate * 5 if doctor_id in high_risk_doctor_ids else fraud_rate
        is_fraud = random.random() < fraud_probability

        if is_fraud:
            fraud_type = random.choice(['overbilling', 'duplicate', 'unnecessary_procedure'])
            if fraud_type == 'overbilling':
                fees_charged = base_fee * random.uniform(2.5, 4.0)
                amount_paid = min(fees_charged * 0.8, base_fee * 1.2)
            elif fraud_type == 'duplicate':
                # The 'duplicate' type will be handled by the duplicate creation logic below
                fees_charged = base_fee * random.uniform(1.0, 1.2)
                amount_paid = fees_charged * 0.9
            else: # unnecessary_procedure
                fees_charged = base_fee + random.randint(10000, 30000)
                amount_paid = fees_charged * 0.7
                needs_laparoscopy = True
        else:
            fees_charged = base_fee * random.uniform(0.9, 1.1)
            amount_paid = fees_charged * random.uniform(0.85, 0.95)

        reimbursement_rate = amount_paid / (fees_charged + 1e-6)
        previous_claims = random.randint(0, 10)
        high_risk_region = random.choice([True, False])
        chronic_condition = random.choice([True, False])

        record = [
            claim_id, patient_name, policy_number, date_of_bill, round(amount_paid, 2),
            diagnosis_code, round(fees_charged, 2), days_in_hospital, is_fever,
            has_fracture, needs_laparoscopy, claim_status, is_fraud,
            doctor_id, hospital_id, round(reimbursement_rate, 3), previous_claims,
            high_risk_region, chronic_condition
        ]

        data.append(record)

        # --- NEW: Subtle duplicate generation ---
        if is_fraud and fraud_type == 'duplicate' and random.random() < 0.7: # 70% of 'duplicate' frauds get a subtle duplicate
            dup_record = record.copy()
            # Modify the duplicate slightly
            dup_record[0] = f"CLM{str(num_claims + len(data) + 1).zfill(6)}_DUP" # New unique ClaimID
            original_date = datetime.strptime(dup_record[3], "%Y-%m-%d")
            dup_record[3] = (original_date + timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%d")
            dup_record[6] = round(dup_record[6] * random.uniform(0.98, 1.02), 2) # Slightly alter fee
            data.append(dup_record)

    return pd.DataFrame(data, columns=headers)


def main():
    """
    Main function to generate the synthetic healthcare claims data.
    """
    print("Generating synthetic healthcare claims dataset...")
    df_full = generate_synthetic_claims_data(10000, fraud_rate=0.25)
    
    # Save the dataset
    output_filename = "healthcare_claims_complete.csv"
    df_full.to_csv(output_filename, index=False)
    print(f"Dataset saved to: {output_filename}")

if __name__ == "__main__":
    main()
