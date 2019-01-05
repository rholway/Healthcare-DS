# Analyzing Healthcare Data

We obtained heath care data of patients, whose names and information has been made private.  The information provided in the data includes the following:

* account_id - patients account identifier
* LocationID - six unique values (Hospital 1, 2, 3, 4, 5, 6)
* Service_Code - seven unique values (Outpatient, Emergency, Observation, Dialysis, Psych, Rehab, Inpatient)
* Transaction_Number
* TransactionType_Tdata - all 'Adjustment'
* Service_Date
* Post_Date
* Account_Discharge_Date
* Procedure_Code - 515 unique values
* Procedure_Description - 30 unique values
* Transaction_Amount - avoidable write-off amount
* AWO $ Bucket - six unique values (1. <0, 2. 0-1000, 3. 1000-2500, 4. 2500-5000, 5. 5000-10000, 6.10000+)
* Insurance_Code - 141 unique values
* Insurance_Code_Description - 15 unique values
* Financial_Class - three unique values (Government, Commercial, Other)
* NCI_Transaction_Category - all 'Adjustment'
* NCI_Transaction_Type - all 'Avoidable Write-offs'
* NCI_Transaction_Detail - five unique values (Non-Covered, Authorization-Referral, Untimely Filing, Medical Necessity, Other Avoidable Write-offs)
* Account_Balance
* Account_Billed_Date
* Admit_Date
* Admit_Diagnosis - 5310 unique values, ~15% null
* Admit_Type - five unique values (Elective, Emergency, Other, Urgent, Triage)
* Billing_Status - five unique value (Final Bill, Bad Debt, Other, Un-billed, In-house)
* Current_Patient_Class_Description - seven unique values (Office visit, Outpatient, Emergent, Observation, Same Day Surgery, Recurring, Inpatient)
* Discharge_Department_ID - 28 unique values
* First_Claim_Date
* Last_Claim_Date
* Last_Payment_Amount
* Last_Payment_Date
* Length_Of_Stay
* Medicaid_Pending_Flag - Yes or No
* Medical_Record_Number
* Patient_Insurance_Balance
* Patient_Self_Pay_Balance
* Primary_Diagnosis - 5033 unique values
* Primary_Service_Name - 29 unique values
* ICD-10_Diagnosis_Principal - 5310 unique values
* ICD-10_Diagnosis - 5310 unique values
* Region - three unique values (Central, East, West)
* NPSR - Net Patient Service Revenue

From this information, we looked at avoidable write-off transaction amounts by the following:
* Insurance
* Transaction Detail
* Hospital
* Patient Class
* Department
* Date
* Service Type

We want to look at avoidable write-off transaction amount as a percentage of NPSR.  We want to compare that percentage to an industry average of:
* 0.74% - Top Quartile Performance
* 1.54% - National Average Performance
If we see a gap between actual hospital performance and the national average, we see a potential financial opportunity.  We want to decrease the AWO/NPSR percentage.  If we can decrease the AWO transaction amount, we can save money.

To get an idea of the data we are dealing with, its easiest to visualize the data.

# Exploratory Data Analysis and Data Visualization
