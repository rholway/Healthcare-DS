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

We want to look at avoidable write-off (AWO) transaction amount as a percentage of NPSR.  We want to compare that percentage to an industry average of:
* 0.74% - Top Quartile Performance
* 1.54% - National Average Performance
If we see a gap between actual hospital performance and the national average, we see a potential financial opportunity.  We want to decrease the AWO/NPSR percentage.  If we can decrease the AWO transaction amount, we can save money.

To get an idea of the data we are dealing with, its easiest to visualize the data.

# Exploratory Data Analysis and Data Visualization

We looked at all six hospitals grouped together, and pivoted the six tables out by the following categories, and from those pivoted categories calculated the AWO as a percentage of NPSR.

## All Hospitals

### By Insurance (Insurance_Code_Description)
![All H - by insurance](images/all_hospitals/all-insurance.png)

<!-- ### By Financial Class (Financial_Class)
![All H - by patient class](images/all_hospitals/all-fin-class.png) -->

### By Transaction Detail (NCI_Transaction_Detail)
![All H - by transaction detail](images/all_hospitals/all-trans-det.png)

### By Hospital (LocationID)
![All H - by hospital](images/all_hospitals/all-loc.png)

### By Department (Discharge_Department_ID)
![All H - by dept](images/all_hospitals/all-dept.png)

### By Patient Class (Service_Code)
![All H - by service](images/all_hospitals/all-patient-class.png)

## Hospital 1

### H1 - By Insurance (Insurance_Code_Description)
![H1 - by insurance](images/hosp_1/h1-insurance.png)

### H1 - By Transaction Detail (NCI_Transaction_Detail)
![H1 - by transaction detail](images/hosp_1/h1-trans-det.png)

### H1 - By Department (Discharge_Department_ID)
![H1 - by dept](images/hosp_1/h1-dept.png)

### H1 - By Patient Class (Service_Code)
![H1 - by service](images/hosp_1/h1-patient-class.png)

## Hospital 2

### H2 - By Insurance (Insurance_Code_Description)
![H2 - by insurance](images/hosp_2/h2-insurance.png)

### H2 - By Transaction Detail (NCI_Transaction_Detail)
![H2 - by transaction detail](images/hosp_2/h2-trans-det.png)

### H2 - By Department (Discharge_Department_ID)
![H2 - by dept](images/hosp_2/h2-dept.png)

### H2 - By Patient Class (Service_Code)
![H2 - by service](images/hosp_2/h2-patient-class.png)

## Hospital 3

### H3 - By Insurance (Insurance_Code_Description)
![H3 - by insurance](images/hosp_3/h3-insurance.png)

### H3 - By Transaction Detail (NCI_Transaction_Detail)
![H3 - by transaction detail](images/hosp_3/h3-trans-det.png)

### H3 - By Department (Discharge_Department_ID)
![H3 - by dept](images/hosp_3/h3-dept.png)

### H3 - By Patient Class (Service_Code)
![H3 - by service](images/hosp_3/h3-patient-class.png)

## Hospital 4

### H4 - By Insurance (Insurance_Code_Description)
![H4 - by insurance](images/hosp_4/h4-insurance.png)

### H4 - By Transaction Detail (NCI_Transaction_Detail)
![H4 - by transaction detail](images/hosp_4/h4-trans-det.png)

### H4 - By Department (Discharge_Department_ID)
![H4 - by dept](images/hosp_4/h4-dept.png)

### H4 - By Patient Class (Service_Code)
![H4 - by service](images/hosp_4/h4-patient-class.png)

## Hospital 5

### H5 - By Insurance (Insurance_Code_Description)
![H5 - by insurance](images/hosp_5/h5-insurance.png)

### H5 - By Transaction Detail (NCI_Transaction_Detail)
![H5 - by transaction detail](images/hosp_5/h5-trans-det.png)

### H5 - By Department (Discharge_Department_ID)
![H5 - by dept](images/hosp_5/h5-dept.png)

### H5 - By Patient Class (Service_Code)
![H5 - by service](images/hosp_5/h5-patient-class.png)

## Hospital 6

### H6 - By Insurance (Insurance_Code_Description)
![H6 - by insurance](images/hosp_6/h6-insurance.png)

### H6 - By Transaction Detail (NCI_Transaction_Detail)
![H6 - by transaction detail](images/hosp_6/h6-trans-det.png)

### H6 - By Department (Discharge_Department_ID)
![H6 - by dept](images/hosp_6/h6-dept.png)

### H6 - By Patient Class (Service_Code)
![H6 - by service](images/hosp_6/h6-patient-class.png)
