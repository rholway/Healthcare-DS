-EDA
  total AWO amount/awo-npsr percentage by service code
  total AWO amount/awo-npsr by NCI_transaction detail
    BOTH all hospitals and by individual hospital

  Keep titles/descriptions clear as we go

  Keep in mind features opportunities

-Unsupervised learning/clustering
  PCA -> to show difficulty
  KMeans clustering over all hospitals -> drop hospitals (mimic total system)
  KMeans clustering in each hospital
    -Inspect aspects of each cluster
  GMMs for soft clustering
    GMM pipeline with ML models

  keep description for methods clear


-Predictive modeling
  Start small - > linear regression, logistic regression (use VIF)

  ML models for regression (AWO amount OR awo/npsr percent)
                classification AWO buckets

-Timeline:
  Monday (1/14) EOD: Finish all EDA
  Friday (1/16) EOD: Clustering complete
  Following: Brainstorm and complete ML models

  Columns to drop:
  -account_id
  -Transaction_Number
  -TransactionType_Tdata
  -Procedure_Code
  -Insurance_Code
  -Insurance_Code_Description -> This is kept in with "Financial_Class"
  -NCI_Transaction_Category
  -NCI_Transaction_Type
  -Account_Balance
  -Admit_Diagnosis -> Consider adding back in
  -'Billing_Status'
  -'Medicaid_Pending_Flag'
  -Medical_Record_number
  -Patient_insurance_balance -> largely zeros
  -Patient_Self_Pay_Balance -> largely zeros
  -Primary_Diagnosis -> Consider adding back in
  -ICD-10_Diagnosis_Principal
  -ICD-10_Diagnosis

  -Service_Date
  -Post_Date
  -Account_Discharge_Date
  -Account_Billed_Date
  -Admit_Date
  -First_Claim_Date
  -Last_Claim_Date
  -Last_Payment_Date

  Columns to keep:
  -LocationID
  -Service_Code
  -Procedure_Description
  -AWO $ Bucket
  -Financial_Class
  -NCI_Transaction_Detail -> Consider removing
  -Admit_type
  -Current_Patient_class_Description
  -Discharge_Department_ID
  -Last_Payment_Amount
  -Length_of_Stay
  -Primary_Service_Name
  -NPSR

  columns to remain not dummied:
  -locationid
  -account_id.1 -> changing to awo_amount
  -awo_bucket
  -last_payment_amount
  -length_of_stay
  -region
  -npsr
