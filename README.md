So, I have created two python scripts .

The first one [ numeric-rf-prediction.py ] 
it takes as parameters a training set with these columns: [ LabId	ageAtDiagnosis	isRelapse	isDenovo	isTransformed	dxAtInclusion	specificDxAtInclusion	cumulativeTreatmentTypes	drug_label ] ,
and a test dataset with the same columns , excecpt the drug_label column, where is the column that we expect , our model to predict.

