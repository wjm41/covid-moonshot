#! /bin/bash
#head -1 expanded_noncovalent_model_2_scores_batch_0.csv > expanded_noncovalent_model_1_scores.csv
#head -1 expanded_noncovalent_model_2_scores_batch_0.csv > expanded_noncovalent_model_2_scores.csv
#head -1 expanded_noncovalent_model_2_scores_batch_0.csv > expanded_noncovalent_model_3_scores.csv
#head -1 expanded_noncovalent_model_2_scores_batch_0.csv > expanded_noncovalent_model_4_scores.csv
#head -1 expanded_noncovalent_model_2_scores_batch_0.csv > expanded_noncovalent_model_5_scores.csv
#cat expanded_noncovalent_model_1_scores_batch_* | grep -v 'SMILES' >> expanded_noncovalent_model_1_scores.csv
#cat expanded_noncovalent_model_2_scores_batch_* | grep -v 'SMILES' >> expanded_noncovalent_model_2_scores.csv
#cat expanded_noncovalent_model_3_scores_batch_* | grep -v 'SMILES' >> expanded_noncovalent_model_3_scores.csv
#cat expanded_noncovalent_model_4_scores_batch_* | grep -v 'SMILES' >> expanded_noncovalent_model_4_scores.csv
#cat expanded_noncovalent_model_5_scores_batch_* | grep -v 'SMILES' >> expanded_noncovalent_model_5_scores.csv
#python ensemble_preds_top.py expanded_noncovalent
#python process_scores.py -input expanded_noncovalent_ensemble4_topscore.csv -output top_noncovalent_ensemble4_topscore.csv -target rest
head -1 expanded_acrylib_model_2_scores_batch_0.csv > expanded_acrylib_model_1_scores.csv
head -1 expanded_acrylib_model_2_scores_batch_0.csv > expanded_acrylib_model_2_scores.csv
head -1 expanded_acrylib_model_2_scores_batch_0.csv > expanded_acrylib_model_3_scores.csv
head -1 expanded_acrylib_model_2_scores_batch_0.csv > expanded_acrylib_model_4_scores.csv
head -1 expanded_acrylib_model_2_scores_batch_0.csv > expanded_acrylib_model_5_scores.csv
cat expanded_acrylib_model_1_scores_batch_* | grep -v 'SMILES' >> expanded_acrylib_model_1_scores.csv
cat expanded_acrylib_model_2_scores_batch_* | grep -v 'SMILES' >> expanded_acrylib_model_2_scores.csv
cat expanded_acrylib_model_3_scores_batch_* | grep -v 'SMILES' >> expanded_acrylib_model_3_scores.csv
cat expanded_acrylib_model_4_scores_batch_* | grep -v 'SMILES' >> expanded_acrylib_model_4_scores.csv
cat expanded_acrylib_model_5_scores_batch_* | grep -v 'SMILES' >> expanded_acrylib_model_5_scores.csv
python ensemble_preds.py expanded_noncovalent
python ensemble_preds_top.py expanded_noncovalent
python process_scores.py -input expanded_noncovalent_ensemble4.csv -output top_noncovalent_ensemble4_final.csv -target rest
python process_scores.py -input expanded_noncovalent_model_1_scores.csv -output top_noncovalent_model1_final.csv -target rest
#python ensemble_preds.py expanded_noncovalent
#python process_scores.py -input expanded_noncovalent_ensemble4.csv -output top_noncovalent_ensemble4_final.csv -target rest
#python process_scores.py -input expanded_noncovalent_model_1_scores.csv -output top_noncovalent_model1_final.csv -target rest
