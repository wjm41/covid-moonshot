import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.Descriptors import ExactMolWt


def main():

    enamine_df = pd.read_csv('Enamine_submissions.csv', usecols=['SMILES','CID'])
    SA_df = pd.read_csv('covid_SA_file_new.csv', usecols=['SMILES', 'MW', 'CID'])
    #SA_df = pd.read_csv('covid_SA_file.csv', usecols=['SMILES', 'MW', 'CID'])
    score_df = pd.read_csv('score_data.csv', usecols=['SMILES', 'TITLE', 'Chemgauss4 Score'])

    SA_df['SMILES'] = SA_df['SMILES'].apply(lambda x: MolToSmiles(MolFromSmiles(x)))
    enamine_df['SMILES'] = enamine_df['SMILES'].apply(lambda x: MolToSmiles(MolFromSmiles(x)))
    enamine_df['MW'] = enamine_df['SMILES'].apply(lambda x: ExactMolWt(MolFromSmiles(x)))
    score_df['SMILES'] = score_df['SMILES'].apply(lambda x: MolToSmiles(MolFromSmiles(x)))
    score_df = score_df.rename(columns={'TITLE':'CID'})

    score_df = score_df[score_df['Chemgauss4 Score']<-6]
    SA_df = SA_df[SA_df['MW'] > 250]
    enamine_df = enamine_df[enamine_df['MW'] > 250]

    #SA_df.to_csv('covid_SA_file.csv', index=False)
    #enamine_df.to_csv('Enamine_submissions.csv', index=False)
    #score_df.to_csv('score_data.csv', index=False)

    SA_df['source'] = 'covid_SA'
    enamine_df['source'] = 'enamine'

    print('Post-filtering:')
    print(SA_df.describe())
    print(enamine_df.describe())

    merged_synth_df = pd.merge(SA_df, enamine_df, how='outer')
    print(merged_synth_df[merged_synth_df.duplicated(subset='SMILES', keep=False)])
    # merged_synth_df = SA_df

    score_synth_df = pd.merge(merged_synth_df.drop_duplicates(subset='CID'), score_df.drop_duplicates(subset='CID'),
                              how='inner', on='CID')
    score_synth_df = score_synth_df.rename(columns={'SMILES_y':'SMILES', 'Chemgauss4 Score':'dock_score'})
    score_synth_df = score_synth_df.drop_duplicates(subset='SMILES')

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #    print(score_synth_df)
    merged_synth_df[~merged_synth_df['CID'].isin(score_synth_df['CID'])].to_csv('missing.csv', index=False)

    final_df = score_synth_df[['SMILES', 'source', 'CID', 'MW', 'dock_score']].sort_values(by='dock_score')
    print(final_df['source'].value_counts())
    final_df.to_csv('processed_data.csv', index=False)

    score_synth_df = pd.merge(enamine_df.drop_duplicates(subset='CID'),
                              score_df.drop_duplicates(subset='CID'), how='inner', on='CID')
    score_synth_df = score_synth_df.rename(columns={'SMILES_y':'SMILES', 'Chemgauss4 Score':'dock_score'})
    score_synth_df = score_synth_df.drop_duplicates(subset='SMILES')
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #    print(score_synth_df)
    final_df_2 = score_synth_df[['SMILES', 'source', 'CID', 'MW', 'dock_score']].sort_values(by='dock_score')
    print(final_df_2['source'].value_counts())
    final_df_2.to_csv('processed_data_enamine.csv', index=False)

    # print(final_df.describe())


if __name__ == '__main__':

    main()