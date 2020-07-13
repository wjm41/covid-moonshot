from rdkit.Chem import AllChem, MolFromSmiles, MolToSmiles, Draw, Crippen
from rdkit import Chem
from rdkit.Chem.rdmolops import FastFindRings
import pandas as pd
from tqdm import tqdm
import numpy as np

def canonicalize(mol_list, showprogress=False):
    if showprogress:
        mol_list = [MolFromSmiles(MolToSmiles(mol)) for mol in tqdm(mol_list)]
    else:
        mol_list = [MolFromSmiles(MolToSmiles(mol)) for mol in mol_list]
    mol_list = [mol for mol in mol_list if mol]
    mol_list = list(set([MolToSmiles(mol) for mol in mol_list]))
    if showprogress:
        mol_list = [MolFromSmiles(smi) for smi in tqdm(mol_list)]
    else:
        mol_list = [MolFromSmiles(smi) for smi in mol_list]
    return mol_list


def multi_prods(mol_list, rxn, debug=False):
    prod1_list = []
    prod2_list = []
    prod3_list = []
    prod4_list = []
    for mol in mol_list:
        if debug:
            print(MolToSmiles(mol))
        try:
            mol.UpdatePropertyCache()
            FastFindRings(mol)
        except:
            print('This mol fails! ' + MolToSmiles(mol))
#             print('This mol fails! ' +mol)
            continue
        products = rxn.RunReactants((Chem.AddHs(mol),))
        if products != ():
            for prod in products:
                prod1_list.append(prod[0])
                prod2_list.append(prod[1])
                prod3_list.append(prod[2])
                prod4_list.append(prod[3])
    return prod1_list, prod2_list, prod3_list, prod4_list

def multi_rxnts(mol1_list, mol2_list, mol3_list, mol4_list, rxn, debug=False):
    prod_list = []
    for mol1 in tqdm(mol1_list):
        for mol2 in mol2_list:
            for mol3 in mol3_list:
                for mol4 in mol4_list:
                    products = rxn.RunReactants((Chem.AddHs(mol1), Chem.AddHs(mol2), Chem.AddHs(mol3), Chem.AddHs(mol4)))
                    # print(products)
                    if products != ():
                        for prod in products:
                            # print(prod)
                            # if debug:
                                # print(MolToSmiles(prod[0]))
                            prod_list.append(prod[0])
    return prod_list

def simple_rxn(mol_list, rxn, debug=False):
    prod_list = []
    for mol in mol_list:
        if debug:
            print('Input: '+ MolToSmiles(mol))
        products = rxn.RunReactants((Chem.AddHs(mol),))
        if debug:
            print('Products: {}'.format(products))
        if products != ():
            for prod in products:
                if debug:
                    print(prod)
                    print(MolToSmiles(prod[0]))
                prod_list.append(prod[0])
    return prod_list

def pair_prods(mol_list, rxn, debug=False):
    prod1_list = []
    prod2_list = []
    for mol in mol_list:
        if debug:
            print(MolToSmiles(mol))
        try:
            mol.UpdatePropertyCache()
            FastFindRings(mol)
        except:
            print('This mol fails! ' + MolToSmiles(mol))
            continue
        products = rxn.RunReactants((Chem.AddHs(mol),))
        if products != ():
            for prod in products:
                prod1_list.append(prod[0])
                prod2_list.append(prod[1])
    return prod1_list, prod2_list

df_orig = pd.read_csv('new_activities/acry_activity.smi')
# df_actives = df_orig[df_orig['activity']==1]
# print('Number of acry actives: {}'.format(len(df_actives)))
smiles_list = df_orig['SMILES'].values
smiles_list = list(set([MolToSmiles(MolFromSmiles(smi)) for smi in smiles_list]))
# print(smiles_list)
# print([MolFromSmiles(smi) for smi in smiles_list])
# mol_list = [MolFromSmiles(MolToSmiles(MolFromSmiles(smi))) for smi in smiles_list]
# mol_list = [mol for mol in mol_list if mol]
# print(mol_list)
# print(len(list(set([MolToSmiles(mol) for mol in mol_list]))))
mols = [MolFromSmiles(smi) for smi in smiles_list]
#print('Size of actives: {}'.format(len(canonicalize(mols))))
print('Size of original dataset: {}'.format(len(canonicalize(mols))))
acry_slice = AllChem.ReactionFromSmarts('[c,C:1][C](=[O])[N]([c,C,#1:2])[C]([c,C,#1:3])([c,C,#1:4])[C](=[O])[N]([#1])[c,C:5]>>[*:1][C](=[O])[O][#1].[*:2][N]([#1])[#1].[*:3][C](=[O])[*:4].[*:5][N+]#[C-]')
acry_comb = AllChem.ReactionFromSmarts('[c,C:1][C](=[O])[O][#1].[c,C:2][N]([#1])[#1].[c,C,#1:3][C](=[O])[c,C,#1:4].[c,C:5][N+]#[C-]>>[*:1][C](=[O])[N]([*:2])[C]([*:3])([*:4])[C](=[O])[N]([#1])[*:5]')

comp1, comp2, comp3, comp4 = multi_prods(mols,acry_slice)
comp1 = canonicalize(comp1)
comp2 = canonicalize(comp2)
comp3 = canonicalize(comp3)
comp4 = canonicalize(comp4)
# print(comp1)
# print(comp2)
# print(comp3)
# print(comp4)

df = pd.read_csv('new_activities/rest_activity.smi')
df = df[df['activity']==1]
print('Number of non-covalent actives: {}'.format(len(df)))
smiles_list = df['SMILES'].values
rest_mols = [MolFromSmiles(smi) for smi in smiles_list]

amide_to_urea = AllChem.ReactionFromSmarts('[C:1]([#1])[C:2](=[O:3])[N:4] >> [N:1][C:2](=[O:3])[N:4]')
amide_swap = AllChem.ReactionFromSmarts('[C:1]([#1])[C:2](=[O:3])[N:4] >> [N:1][C:2](=[O:3])[C:4][#1]')
urea_to_amide = AllChem.ReactionFromSmarts('[N:1][C:2](=[O:3])[N:4]>>[N:1][C:2](=[O:3])[C:4]')
amide_slice = AllChem.ReactionFromSmarts('[C:1](=[O:2])[N:3]>>[C:1](=[O:2])[O][#1].[N:3][#1]')
amine_decomp = AllChem.ReactionFromSmarts('[NR0H1:1]([#6:2])[#6:3]>>[N:1]([#6:2])[#1].[#6:3][Br]')
aldehyde_lib = AllChem.ReactionFromSmarts('[#6,#7:1][NX3H2]>>[*:1][C](=[O])[#1]')
urea_list = simple_rxn(rest_mols, amide_to_urea)
print(len(urea_list))
swap_list = simple_rxn(rest_mols, amide_swap)
print(len(swap_list))
amide_list = simple_rxn(rest_mols, urea_to_amide)
print(len(amide_list))
products_so_far = canonicalize(rest_mols + urea_list + swap_list + amide_list)
# print('products so far')
# for mol in products_so_far:
#     print(MolToSmiles(mol))
#     print(mol.GetSubstructMatch(interesting_mol))
#print([mol.GetSubstructMatch(interesting_mol) for mol in products_so_far])

# print('For Amide Slicing:')
acid_list, amine_list = pair_prods(products_so_far, amide_slice)
acid_list = canonicalize(acid_list)
amine_list = canonicalize(amine_list)
print(len(acid_list))
print(len(amine_list))
# print('amine')
# print([mol.GetSubstructMatch(interesting_mol) for mol in amine_list])

# print('For Amine Decomposition:')
amine2_list, subs_list = pair_prods(amine_list, amine_decomp)
# print(len(amine2_list))
# print(subs_list)
# print('amine2')
# print([mol.GetSubstructMatch(interesting_mol) for mol in amine2_list])

amine_list = canonicalize(amine_list + amine2_list)
print(len(amine_list))
aldy_list = canonicalize(simple_rxn(amine_list, aldehyde_lib))
comp2 = canonicalize(comp2 + amine_list)
comp3 = canonicalize(comp3 + aldy_list)
print('Number of amines: {}'.format(len(comp2)))
print('Number of aldehydes: {}'.format(len(comp3)))
final_lib = multi_rxnts(comp1, comp2, comp3, comp4, acry_comb)
# # print(final_lib)
final_lib = canonicalize(final_lib, showprogress=True)
print('Size of pre-filtered library: {}'.format(len(final_lib)))
# with open('new_activities/final_lib.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % mol for mol in final_lib)
# np.savetxt('new_activities/final_lib.txt',final_lib)
df = pd.DataFrame(final_lib, columns=['mol'])
df['SMILES'] = [MolToSmiles(mol) for mol in df['mol']]
# n_dup = 0
# for smi in tqdm(smiles_list):
#     if smi in df['SMILES'].values:
#         n_dup+=1
# import numpy as np
# main_list = np.setdiff1d(smiles_list,df['SMILES'].values)
# print(main_list)
# df_missed = df_orig[df_orig['SMILES'].isin(main_list)]
# print(df_missed['SMILES'].values)
# print('Number of duplicates: {}'.format(n_dup))
# df = pd.read_csv('new_activities/final_lib.txt')
# df['mol'] = [MolFromSmiles(smi) for smi in df['SMILES']]
df = df[~df['mol'].isna()]
# for mol in df['mol']:
#     print(MolToSmiles(mol))
#     print(Crippen.MolLogP(mol))
df['logP'] = [Crippen.MolLogP(mol) for mol in df['mol']]
df['num_heavy_atoms'] = [mol.GetNumHeavyAtoms() for mol in df['mol']]
df = df[df['logP']<=5]
df = df[df['num_heavy_atoms']>17]

df = df['SMILES']
print('Size of library: {}'.format(len(df)))
# print(df)
df.to_csv('new_activities/aldehyde_library_expanded.smi', index=False)

# final_lib = np.random.choice(final_lib, size=20, replace=False)
# for mol in final_lib:
# for mol in mols:
#     tmp = AllChem.Compute2DCoords(mol)
# img=Draw.MolsToGridImage(mols,molsPerRow=5,subImgSize=(300,300))
# img.save('non_covalent_actives.png')
# img=Draw.MolsToGridImage(final_lib,molsPerRow=5,subImgSize=(300,300))
# img.save('amide_lib_samples.png')