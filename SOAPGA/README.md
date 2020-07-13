# CovidGenetic
Archived code for conducting a genetic algorithm search for molecules with high similarities to known COVID-19 protease inhibitors - recap on the protease can be found [here](https://pdb101.rcsb.org/motm/242)

Visualise protease as well as the ligands which inhibit specific sites [here](https://fragalysis.diamond.ac.uk/viewer/react/preview/target/Mpro)

3D coordinates of all the ligands (need to filter to get the relevant ones) can be found [here](https://www.diamond.ac.uk/covid-19/for-scientists/Main-protease-structure-and-XChem/Downloads.html) in the .pdb files which are formatted like [this](https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/dealing-with-coordinates). P.S. This data has been removed from the git repo because it takes up too much space.

Background on the graph-based genetic algorithm (GB-GA) that I used can be found in [this paper](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c#!divAbstract) - I co-opted the GB-GA code from [this Github](https://github.com/jensengroup/GB-GA)

## Dependencies (for running the GA)
- [dscribe](https://singroup.github.io/dscribe/install.html) - make sure you get the latest version which is much quicker at calculating SOAP descriptors
- [RDKit](https://www.rdkit.org/docs/Install.html)
- pandas

## Data
Fragment data was preprocessed with openbabel to convert the .mol files into .xyz, the coordinates then concatenated the atom coordinates into one file.

Candidates were found from this [Google Sheets](https://docs.google.com/spreadsheets/d/1zELgd-kDEkIjRqc_jdKm5EzDQmRrrYAbErghTPkcA5c/edit#gid=0) and the 'SMILES' column is saved in `data/covid_submissions.csv`
## How to use
running `python GA-soap.py` will start running a genetic algorithm. It uses `crossover.py` and `mutate.py` from Jensen's [Github](https://github.com/jensengroup/GB-GA). Doc-strings and comments in `GA-soap.py` should be enough to help you understand what's going on.
