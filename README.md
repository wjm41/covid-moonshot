# CovidGenetic
Genetic algorithm search for molecules with high similarities to known COVID-19 protease inhibitors - recap on the protease can be found [here](https://pdb101.rcsb.org/motm/242)

Visualise protease as well as the ligands which inhibit specific sites [here](https://fragalysis.diamond.ac.uk/viewer/react/preview/target/Mpro)

3D coordinates of all the ligands (need to filter to get the relevant ones) can be found [here](https://www.diamond.ac.uk/covid-19/for-scientists/Main-protease-structure-and-XChem/Downloads.html) in the .pdb files which are formatted like [this](https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/dealing-with-coordinates)

Background on the graph-based genetic algorithm (GB-GA) that I used can be found in [this paper](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c#!divAbstract) - I co-opted the GB-GA code from [this Github](https://github.com/jensengroup/GB-GA)

## Dependencies
- [dscribe](https://singroup.github.io/dscribe/install.html)
- [RDKit](https://www.rdkit.org/docs/Install.html)
- pandas

## How to use
running `GA-soap.py` will start running a genetic algorithm.
