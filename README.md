# Automatic-individual-animal-identification-hotspotter

## Package
1. pyhesaff
```
pip install pyhesaff==2.0.1
### works on MacOS
```

## Contribution
1. from [Erotemic](https://github.com/Erotemic/hotspotter)
   
     Hotspotter macOS Installer: https://www.dropbox.com/s/q0vzz3xnjbxhsda/hotspotter_installer_mac.dmg?dl=0 
2. from [ECE 18-7 in Seattle University](https://github.com/SU-ECE-18-7/hotspotter)
3. from [ECE 17-7 in Seattle University](https://github.com/SU-ECE-17-7/hotspotter)

## Public dataset
### [sealID](https://etsin.fairdata.fi/dataset/22b5191e-f24b-4457-93d3-95797c900fc0)
Reference:

Nepovinnykh, E., Eerola, T., Biard, V., Mutka, P., Niemi, M., Kunnasranta, M. and Kälviäinen, H., 2022. SealID: Saimaa ringed seal re-identification dataset. Sensors, 22(19), p.7602.

Nepovinnykh, E., Chelak, I., Eerola, T. and Kälviäinen, H., 2022. Norppa: Novel ringed seal re-identification by pelage pattern aggregation. arXiv preprint arXiv:2206.02498.

## Create a custom image dataset
1. Create a new custom image dataset by [create_new_db_step2.py](https://github.com/obaiga/Automatic-individual-animal-identification-hotspotter/blob/master/create_new_db_step2.py)
2. Use the Hotspotter application to query all images in the dataset: [Batch]-[Precompute queries]
3. Generate the similarity score matrix based on query results by [save_hsres_step3.py](https://github.com/obaiga/Automatic-individual-animal-identification-hotspotter/blob/master/save_hsres_step3.py)
