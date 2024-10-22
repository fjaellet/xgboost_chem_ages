# xgboost_chem_ages

## Estimating spectroscopic stellar ages for APOGEE red-giant stars

![Age map](im/RZ_agecoloured2.png)

We estimate spectroscopic stellar ages for 179 247 red-giant stars from the APOGEE DR17 catalogue [(Abdurro'uf et al. 2022)](https://ui.adsabs.harvard.edu/abs/2022ApJS..259...35A/abstract) with a median statistical uncertainty of 1.0 Gyr. To this end, we use the supervised machine learning technique XGBoost [(Chen & Guestrin 2016)](https://ui.adsabs.harvard.edu/abs/2016arXiv160302754C/abstract), trained on a high-quality dataset of 3 060 red-giant and red-clump stars with asteroseismic ages observed by both APOGEE and Kepler [(Miglio et al. 2021)](https://ui.adsabs.harvard.edu/abs/2021A%26A...645A..85M/abstract). 

## Age catalogue

* [data/spec_ages_published.fits](data/spec_ages_published.fits): Catalogue of spectroscopic age estimates for APOGEE DR17 stars. Duplicates are cleaned. Columns (recommended ones highlighted):

- APOGEE_ID
- **spec_age_xgb**: XGBoost age in Gyr (using the default model described in the paper)
- spec_age_xgb_calib: calibrated age in Gyr (adding the small correction shown in Fig. 4, top panel)
- **spec_age_xgb_uncert**: age uncertainty in Gyr (based on Fig. 4, bottom panel)
- **spec_age_xgb_flag**: Human-readable warning flag for potentially problematic stars
- spec_age_xgb_quantilereg: XGBoost quantile regression age in Gyr (using xgboost version 2.0.0)
- spec_age_xgb_quantilereg_calib: calibrated quantile regression age in Gyr
- spec_age_xgb_quantilereg_sigl: quantile regression lower 1sigma age uncertainty in Gyr
- spec_age_xgb_quantilereg_sigu: quantile regression upper 1sigma age uncertainty in Gyr
- spec_age_xgb_quantilereg_flag: Human-readable warning flag for potentially problematic stars in the quantile regression case

## Jupyter notebooks

This repository contains the jupyter notebooks that allow you to reproduce the figures and analysis presented in [Anders, Gispert, Ratcliffe, et al. 2023, A&A, accepted)](https://arxiv.org/abs/2304.08276):

* [train_xgboost_miglio2021.ipynb](py/train_xgboost_miglio2021.ipynb): Creating the training set, running XGBoost, and predicting ages for the APOGEE DR17 data. Reproduces Figs. 1-4 in the paper.
* [train_xgboost_miglio2021_quantileregression.ipynb](py/train_xgboost_miglio2021_quantileregression.ipynb): Creating the training set, running XGBoost, and predicting ages for the APOGEE DR17 data. Reproduces Figs. 1-4 in the paper.
* [test_age_catalogues.ipynb](py/test_age_catalogues.ipynb): Comparing the age estimates with other independent age determinations (CoRoT, K2, TESS, open clusters, [C/N] calibrations, astroNN, StarHorse, ...). Reproduces Figs. 5, 6, & A.1 in the paper.
* [test_ages_science.ipynb](py/test_ages_science.ipynb): Testing the age estimates on some typical age-chemokinematics relations (Age-metallicity, age-[Mg/Fe] relations, radial abundance gradient as a function of age, age-velocity relation). Reproduces Figs. 7-13, A.2, A.3, B.1 in the paper.

Comments, questions, feedback are welcome: fanders[ät]icc.ub.edu
