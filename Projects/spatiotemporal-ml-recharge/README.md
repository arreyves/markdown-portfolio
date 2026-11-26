# Spatio-Temporal Machine Learning for Land-Cover Classification & Groundwater Recharge Estimation

**Role:** Lead Computational Researcher & ML Modeler  
**Stack:** Python 3, Scikit-Learn, TensorFlow, GeoPandas, NumPy, Pandas, QGIS  
**Context:** University of Venda / Regional Environmental Monitoring Initiative  

## Problem Statement
Accurate groundwater recharge modeling in semi-arid, data-sparse catchments requires high-resolution land-use and land-cover (LULC) parameterization. Traditional manual classification of multi-spectral satellite imagery suffered from spectral confusion across heterogeneous terrain, introducing substantial bias into downstream hydrological mass-balance models.

## Technical Architecture & Methodology
* **Multi-Source Feature Engineering:** Integrated multi-temporal optical satellite bands with digital elevation derivatives (slope, aspect, topographic wetness index) and field-verified ground-truth polygons.
* **Leakage-Free Spatial Validation:** Implemented spatially blocked cross-validation rather than naive random train/test splits to eliminate spatial autocorrelation leakage and accurately measure out-of-sample generalization.
* **Supervised Classification & Ensemble Benchmarking:** Trained and evaluated Random Forest, Gradient Boosted Trees (`scikit-learn`), and neural network classifiers (`TensorFlow`) to segment complex land-cover classes under severe class imbalance.
* **Downstream Physical Integration:** Coupled classified spatial rasters directly into unsaturated-zone hydrological models to constrain surface runoff and infiltration boundaries.

## Key Outcomes & Impact
* **+25% Improvement in Recharge Estimation Accuracy:** Reduced structural error in catchment-scale groundwater recharge predictions compared to legacy static land-cover baselines.
* **Reproducible Geospatial Pipeline:** Automated raster ingestion, feature extraction, and accuracy assessment (confusion matrices, per-class F1 scores, Kappa statistics) into repeatable Python scripts.