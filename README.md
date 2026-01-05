***EcoAgent-RL***

The core objective of this research is to evaluate and refine Geographic Profiling (GP)—a spatial statistical technique originally developed in criminology—by testing its efficacy on synthetic wildlife movement data generated through Reinforcement Learning (RL).

1. Multi-Taxa Synthetic Data Generation
The project aims to develop high-fidelity RL agents representing diverse ecological "profiles" (e.g., avian vs. terrestrial, specialist vs. generalist).

By simulating these agents across real Northern Ireland landscapes, we generate ground-truth datasets where the exact nest location and foraging logic are known.

This allows for a controlled comparison that is impossible with wild animals, where "true" movement intent is often hidden.

2. Efficacy Testing of Geographic Profiling Models
Using the synthetic sighting data, the project will stress-test current Bayesian GP algorithms (Dirochlet Process Mixed Models vs Poisson/Negative Binomial Methods) to determine:

Accuracy: How closely the predicted "anchor point" (nest) matches the agent's actual starting coordinate.

Sensitivity: How the quality of the profile degrades as the number of sightings decreases or the landscape becomes more fragmented.

Limitations: What are the inherent limitations of the current Bayesian GP models in handling complex ecological factors which impact accuracy.

3. Informing Field Study Design & Best Practices
A critical outcome of this work is to provide evidence-based guidance for ecologists on how to collect data for GP analysis. The project aims to answer:

Sampling Thresholds: What is the minimum number of sightings required for a reliable profile for a specific taxon?

Spatial Bias: How do landscape features (like barriers or corridors) distort GP predictions, and how can we correct for these distortions in study design?

4. Methodological Optimization
The project will investigate if current GP models—which often assume a simple distance-decay function—need to be adapted to account for complex ecological resistance and Central Place Foraging constraints identified during the RL simulation phase.

Research Value
By the end of this body of work, I will have created a framework that doesn't just map animals, but audits the very tools we use to find them, with the aim develop a new best in practice ecological framework for Geographic Profiling. This will lead to more robust conservation strategies and a standardized "best practice" manual for applying geographic profiling to wildlife biology.
