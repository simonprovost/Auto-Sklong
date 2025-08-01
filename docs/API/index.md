# :book: API Reference 
# :book: API Reference 

Welcome to the full API documentation of the `Auto-Sklong` AutoML toolbox. :toolbox:

!!! warning "Be aware!"
    Considering `Auto-Sklong` is based on [`GAMA`](https://openml-labs.github.io/gama/master/index.html#). 
    We frequently redirect users to the official `GAMA` documentation. In addition to the extensive 
    [`Sklong`](https://simonprovost.github.io/scikit-longitudinal/) official documentation. Therefore, be aware that the following documentation will be more brief due to the extensive documentation of 
    [`GAMA`](https://openml-labs.github.io/gama/master/index.html#) and 
    [`Sklong`](https://simonprovost.github.io/scikit-longitudinal/).

    As a result, by relying on those two, we greatly improves maintainability and consistency.
    However, visually speaking, you will need to switch between documents. As our contributor base grows, 
    we may consider including the full [`GAMA`](https://openml-labs.github.io/gama/master/index.html#) documentation as a subset to reduce back and forth.

    Nonetheless, [`Sklong`](https://simonprovost.github.io/scikit-longitudinal/) will remains a separate documentation as it is a different project. No `AutoML` related,
    but a `Longitudinal` machine learning library toolbox.

## :simple-databricks: AutoML Classes
- [Gama Longitudinal Classifier](Gama_Longitudinal_Classifier.md)
- [Gama Base (redirect to GAMA)](https://openml-labs.github.io/gama/master/api/index.html#gama)

## :simple-jfrogpipelines: Search Methods
- [Bayesian optimisation (BO)](search_methods/bayesian_optimisation.md)
- [Asynchronous Evolutionary Algorithm (Async EA) (redirect to GAMA)](https://openml-labs.github.io/gama/master/api/index.html#asyncea)
- [Asynchronous Successive Halving (ASHA) (redirect to GAMA)](https://openml-labs.github.io/gama/master/api/index.html#asynchronoussuccessivehalving)
- [Random Search (redirect to GAMA)](https://openml-labs.github.io/gama/master/api/index.html#randomsearch)


!!! tip "Were you looking to update/look-into the `Auto-Skong`'s search space?"
    We got you covered! Check out the [`search space`](../tutorials/search_space.md) page.