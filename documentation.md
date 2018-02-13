## TractaML v1.0 - Documentation

### Index:

- Module 1 - "model_optimizer"

    - Background
    - Usage
    - Class methods
    - References
      
_____________________________


### Module 1 – “model_optimizer”

#### Background

While tuning model hyper-parameters, ML practitioners typically tend to use grid-search on a selected set of features through some cross-validation scheme. The problem with this method is that we are tuning the model for a pre-selected set of features. The model that we used for selecting these features, and thus the features themselves, might not be appropriate now. If we tune the model first and then apply feature selection, we are selecting features according to one model. So, different hyper-parameters settings would favor different feature subsets. In such a symbiotic relationship, the right solution would be to combine hyper-parameter optimization and feature selection. If viewed from a grid-search perspective, this problem would be intractable. Thus, we have to depend on a search heuristic.

Genetic algorithms (here we refer to standard-GAs or SGAs) are one of the most promising meta-search heuristics for global optimization which can efficiently explore vast solution spaces to find near-optimal or even the most optimal solution. However, their usage in machine learning has been limited due to the number of settings required to be tuned by the user (E.g., type of mutation/crossover, population size, crossover rate, mutation rate, etc.). Mutation is an important operator to explore new solution sub-spaces and crossover is essential in combining know good solutions to arrive at a better solution. Thus, tuning their behavior is vital for the optimization to work. An un-tuned algorithm might get stuck in a local optimum or might take infinite time to converge.

Micro-genetic algorithms (Micro-GAs) are variants of SGAs where the population size is kept low. On population convergence/loss of genetic diversity, the best solution is kept and the rest of the population is replaced with randomly generated individuals. The optimization restarts and this process would continue till we hit a near-optimal/optimal solution. In contrast to SGAs, a micro-GA has very limited parameters to tune. For example, in an SGA, mutation is required to maintain genetic diversity in the population, since loss of diversity might give rise to a local optimum. In micro-GAs, this problem is circumvented by restarting the search with new individuals whenever there is loss of genetic diversity, negating the need for a mutation operator. Due to a low size, micro-GA populations converge more quickly and the evolutionary process restarts more often. This enables the algorithm to rapidly converge to a near-optimal region. The self-adaptive capabilities of one-point and k-point crossovers are explored in [2]. The crossover operator used in this module is a combination of inter-crossover and uniform crossover, both useful variants of the standard k-point crossover. Thus, the user does not have to specify the type or rate of crossover to be used.

The author would direct the reader to [1] for more theoretical and implementation details regarding micro-GAs. This module is designed based on the method described for stationary optimization in [1]. The author also recommends [3] for understanding the usefulness of micro-GAs.

The author would like to specially thank Mr. Clinton Sheppard for his wonderful book on genetic algorithms [6]. Please refer it for more information on SGAs.

____________________

#### Usage 

The interface is very similar to that of scikit-learn. All modules are implemented as class objects.

```
class objective_ml.model_optimizer. ModelTuner(mod, param_dict, cv, score) 
```

Combined hyper-parameter optimization and feature selection for machine learning models

Parameters:

- mod: Scikit-learn estimator to be used.

- param_dict: Parameter dictionary defining the search space for hyper-parameters.

    Format:
    ```
    param_dict = { param_1: [‘int’, lower_limit, upper_limit],
		               param_2: [‘float’, lower_limit, upper_limit],
		               param_3: [‘list’, p1, p2, p3...]
		              }
    ```
    Note:

        -	p1, p2, p3... are elements of the list to be explored. 
        -	The limits are half open, meaning the exploration space will be: [lower-limit, upper_limit)

- cv: Cross validation scheme to be used. 

    You can pass a scikit-learn KFold cross-validation object or just a positive integer (k) for stratified K-fold cross-validation with no shuffling. In short, all conditions that apply to scikit-learn’s cross_validation function apply here as well.

- score: Metric to be optimized (maximized by default). See scikit-learn documentation for the available list of metrics. You could also pass your own scoring function that is compliant with scikit-learn. The user would have to write a custom function if the score is to be minimized (For example, multiplying by -1 and converting it to a maximization problem).

___________________

#### Class methods
```
1. fit(X, y, verbose, max_back, tot_gen, known_best)
```
   Run the optimization algorithm on the given data. Save best model and feature subset to current working directory. 

Parameters: 

- X: Pandas dataframe with feature names as column names, shape (n_samples, n_features)

- y: array-like, shape(n_samples,)
Target-values (class labels in classification, real numbers in regression) 

- verbose: boolean, optional (default=False). 
If set to True, verbose logs will be displayed for each iteration.

- max_back: float (0,1], optional (default=0.5). 
Algorithm will be terminated if in the last max_back*tot_gen iterations, no better solution (than the current best) was found.

- tot_gen: int, optional (default=2000) 
Upper limit on the total number of iterations/generations the micro-GA can proceed for. 
Depending on the accuracy & run-time requirements, the user can adjust the ‘max_back’ and ‘tot_gen’ parameters. But, the author would suggest using the default settings for best results. 

- known_best: Chromosome object loaded from file, optional (default=None)
Previously known best solution (if any). You can load this from the pickled ‘best_solution.pkl’ file in your current working directory.
######
```2. transform(X)```

Transforms the input dataframe so that it has just the best set of features

Parameters:
 
- X: Pandas dataframe with feature names as column names, shape (n_samples, n_features)
######
```3. load_file(file)```

Loads pickled files from disk

Parameters: 

- file: File (filename with path) to be loaded from disk.
######
```4.	get_best_model()```

Returns the best model found (as scikit-learn object)
######
```5.	plot_monitor(metric)```

Plots the optimization monitor for the chosen metric

Parameters: 
- metric: Metric to be plotted against the number of iterations. Can have one of the following values:

    i.	‘Model_Fitness’ (plots the cross validation score)

    ii.	‘Feature_Fitness’ (if feature fitness = 1/k (where k is some positive real number, then (100/k)% of features from the original input have been excluded)

    iii.	‘Stdev’ (plots the standard deviation in cross validation score, indicating model stability)

Note: The optimization process prefers a high scoring model (model fitness) with the most stability (low standard deviation) and a fewer number of features (high feature fitness)  <br/>

```6.	get_features()```

Returns a boolean array with ‘True’ indicating selected feature and ‘False’ indicating otherwise

_________________

### References

[1] K. Krishnakumar, "Micro-Genetic Algorithms For Stationary And Non-Stationary Function Optimization," in 1989 Symposium on Visual Communications, Image Processing, and Intelligent Robotics Systems, 1990, vol. 1196, p. 8: SPIE.

[2] S. Meyer-Nieberg and H.-G. Beyer, "Self-Adaptation in Evolutionary Algorithms," ed. Conference Proceedings: Parameter Setting in Evolutionary Algorithms, 2007

[3]  G. Alvarez, "Can we make genetic algorithms work in high-dimensionality problems?", Stanford Exploration Project (SEP) Report 112, pp. 195-212, 2002.

[4] G. I. Salama, m. b. Abdelhalim, and M. Zeid, Breast Cancer Diagnosis on Three Different Datasets using Multi-classifiers. 2012, pp. 36-43.

[5] Ahmet Mert, Niyazi Kılıç, Erdem Bilgili, and Aydin Akan, “Breast Cancer Detection with Reduced Feature Set,” Computational and Mathematical Methods in Medicine, vol. 2015, Article ID 265138, 11 pages, 2015. 

[6] Clinton, Sheppard. Genetic Algorithms with Python. CreateSpace Independent Publishing Platform, 2016.







