#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline  # Not sklearn.pipeline!
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneOut, StratifiedKFold , GridSearchCV

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical, Continuous
from sklearn_genetic.plots import plot_fitness_evolution
import matplotlib.pyplot as plt

import threading
import time
#psutil v7.0.0
import psutil
import os
from datetime import datetime


class ModelTrainer:
    def __init__(self, pipeline: Pipeline, verbose: bool = False, printer = print):
        """
        Parameters:
        - pipeline (Pipeline): sklearn pipeline with preprocessing + estimator
        - verbose (bool): Verbosity flag
        """
        self.pipeline = pipeline
        self.verbose = verbose
        self.printer = printer

    def train(self, X_train, y_train):
        if self.verbose:
            self.printer("[Trainer] Training model...")
        self.pipeline.fit(X_train, y_train)
        if self.verbose:
            self.printer("[Trainer] Training complete.")
        return self.pipeline
    
    def tune_hyperparameters_and_train_by_cv(self, X, y, param_grid, cv=5, scoring=None, n_jobs=-1):
        """
        Perform hyperparameter tuning using GridSearchCV.

        Parameters:
        - X: Features for training
        - y: Target labels
        - param_grid (dict): Dictionary with parameters names as keys and lists of parameter settings to try
        - cv (int): Number of cross-validation folds
        - scoring (str or callable): Scoring strategy
        - n_jobs (int): Number of jobs to run in parallel (-1 means using all processors)

        Returns:
        - best_estimator_: The best estimator found by GridSearchCV
        """
        if self.verbose:
            print("[Trainer] Starting hyperparameter tuning...")

        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=self.verbose
        )
        grid_search.fit(X, y)

        if self.verbose:
            self.printer(f"[Trainer] Best parameters found: {grid_search.best_params_}")
            self.printer("[Trainer] Hyperparameter tuning complete.")

        self.pipeline = grid_search.best_estimator_
        return self.pipeline
    
    def tune_hyperparameters_and_train_by_loo(self, X, y, param_grid, scoring=None, n_jobs=-1):
        """
        Perform hyperparameter tuning using Leave-One-Out Cross-Validation (LOOCV).

        Parameters:
        - X: Features for training
        - y: Target labels
        - param_grid (dict): Dictionary with parameter names as keys and lists of parameter settings to try
        - scoring (str or callable): Scoring strategy
        - n_jobs (int): Number of jobs to run in parallel (-1 means all processors)

        Returns:
        - best_estimator_: The best estimator found by GridSearchCV
        """
        if self.verbose:
            self.printer("[Trainer] Starting LOOCV hyperparameter tuning...")

        loo = LeaveOneOut()

        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            cv=loo,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=self.verbose
        )

        grid_search.fit(X, y)

        if self.verbose:
            self.printer(f"[Trainer] Best parameters found (LOOCV): {grid_search.best_params_}")
            self.printer("[Trainer] LOOCV hyperparameter tuning complete.")

        self.pipeline = grid_search.best_estimator_
        return self.pipeline

    def tune_hyperparameters_and_train_by_loo_and_ga(self, X, y, param_grid, scoring=None, random_state = None, n_jobs=-1):
        """
        Perform hyperparameter tuning using Leave-One-Out Cross-Validation (LOOCV)
        and a Genetic Algorithm.

        Parameters:
        - X: Features for training
        - y: Target labels
        - param_grid (dict): Dictionary with parameter names as keys and search space objects (Integer, Categorical, etc.)
        - scoring (str or callable): Scoring strategy
        - n_jobs (int): Number of jobs to run in parallel (-1 means all processors)

        Returns:
        - best_estimator_: The best estimator found by GASearchCV
        """
        if self.verbose:
            print("[Trainer] Starting LOOCV hyperparameter tuning with Genetic Algorithm...")

        loo = LeaveOneOut()

        ga_search = GASearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            cv=loo,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=self.verbose,
            population_size=20,
            generations=5,
            tournament_size=3,
            mutation_probability=0.1,
            crossover_probability=0.5
        )

    
    # def train_by_grid_search_cv(self, X, y, param_grid, scoring=None, random_state = 42, n_jobs=-1):
    #     """
    #     Perform hyperparameter tuning using Leave-One-Out Cross-Validation (LOOCV).

    #     Parameters:
    #     - X: Features for training
    #     - y: Target labels
    #     - param_grid (dict): Dictionary with parameter names as keys and lists of parameter settings to try
    #     - scoring (str or callable): Scoring strategy
    #     - n_jobs (int): Number of jobs to run in parallel (-1 means all processors)

    #     Returns:
    #     - best_estimator_: The best estimator found by GridSearchCV
    #     """
    #     num_folder = 3 
    #     repeated_cross_val = True
    #     if repeated_cross_val == False:
    #         cv = num_folder
    #         kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    #     if repeated_cross_val == True:
    #         kf = RepeatedStratifiedKFold(
    #             n_splits=num_folder,          # your fold count
    #             n_repeats=1,        # how many times to repeat
    #             random_state=random_state
    #         )

    #     if self.verbose:
    #         print(f"[Trainer] Starting {num_folder}-Fold CV hyperparameter tuning...")
            
    #     grid_search = GridSearchCV(
    #         estimator=self.pipeline,
    #         param_grid=param_grid,
    #         cv=kf,
    #         scoring=scoring,
    #         n_jobs=n_jobs,
    #         verbose=2,
    #         refit=True                # refit on all data using the best params
    #     )

    #     grid_search.fit(X, y)

    #     if self.verbose:
    #         self.printer(f"[Trainer] Best parameters found ({num_folder}-Fold CV): {grid_search.best_params_}")
    #         self.printer(f"[Trainer] {num_folder}-Fold CV hyperparameter tuning complete.")

    #     self.pipeline = grid_search.best_estimator_
    #     return self.pipeline

    def train_by_grid_search_cv(self, X, y, param_grid, scoring=None, random_state = 42, n_jobs=-1):
        """
        Perform hyperparameter tuning using Leave-One-Out Cross-Validation (LOOCV).

        Parameters:
        - X: Features for training
        - y: Target labels
        - param_grid (dict): Dictionary with parameter names as keys and lists of parameter settings to try
        - scoring (str or callable): Scoring strategy
        - n_jobs (int): Number of jobs to run in parallel (-1 means all processors)

        Returns:
        - best_estimator_: The best estimator found by GridSearchCV
        """
            
        # Assume param_grid is a dict or list of dicts, like in GridSearchCV
        first_params = (
            param_grid[0] if isinstance(param_grid, list) else {k: v[0] for k, v in param_grid.items()}
        )
        # Apply the parameters to the pipeline
        # Fit the pipeline directly
        self.pipeline.fit(X, y)

        if self.verbose:
            self.printer(f"[Trainer] Best parameters used: {first_params}")
            self.printer(f"[Trainer] Random Seed {random_state} Training complete.")

        return self.pipeline


    # def train_by_grid_search_cv(self, X, y, param_grid, scoring=None, random_state=42, n_jobs=-1):
    #     from .mpi_search import train_by_grid_search_cv_mpi
    #     # Ignore n_jobs under MPI; every rank runs single-threaded
    #     pipeline = train_by_grid_search_cv_mpi(
    #         self, X, y, param_grid, scoring=scoring,
    #         random_state=random_state, n_splits=3, n_repeats=100,
    #         repeated=True, verbose=True
    #     )
    #     return pipeline



        # def monitor_cpu_and_memory(interval=60.0, duration=3600, log_path="monitor.log"):
        #     with open(log_path, "w") as log:
        #         log.write("[Monitor] Starting system monitoring...\n")
        #         log.flush()
        #         start_time = time.time()
        #         while time.time() - start_time < duration:
        #             cpu = psutil.cpu_percent(interval=0)  # Non-blocking
        #             mem = psutil.virtual_memory().percent
        #             log.write(f"[Monitor] CPU: {cpu}% | RAM: {mem}%\n")
        #             log.flush()
        #             time.sleep(interval)
        #         log.write("[Monitor] Monitoring finished.\n")
        #         log.flush()

        # # Start monitoring in a separate thread
        # monitor_thread = threading.Thread(
        #     target=monitor_cpu_and_memory,
        #     args=(1.0, 60, "monitor.log")
        # )
        # monitor_thread.start()

    # def train_by_genetic_opt_cv(self, X, y, param_grid, scoring=None, random_state = 42, n_jobs=-1, save_path = "Unknown_path"):
    #     """
    #     Perform hyperparameter tuning using Leave-One-Out Cross-Validation (LOOCV).

    #     Parameters:
    #     - X: Features for training
    #     - y: Target labels
    #     - param_grid (dict): Dictionary with parameter names as keys and lists of parameter settings to try
    #     - scoring (str or callable): Scoring strategy
    #     - n_jobs (int): Number of jobs to run in parallel (-1 means all processors)

    #     Returns:
    #     - best_estimator_: The best estimator found by GridSearchCV
    #     """

    #     if self.verbose:
    #         self.printer("[Trainer] Starting Seven-Fold CV hyperparameter tuning...")
    #         print("[Trainer] GASearchCV log path: ", save_path)

    #     kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    #     gas = GASearchCV(
    #         estimator=self.pipeline,
    #         param_grid=param_grid,
    #         cv=kf,
    #         scoring=scoring,
    #         population_size=500,
    #         generations=10,
    #         n_jobs=n_jobs,
    #         #verbose=self.verbose
    #         verbose = 2,
    #         criteria = 'max',
    #         algorithm = 'eaMuCommaLambda'
    #     )
    #     # Total model evaluations = 20 (population) × 10 (generations) = 200 fits
    #     # Compare this to the GridSearchCV which did: 96 candidates × 8 folds = 768 fits
    #     # Genetic search runs ~3.8x fewer fits than GridSearchCV in this case.
    #     # population_size=20	Large enough to cover diverse areas of the space; small enough to be fast
    #     # generations=10	Enough iterations for evolution to refine good solutions

    #     gas.fit(X, y)
    #     # Plot fitness over generations
    #     # Plot without showing


    #     plot_fitness_evolution(gas)

    #     # Build path
    #     saved_figure_path = os.path.join(save_path, "figure", "gen_opt")
    #     os.makedirs(saved_figure_path, exist_ok=True)

    #     # Save the current figure
    #     plt.savefig(os.path.join(saved_figure_path, "gen_opt.png"), dpi=600, bbox_inches="tight")
    #     plt.close()

    #     if self.verbose:
    #         self.printer(f"[Trainer] Best parameters found (Seven-Fold CV, Genetic Opt): {gas.best_params_}")
    #         self.printer("[Trainer] Seven-Fold CV hyperparameter tuning complete.")

    #     self.pipeline = gas.best_estimator_
        
    #     return self.pipeline



        # def monitor_cpu_and_memory(interval=60.0, duration=3600, log_path="monitor.log"):
        #     with open(log_path, "w") as log:
        #         log.write("[Monitor] Starting system monitoring...\n")
        #         log.flush()
        #         start_time = time.time()
        #         while time.time() - start_time < duration:
        #             cpu = psutil.cpu_percent(interval=0)  # Non-blocking
        #             mem = psutil.virtual_memory().percent
        #             log.write(f"[Monitor] CPU: {cpu}% | RAM: {mem}%\n")
        #             log.flush()
        #             time.sleep(interval)
        #         log.write("[Monitor] Monitoring finished.\n")
        #         log.flush()

        # # Start monitoring in a separate thread
        # monitor_thread = threading.Thread(
        #     target=monitor_cpu_and_memory,
        #     args=(1.0, 60, "monitor.log")
        # )
        # monitor_thread.start()



if __name__ == "__main__":
    from ml_clinical_act_jmap.ml_clinical_act.ml_clinical.data_importer import DataImporter
    from ml_clinical_act_jmap.ml_clinical_act.ml_clinical.data_cleaner import DataCleaner
    from ml_clinical_act_jmap.ml_clinical_act.ml_clinical.data_labeler import DataLabeler
    from ml_clinical_act_jmap.ml_clinical_act.ml_clinical.data_spiltter import DataSpiltter
    from ml_clinical_act_jmap.ml_clinical_act.ml_clinical.feature_selector import FeatureSelector
    from ml_clinical_act_jmap.ml_clinical_act.ml_clinical.feature_type_identifier import FeatureTypeIdentifier

    from ml_clinical_act_jmap.ml_clinical_act.ml_clinical.preprocess_wrapper import PreprocessWrapper
    #from umapTransformer import UMAPTransformer

    from ml_clinical_trial.model_evaluator import ModelEvaluator
    from sklearn.ensemble import RandomForestClassifier

    from imblearn.over_sampling import SMOTE
    from sklearn.decomposition import PCA
    
    import logging
    import traceback

    import random
    from deap import tools
    

    relative_path = 'data'
    random_state = 42
    random.seed(random_state) 


    importer = DataImporter(relative_path)

    cleaner = DataCleaner(importer, relative_path, versbose=False)
    subject_ids = cleaner.get_cleaned_subject_ids_by_labeler()
    #subject_ids = cleaner.clean_data_by_site_and_group(subject_ids)
        
    labeler = DataLabeler(importer, subject_ids, relative_path)
    responder_df_origin = labeler.label_data(label_columns=['total_womac', 'total_womac_v7'],
                                        responder_criteria='above_median_decrease')
    responder_df = responder_df_origin.dropna(subset=['responder'])

    # BAT+tDCS
    subject_ids = cleaner.clean_data_by_site_and_group(subject_ids)
    if len(subject_ids) == 0:
        raise ValueError("No subjects left after cleaning by site and group. Please check the data.")
    responder_df = responder_df.loc[subject_ids]



    splitter = DataSpiltter(importer, responder_df, relative_path)
    train_df, test_df = splitter.split_data_by_site()
    splitter.visualize_distribution(train_df, test_df)

    print(f"head of training set:\n{train_df.head()}")
    print(f"head of testing set:\n{test_df.head()}")

    feature_selector = FeatureSelector(relative_path)
    features = feature_selector.select_features(visit_times=['Visit 1', 'Visit 2'])

    feature_type_identifier = FeatureTypeIdentifier( relative_path)
    numerical_features, categorical_features = feature_type_identifier.get_feature_type_lists(features)

    X = importer.get_filtered_data(subject_ids, features)

    def prepare_train_test_data(X, train_df, test_df):
        """
        Splits the X dataframe into training and test features and labels based on train_df and test_df.
        
        Returns:
            X_train (DataFrame): Training features
            X_test (DataFrame): Testing features
            y_train (Series): Training labels
            y_test (Series): Testing labels
        """
        # Ensure subject_id is the index
        X = X.copy()
        train_df = train_df.copy()
        test_df = test_df.copy()

        # Align indices and extract data
        X_train = X.loc[train_df.index]
        y_train = train_df['responder']

        X_test = X.loc[test_df.index]
        y_test = test_df['responder']

        return X_train, X_test, y_train, y_test
    
    X_train, X_test, y_train, y_test = prepare_train_test_data(X, train_df, test_df)

    # Create pipeline
    pipeline = Pipeline([
        ('preprocess', PreprocessWrapper( features, numerical_features, categorical_features, verbose=False)),
        ('smote', SMOTE(sampling_strategy=1, random_state=random_state)),  # Oversampling stage
        ('PCA', PCA(n_components=0.95, random_state=random_state)),  # PCA for dimensionality reduction')
        #('umap', UMAPTransformer(n_components=10, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=random_state)),
        ('clf', RandomForestClassifier(random_state=random_state, class_weight='balanced')),
    ])

    # Train model
    logging.basicConfig(level=logging.INFO)

    try:
        logging.info("[Trainer] Initializing ModelTrainer...")
        trainer = ModelTrainer(pipeline, verbose=True)

        # Step 1: Fine-tune hyperparameters
        logging.info("[Trainer] Starting hyperparameter tuning...")
        # param_grid = {
        #     'smote__k_neighbors': [ 4, 5, 6],
        #     'smote__sampling_strategy': [0.65, 0.75, 0.85, 0.95],
        #     # UMAP dimensionality & structure
        #     'umap__n_components': [15],
        #     'umap__n_neighbors': [2],
        #     'umap__min_dist': [0.01],
        #     #'umap__metric': ['canberra', 'hamming','minkowski'],
        #     'umap__metric': [ 'hamming'],
        #     'clf__n_estimators': [50],
        #     'clf__max_depth': [None],
        #     'clf__min_samples_split': [2],
        #     'clf__min_samples_leaf': [1]
        # }
        param_grid = {
            # SMOTE parameters
            'smote__k_neighbors': Integer(3, 5),

            # UMAP dimensionality & structure
            #'umap__n_components': Integer(5, 30),
            #'umap__n_neighbors': Integer(2,2),
            #'umap__min_dist': Continuous(0.001, 0.5),
            #'umap__metric': Categorical(['euclidean', 'manhattan', 'hamming', 'canberra', 'minkowski']),

            # RandomForestClassifier hyperparameters
            'clf__n_estimators': Integer(50, 200),
            'clf__max_depth': Integer(5, 50),
            'clf__min_samples_split': Integer(2, 5),
            'clf__min_samples_leaf': Integer(1, 5)
        }

        trainer.tune_hyperparameters_and_train_by_loo_and_ga(X_train, y_train, param_grid, scoring='balanced_accuracy', random_state=random_state, n_jobs=-1)
        logging.info("[Trainer] Hyperparameter tuning completed.")

        model_evaluator = ModelEvaluator( verbose=True)
        model_evaluator.evaluate(trainer.pipeline, X_test, y_test)

        # Step 2: (Optional) Re-train on full training data if needed
        logging.info("[Trainer] Starting final model training...")
        trained_pipeline = trainer.train(X_train, y_train)
        logging.info("[Trainer] Model training completed successfully.")

    except Exception as e:
        logging.error("[Trainer] An error occurred during model training or tuning:")
        logging.error(traceback.format_exc())


