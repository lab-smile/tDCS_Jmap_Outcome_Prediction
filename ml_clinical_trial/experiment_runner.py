import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (
    train_test_split, KFold, RepeatedKFold, LeaveOneOut, StratifiedShuffleSplit
)
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator

from .ml_clinical_act_jmap.jmap_act_importer_ml import JmapActDataImporterML
from .ml_clinical_act_jmap.jmap_act_feature_type_identifier import JmapActFeatureTypeIdentifier

from .ml_clinical_act_jmap.ml_clinical_act.act_importer_ml import ActDataImporterML
from .ml_clinical_act_jmap.ml_clinical_act.act_data_cleaner import ActDataCleaner
from .ml_clinical_act_jmap.ml_clinical_act.act_feature_selector import ActFeatureSelector
from .ml_clinical_act_jmap.ml_clinical_act.act_feature_type_identifier import ActFeatureTypeIdentifier


from .ml_clinical_act_jmap.ml_clinical_act.ml_clinical.data_labeler import DataLabeler
from .ml_clinical_act_jmap.ml_clinical_act.ml_clinical.data_spiltter import DataSpiltter

from .ml_clinical_act_jmap.ml_clinical_act.ml_clinical.preprocess_wrapper import PreprocessWrapper
from .ml_clinical_act_jmap.jmap_act_preprocessWrapper import JmapACTPreprocessor
#from umapTransformer import UMAPTransformer

from .model_evaluator import ModelEvaluator
from .model_dataset_preparator import ModelDatasetPreparator

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline  # Not sklearn.pipeline!

from sklearn_genetic.space import Integer, Categorical, Continuous
from sklearn.metrics import log_loss, make_scorer




# system and debugging
import warnings
import traceback
from datetime import datetime
import random
import os
from deap import tools

class ExperimentRunner:
    def __init__(self,
                 experiment_name: str,
                 verbose: bool = False):
        """
        Supports: holdout, repeated_holdout, cross_validation, repeated_cv, leave_one_out
        """
        self.experiment_name = experiment_name
 
        self.verbose = verbose


        # Get current date and time
        now = datetime.now()
        # Format as "year_month_date_hour_minute_second"
        formatted_datetime = now.strftime("%Y_%m_%d_%H_%M_%S")
        
        experiment_dir = "./results_" + formatted_datetime
        log_path = os.path.join(experiment_dir, f'output_{formatted_datetime}.log')
        
        if experiment_dir and not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir, exist_ok=True)

        def printer(log_str):
            with open(log_path, "a") as f:
                f.write(log_str)

        self.experiment_dir = experiment_dir
        self.printer = printer

    #############################
    # model training
    #############################
    def train_by_loo_and_ga(
        self,
        X_train, 
        X_test, 
        y_train, 
        y_test,
        pipeline,
        param_grid,
        printer,
        random_state):

        """
        Using neg_log_loss as the scoring metric is a robust choice, 
        especially in Leave-One-Out Cross-Validation (LOOCV) scenarios 
        or when dealing with imbalanced datasets. 
        Unlike metrics such as F1 score or balanced accuracy, 
        which can produce undefined results 
        when only one class is present in a fold, 
        log loss remains valid as long as the model outputs class probabilities. 
        neg_log_loss evaluates not just whether a prediction is correct, 
        but also how confident the model is in its predictions—penalizing overconfident 
        wrong predictions more heavily. 
        This makes it a more informative and stable metric in LOOCV, 
        where each test set consists of a single sample, 
        and ensures that the model is calibrated in addition to being accurate.
        """
        """
        In LOOCV, each validation set has exactly one sample. 
        Following score causes issues for metrics that assume multiple classes, like:
        f1_score
        balanced_accuracy
        confusion_matrix
        classification_report
        """
        """
        'accuracy'
        Simple alternative and works with binary/multiclass, even in small samples.
        Returns 1 if the prediction matches the actual label, 0 otherwise.
        Safe for LOOCV because each fold only evaluates on one sample.
        """
        def custom_log_loss_func(y_true, y_pred_proba, **kwargs):
            return log_loss(y_true, y_pred_proba, labels=[0, 1])
        custom_log_loss = make_scorer(
            custom_log_loss_func,
            greater_is_better=False,
            needs_proba=True
        )
        n_jobs = -1

        # Train model
        #try:
        printer("[Trainer] Initializing ModelTrainer...\n")
        trainer = ModelTrainer(pipeline, verbose=True)
        # Step 1: Fine-tune hyperparameters
        printer("[Trainer] Starting hyperparameter tuning...\n")
        trainer.tune_hyperparameters_and_train_by_loo_and_ga(X_train, 
                                                            y_train, 
                                                            param_grid, 
                                                            scoring=custom_log_loss, 
                                                            random_state=random_state, 
                                                            n_jobs=n_jobs)
        """
        LOOCV means Leave-One-Out Cross-Validation, 
        a strict way to evaluate model performance by training on all data except one sample and testing on that single sample, repeated for every sample.
        
        Genetic Algorithm (GA) is being used here to tune your model’s hyperparameters.
        The algorithm evolves a population of candidate solutions over generations (gen), trying to  maximize some fitness metric (here, neg log loss) or minimize log loss.
        
        
        gen	Generation number in the GA process (iteration count).
        nevals	Number of evaluations performed in this generation (number of candidate solutions tested).
        fitness	Mean fitness score of all candidates in this generation. Here, negative log loss (since it's negative).
        fitness_std	Standard deviation of the fitness scores across candidates — how spread out the scores are.
        fitness_max	The best (highest) fitness score in the population for that generation.
        fitness_min	The worst (lowest) fitness score in the population for that generation.
        """
        printer("[Trainer] Hyperparameter tuning completed.\n")
        return trainer.pipeline, X_test, y_test
        
        # Step 2: (Optional) Re-train on full training data if needed
        # printer("[Trainer] Starting final model training...")
        # trained_pipeline = trainer.train(X_train, y_train)
        # printer("[Trainer] Model training completed successfully.")

        # except Exception as e:
        #     print("[Trainer] An error occurred during model training or tuning:\n")
        #     print(traceback.format_exc())
        #     print('\n')

    def train_by_grid_search_cv(
        self,
        X_train, 
        X_test, 
        y_train, 
        y_test,
        pipeline,
        param_grid,
        printer,
        random_state):

        scoring = 'balanced_accuracy'
        n_jobs = -1

        # Train model
        #try:
        printer("[Trainer] Initializing ModelTrainer...\n")
        trainer = ModelTrainer(pipeline, verbose=True)
        printer("[Trainer] Starting hyperparameter tuning...\n")
        trainer.train_by_grid_search_cv(X_train, 
                                        y_train,
                                        param_grid, 
                                        scoring=scoring, 
                                        random_state=random_state, 
                                        n_jobs=n_jobs)
        printer("[Trainer] Hyperparameter tuning completed.\n")
        return trainer.pipeline, X_test, y_test

        # except Exception as e:
        #     print("[Trainer] An error occurred during model training or tuning:\n")
        #     print(traceback.format_exc())
        #     print('\n')

    def train_by_genetic_opt_cv(
        self,
        X_train, 
        X_test, 
        y_train, 
        y_test,
        pipeline,
        param_grid,
        printer,
        random_state,
        experiment_dir):

        scoring = 'balanced_accuracy'
        n_jobs = -1

        # Train model
        #try:
        printer("[Trainer] Initializing ModelTrainer...\n")
        trainer = ModelTrainer(pipeline, verbose=True, printer = printer)
        printer("[Trainer] Starting hyperparameter tuning...\n")
        trainer.train_by_genetic_opt_cv(X_train, 
                                        y_train, 
                                        param_grid, 
                                        scoring=scoring, 
                                        random_state=random_state, 
                                        n_jobs=n_jobs,
                                        save_path = experiment_dir)
        printer("[Trainer] Hyperparameter tuning completed.\n")
        return trainer.pipeline, X_test, y_test

        # except Exception as e:
        #     raise ValueError("[Trainer] An error occurred during model training or tuning:\n")
        #     raise ValueError(traceback.format_exc())
        #     raise ValueError('\n')
            
    
        

if __name__ == "__main__":
    # validation_strategy = 'cross_validation_genetic_opt_in_jmap'
    # random_state = 42
    # random.seed(random_state) 

    # # Create pipeline
    # def pipeline_generator(features, numerical_features, categorical_features, jmap_features, random_state):
    #     pipeline = Pipeline([
    #         ("jmap", JmapACTPreprocessor(jmap_features=["jmap_tp1"],     # columns containing your volumes
    #                                     strategy="pca",             # "stats", "flatten", or "pca"
    #                                     n_components=8,                # used only if strategy="pca"
    #                                     keep_channel_axis=True,         # if your data are 4D (X,Y,Z,C)
    #                                     random_state=random_state,
    #                                     atlas_path="../hammers_atlas/Hammers_mith_atlas_n30r83_SPM5.nii.gz",        # optional for region mapping
    #                                     atlas_labels_path="../hammers_atlas/n30r83_id2name_clean.txt",
    #                                     scale_volume=True               # <- always StandardScale the features
    #                                 )),
    #         ('smote', SMOTE(sampling_strategy=1.0, random_state=random_state)),  # Oversampling stage
    #         #('PCA', PCA(n_components=0.95, random_state=random_state)),  # PCA for dimensionality reduction'),
    #         #('umap', UMAPTransformer(n_components=10, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=random_state)),
    #         ('clf', RandomForestClassifier(random_state=random_state, class_weight='balanced')),
    #     ])
    #     return pipeline
        

    # # param_grid = {
    # #     'smote__sampling_strategy': Continuous(0.5, 1.0),  # continuous float between 0.5 and 1.0
    # #     'clf__n_estimators': Integer(100, 200),            # integer range from 100 to 200
    # #     'clf__max_depth': Categorical([None, 10, 20]),     # discrete choices
    # #     'clf__min_samples_split': Integer(2, 5),           # integer from 2 to 5
    # #     'clf__min_samples_leaf': Integer(1, 2),            # integer from 1 to 2
    # #     'clf__max_features': Categorical(['sqrt', 'log2']) # discrete choices
    # # }
    # param_grid = {
    #     'smote__sampling_strategy': [1.0],  # Try under- and full-oversampling
    #     #'clf__n_estimators': [100, 200],
    #     #'clf__max_depth': [None, 10, 20],
    #     #'clf__min_samples_split': [2, 5],
    #     #'clf__min_samples_leaf': [1, 2],
    #     'clf__max_features': ['sqrt', 'log2'],
    # }

    # runner = ExperimentRunner(
    #     experiment_name = 'rf_grid_search_cv_jmap',
    #     verbose = True
    # )

    # relative_path = '../../../data_generation_log/act_data'
    
    # runner.run(
    #     validation_strategy,
    #     pipeline_generator,
    #     param_grid,
    #     relative_path,
    #     file_name = 'act_data_generated.csv',
    #     dict_filename='act_data_dict_generated.csv',
    #     target_feature = 'stai_state_score',
    #     responder_criteria = 'decrease',
    #     group_var_name = 'Group_tp0',
    #     group_value = [3, 4],
    #     visit_times = ['0', '1'],
    #     random_state = random_state,
    #     num_training_repetition = 20
    # )
    
#     validation_strategy = 'cross_validation_genetic_opt'
#     random_state = 42
#     random.seed(random_state) 

#     # Create pipeline
#     def pipeline_generator(features, numerical_features, categorical_features):
#         pipeline = Pipeline([
#             ('preprocess', PreprocessWrapper( features, numerical_features, categorical_features, verbose=False)),
#             ('smote', SMOTE(sampling_strategy=1.0, random_state=random_state)),  # Oversampling stage
#             #('PCA', PCA(n_components=0.95, random_state=random_state)),  # PCA for dimensionality reduction'),
#             #('umap', UMAPTransformer(n_components=10, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=random_state)),
#             ('clf', RandomForestClassifier(random_state=random_state, class_weight='balanced')),
#         ])
#         return pipeline

#     param_grid = {
#         'smote__sampling_strategy': Continuous(0.5, 1.0),  # continuous float between 0.5 and 1.0
#         'clf__n_estimators': Integer(100, 200),            # integer range from 100 to 200
#         'clf__max_depth': Categorical([None, 10, 20]),     # discrete choices
#         'clf__min_samples_split': Integer(2, 5),           # integer from 2 to 5
#         'clf__min_samples_leaf': Integer(1, 2),            # integer from 1 to 2
#         'clf__max_features': Categorical(['sqrt', 'log2']) # discrete choices
#     }

#     runner = ExperimentRunner(
#         experiment_name = 'rf_grid_search_cv',
#         verbose = True
#     )

#     runner.run(
#         validation_strategy,
#         pipeline_generator,
#         param_grid,
#         relative_path = '../data_generation_log/act_data',
#         file_name = 'act_data_generated.csv',
#         dict_filename='act_data_dict_generated.csv',
#         target_feature = 'stai_state_score',
#         responder_criteria = 'decrease',
#         group_var_name = 'Group_tp0',
#         group_value = [3, 4],
#         visit_times = ['0', '1'],
#         random_state = random_state,
#         num_training_repetition = 20
#     )

    validation_strategy = 'cross_validation_genetic_opt'
    random_state = 42
    random.seed(random_state) 


    # Create pipeline
    def pipeline_generator(features, numerical_features, categorical_features, random_state = 42):
        pipeline = Pipeline([
            ('preprocess', PreprocessWrapper( features, numerical_features, categorical_features, verbose=False)),
            ('smote', SMOTE(sampling_strategy=1.0, random_state=random_state)),  # Oversampling stage
            #('PCA', PCA(n_components=0.95, random_state=random_state)),  # PCA for dimensionality reduction'),
            #('umap', UMAPTransformer(n_components=10, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=random_state)),
            ('clf', RandomForestClassifier(random_state=random_state, class_weight='balanced')),
        ])
        return pipeline

    param_grid = {
        'smote__sampling_strategy': [1.0],  # Try under- and full-oversampling
        'clf__n_estimators': [100],
    #    'clf__max_depth': [None, 10, 20],
    #    'clf__min_samples_split': [2, 5],
    #    'clf__min_samples_leaf': [1, 2],
        'clf__max_features': [None],
    }

    runner = ExperimentRunner(
        experiment_name = 'rf_grid_search_cv',
        verbose = True
    )

    runner.run(
        validation_strategy,
        pipeline_generator,
        param_grid,
        relative_path = '../../../data_generation_log/act_data',
        file_name = 'act_data_generated.csv',
        dict_filename='act_data_dict_generated.csv',
        target_feature = 'stai_state_score',
        responder_criteria = 'decrease',
        group_var_name = 'Group_tp0',
        group_value = [3, 4],
        visit_times = ['0', '1'],
        random_state = random_state,
        num_training_repetition = 2000
    )



    # random_state = 42
    # validation_strategy = 'cross_site_validation'
    # random.seed(random_state) 

    # # Create pipeline
    # def pipeline_generator(features, numerical_features, categorical_features):
    #     pipeline = Pipeline([
    #         ('preprocess', PreprocessWrapper( features, numerical_features, categorical_features, verbose=False)),
    #         ('smote', SMOTE(sampling_strategy=1.0, random_state=random_state)),  # Oversampling stage
    #         #('PCA', PCA(n_components=0.95, random_state=random_state)),  # PCA for dimensionality reduction'),
    #         #('umap', UMAPTransformer(n_components=10, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=random_state)),
    #         ('clf', RandomForestClassifier(random_state=random_state, class_weight='balanced')),
    #     ])
    #     return pipeline

    # # param_grid = {
    # #     'smote__k_neighbors': [ 4, 5, 6],
    # #     'smote__sampling_strategy': [0.65, 0.75, 0.85, 0.95],
    # #     # UMAP dimensionality & structure
    # #     'umap__n_components': [15],
    # #     'umap__n_neighbors': [2],
    # #     'umap__min_dist': [0.01],
    # #     #'umap__metric': ['canberra', 'hamming','minkowski'],
    # #     'umap__metric': [ 'hamming'],
    # #     'clf__n_estimators': [50],
    # #     'clf__max_depth': [None],
    # #     'clf__min_samples_split': [2],
    # #     'clf__min_samples_leaf': [1]
    # # }
    # param_grid = {
    #     # SMOTE parameters
    #     'smote__k_neighbors': Integer(3, 5),

    #     # UMAP dimensionality & structure
    #     #'umap__n_components': Integer(5, 30),
    #     #'umap__n_neighbors': Integer(2,2),
    #     #'umap__min_dist': Continuous(0.001, 0.5),
    #     #'umap__metric': Categorical(['euclidean', 'manhattan', 'hamming', 'canberra', 'minkowski']),

    #     # RandomForestClassifier hyperparameters
    #     'clf__n_estimators': Integer(50, 200),
    #     'clf__max_depth': Integer(5, 50),
    #     'clf__min_samples_split': Integer(2, 5),
    #     'clf__min_samples_leaf': Integer(1, 5)
    # }

    # runner = ExperimentRunner(
    #     experiment_name = 'rf_loocv_ga',
    #     verbose = True
    # )

    # runner.run(
    #     validation_strategy,
    #     pipeline_generator,
    #     param_grid,
    #     relative_path = '../data_generation_log/act_data',
    #     file_name = 'act_data_generated.csv',
    #     dict_filename='act_data_dict_generated.csv',
    #     target_feature = 'stai_state_score',
    #     responder_criteria = 'decrease',
    #     group_var_name = 'Group_tp0',
    #     group_value = [3, 4],
    #     visit_times = ['0', '1'],
    #     random_state = random_state,
    #     num_training_repetition = 20
    # )