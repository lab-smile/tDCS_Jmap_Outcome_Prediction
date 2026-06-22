# IMPORTANT: set these BEFORE importing numpy/sklearn anywhere in your program.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from .ml_clinical_act_jmap.ml_clinical_act.ml_clinical.preprocess_wrapper import PreprocessWrapper
#from umapTransformer import UMAPTransformer
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

import random


        
if __name__ == "__main__":
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    from sklearn.decomposition import PCA  # optional, commented below
    from sklearn.ensemble import RandomForestClassifier

    # local imports
    from .experiment_manager import ExperimentManager  # or: from .ml_clinical_trial.experiment_manager import ExperimentManager
    from .ml_clinical_act_jmap.ml_clinical_act.ml_clinical.preprocess_wrapper import PreprocessWrapper  # or: from .ml_clinical_act_jmap.ml_clinical_act.ml_clinical.preprocess_wrapper import PreprocessWrapper
    from .experiment_runner import ExperimentRunner     # your existing runner

    # # --- experiment setup (matches your previous script) ---
    # validation_strategy = "cross_validation_grid_search_opt"
    # random_state = 587
    # random.seed(random_state)

    # # Create pipeline factory
    # def pipeline_generator(features, numerical_features, categorical_features, random_state=42, *args, **kwargs):
    #     pipeline = Pipeline([
    #         ('preprocess', PreprocessWrapper(features, numerical_features, categorical_features, verbose=False)),
    #         ('smote', SMOTE(sampling_strategy=1.0, random_state=random_state)),
    #         # ('pca', PCA(n_components=0.95, random_state=random_state)),  # optional
    #         ('clf', RandomForestClassifier(random_state=random_state, class_weight='balanced')),
    #     ])
    #     return pipeline

    # # Hyperparameters to search
    # param_grid = {
    #     'smote__sampling_strategy': [1.0],
    #     'clf__n_estimators': [100],
    #     'clf__max_features': [None],
    #     # 'clf__max_depth': [None, 10, 20],
    #     # 'clf__min_samples_split': [2, 5],
    #     # 'clf__min_samples_leaf': [1, 2],
    # }

    # # Instantiate runner (data prep + low-level training live here)
    # runner = ExperimentRunner(
    #     experiment_name="rf_grid_search_cv",
    #     verbose=True
    # )

    # # Instantiate manager (or use runner.manager if you wired it in __init__)
    # manager = ExperimentManager(
    #     runner=runner,
    #     experiment_dir=runner.experiment_dir,  # reuse the directory runner created
    #     printer=runner.printer,
    #     verbose=True
    # )

    # # Kick off run (same args as before)
    # manager.run(
    #     validation_strategy=validation_strategy,
    #     pipeline_generator=pipeline_generator,
    #     param_grid=param_grid,
    #     relative_path='../../../data_generation_log/act_data',
    #     file_name='act_data_generated.csv',
    #     dict_filename='act_data_dict_generated.csv',
    #     target_feature='stai_state_score',
    #     responder_criteria='decrease',
    #     group_var_name='Group_tp0',
    #     group_value=[3, 4],
    #     visit_times=['0', '1'],
    #     random_state=random_state,
    #     num_training_repetition=1
    # )

    # # --- experiment setup (matches the previous script) ---
    # validation_strategy = "cross_validation_grid_search_opt"
    # random_state = 42
    # random.seed(random_state)

    # # Create pipeline factory
    # def pipeline_generator(features, numerical_features, categorical_features, random_state=42, *args, **kwargs):
    #     pipeline = Pipeline([
    #         ('preprocess', PreprocessWrapper(features, numerical_features, categorical_features, verbose=False)),
    #         ('smote', SMOTE(sampling_strategy=1.0, k_neighbors=1, random_state=random_state)),
    #         # ('pca', PCA(n_components=0.95, random_state=random_state)),  # optional
    #         ('clf', RandomForestClassifier(random_state=random_state, class_weight='balanced')),
    #     ])
    #     return pipeline

    # # Hyperparameters to search
    # param_grid = {
    #     'smote__sampling_strategy': [1.0],
    #     'clf__n_estimators': [100],
    #     'clf__max_features': [None],
    #     # 'clf__max_depth': [None, 10, 20],
    #     # 'clf__min_samples_split': [2, 5],
    #     # 'clf__min_samples_leaf': [1, 2],
    # }

    # # Instantiate runner (data prep + low-level training live here)
    # runner = ExperimentRunner(
    #     experiment_name="rf_grid_search_cv",
    #     verbose=True
    # )

    # # Instantiate manager (or use runner.manager if you wired it in __init__)
    # manager = ExperimentManager(
    #     runner=runner,
    #     experiment_dir=runner.experiment_dir,  # reuse the directory runner created
    #     printer=runner.printer,
    #     verbose=True
    # )

    # # Kick off run (same args as before)
    # manager.run(
    #     validation_strategy=validation_strategy,
    #     pipeline_generator=pipeline_generator,
    #     param_grid=param_grid,
    #     relative_path='../../../data_generation_log/act_data',
    #     file_name='act_data_generated.csv',
    #     dict_filename='act_data_dict_generated.csv',
    #     target_feature='stai_state_score',
    #     responder_criteria='above_median_decrease_in_severe',
    #     group_var_name='Group_tp0',
    #     group_value=[2, 4],
    #     visit_times=['0', '1'],
    #     random_state=random_state,
    #     num_training_repetition=1000
    # )

    # #Clinco-demo
    # # --- experiment setup (matches your previous script) ---
    # validation_strategy = "cross_validation_grid_search_opt"
    # random_state = 42
    # random.seed(random_state)

    # # Create pipeline factory
    # def pipeline_generator(features, numerical_features, categorical_features, random_state=42, *args, **kwargs):
    #     pipeline = Pipeline([
    #         ('preprocess', PreprocessWrapper(features, numerical_features, categorical_features, verbose=False)),
    #         ('smote', SMOTE(sampling_strategy=1.0, k_neighbors=1, random_state=random_state)),
    #         # ('pca', PCA(n_components=0.95, random_state=random_state)),  # optional
    #         ('clf', RandomForestClassifier(random_state=random_state, class_weight='balanced')),
    #     ])
    #     return pipeline

    # # Hyperparameters to search
    # param_grid = {
    #     'smote__sampling_strategy': [1.0],
    #     'clf__n_estimators': [100],
    #     'clf__max_features': [None],
    #     # 'clf__max_depth': [None, 10, 20],
    #     # 'clf__min_samples_split': [2, 5],
    #     # 'clf__min_samples_leaf': [1, 2],
    # }

    # # Instantiate runner (data prep + low-level training live here)
    # runner = ExperimentRunner(
    #     experiment_name="rf_grid_search_cv",
    #     verbose=True
    # )

    # # Instantiate manager (or use runner.manager if you wired it in __init__)
    # manager = ExperimentManager(
    #     runner=runner,
    #     experiment_dir=runner.experiment_dir,  # reuse the directory runner created
    #     printer=runner.printer,
    #     verbose=True
    # )

    # # Kick off run (same args as before)
    # manager.run(
    #     validation_strategy=validation_strategy,
    #     pipeline_generator=pipeline_generator,
    #     param_grid=param_grid,
    #     relative_path='../../../data_generation_log/act_data',
    #     file_name='act_data_generated.csv',
    #     dict_filename='act_data_dict_generated.csv',
    #     target_feature='stai_state_score',
    #     responder_criteria='above_median_decrease_in_severe',
    #     group_var_name='Group_tp0',
    #     group_value=[ 3, 4],
    #     visit_times=['0', '1'],
    #     random_state=random_state,
    #     num_training_repetition=2000
    # )











    
    # J-map
    # --- experiment setup (Jmap running in severe to moderate baseline state anxiety people) ---
    # from .ml_clinical_act_jmap.jmap_act_preprocessWrapper import JmapACTPreprocessor
    # validation_strategy = "cross_validation_grid_search_opt_in_jmap"
    # random_state = 42
    # random.seed(random_state)

    # # Create pipeline factory
    # def pipeline_generator(features, numerical_features, categorical_features, jmap_features, random_state=42, *args, **kwargs):
    #     pipeline = Pipeline([
    #         ("jmap", JmapACTPreprocessor(jmap_features=["jmap_tp1"],     # columns containing your volumes
    #                         strategy="pca",             # "stats", "flatten", or "pca"
    #                         n_components=8,                # used only if strategy="pca"
    #                         keep_channel_axis=True,         # if your data are 4D (X,Y,Z,C)
    #                         random_state=random_state,
    #                         atlas_path="../hammers_atlas/Hammers_mith_atlas_n30r83_SPM5.nii.gz",        # optional for region mapping
    #                         atlas_labels_path="../hammers_atlas/n30r83_id2name_clean.txt",
    #                         scale_volume=True               # <- always StandardScale the features
    #                     )),
    #         ('smote', SMOTE(sampling_strategy=1.0, k_neighbors=1, random_state=random_state)),
    #         # ('pca', PCA(n_components=0.95, random_state=random_state)),  # optional
    #         ('clf', RandomForestClassifier(random_state=random_state, class_weight='balanced')),
    #     ])
    #     return pipeline

    # # Hyperparameters to search
    # param_grid = {
    #     'smote__sampling_strategy': [1.0],
    #     'clf__n_estimators': [100],
    #     'clf__max_features': [None],
    #     # 'clf__max_depth': [None, 10, 20],
    #     # 'clf__min_samples_split': [2, 5],
    #     # 'clf__min_samples_leaf': [1, 2],
    # }

    # # Instantiate runner (data prep + low-level training live here)
    # runner = ExperimentRunner(
    #     experiment_name="rf_grid_search_cv",
    #     verbose=True
    # )

    # # Instantiate manager (or use runner.manager if you wired it in __init__)
    # manager = ExperimentManager(
    #     runner=runner,
    #     experiment_dir=runner.experiment_dir,  # reuse the directory runner created
    #     printer=runner.printer,
    #     verbose=True
    # )

    # # Kick off run (same args as before)
    # manager.run(
    #     validation_strategy=validation_strategy,
    #     pipeline_generator=pipeline_generator,
    #     param_grid=param_grid,
    #     relative_path='../../../data_generation_log/act_data',
    #     file_name='act_data_generated.csv',
    #     dict_filename='act_data_dict_generated.csv',
    #     target_feature='stai_state_score',
    #     responder_criteria='above_median_decrease_in_severe',
    #     group_var_name='Group_tp0',
    #     group_value=[2, 4],
    #     visit_times=['0', '1'],
    #     random_state=random_state,
    #     num_training_repetition=100
    # )

    # clean the data set by removing participants who were not in selected intervention group
    # Category of Intervention Groups:
    # 1 = Education Control Training + Sham tDCS; 
    # 2 = Education Control Training + tDCS; 
    # 3 = Cognitive Training + Sham tDCS; 
    # 4 = Cognitive Training + tDCS

    
    # # J-map
    # from sklearn.ensemble import AdaBoostClassifier
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.preprocessing import StandardScaler
    # from .ml_clinical_act_jmap.mrmr_selector import MRMRSelector  # <- the class above
    # from .ml_clinical_act_jmap.safe_smote import SafeSMOTE
    # # --- experiment setup (Jmap running in severe to moderate baseline state anxiety people) ---
    # from .ml_clinical_act_jmap.jmap_act_preprocessWrapper import JmapACTPreprocessor
    # validation_strategy = "cross_validation_grid_search_opt_in_jmap"
    # random_state = 42
    # random.seed(random_state)

    # # Create pipeline factory
    # def pipeline_generator(features, numerical_features, categorical_features, jmap_features, random_state=42, *args, **kwargs):
    #     pipeline = Pipeline([
    #         ("jmap", JmapACTPreprocessor(jmap_features=["jmap_tp1"],     # columns containing your volumes
    #                         strategy="flatten",             # "stats", "flatten", or "pca"
    #                         n_components=8,                # used only if strategy="pca"
    #                         keep_channel_axis=True,         # if your data are 4D (X,Y,Z,C)
    #                         random_state=random_state,
    #                         atlas_path="../hammers_atlas/Hammers_mith_atlas_n30r83_SPM5.nii.gz",        # optional for region mapping
    #                         atlas_labels_path="../hammers_atlas/n30r83_id2name_clean.txt",
    #                         scale_volume=True               # <- always StandardScale the features
    #                     )),
    #         ('scaler', StandardScaler()),
    #         ('pca', PCA(n_components=0.95, random_state=random_state)),
    #         # --- mRMR goes here (BEFORE SMOTE) ---
    #         ("mrmr", MRMRSelector(
    #             k=20,               # tune this in the grid below
    #             method="MIQ",        # or "MID"
    #             discretize="quantile",
    #             n_bins=5,
    #             random_state=random_state,
    #             dtype="float32"
    #         )),
            
    #         ('smote', SafeSMOTE(sampling_strategy=1.0, k_neighbors=1, random_state=random_state)),
    #         # ('pca', PCA(n_components=0.95, random_state=random_state)),  # optional
    #                 # --- AdaBoost replaces RandomForest here ---
    #         # ('clf', AdaBoostClassifier(
    #         #     estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state),
    #         #     n_estimators=200,
    #         #     learning_rate=0.5,
    #         #     #algorithm='SAMME',
    #         #     random_state=random_state
    #         # )),
    #         ('clf', RandomForestClassifier(random_state=random_state, class_weight='balanced')),
    #     ])
    #     return pipeline

    # # Hyperparameters to search
    # param_grid = {
    #     'smote__sampling_strategy': [1.0],
    #     # 'clf__n_estimators': [100],
    #     # 'clf__learning_rate': [0.2],
    #     # 'clf__estimator__max_depth': [1, 2],   # depth-1 stump is classic; 2 can help if features are noisy
    #     # 'clf__max_depth': [None, 10, 20],
    #     # 'clf__min_samples_split': [2, 5],
    #     # 'clf__min_samples_leaf': [1, 2],
    #     'clf__n_estimators': [20],
    #     'clf__max_features': [None],
    # }

    # # Instantiate runner (data prep + low-level training live here)
    # runner = ExperimentRunner(
    #     experiment_name="rf_grid_search_cv",
    #     verbose=True
    # )

    # # Instantiate manager (or use runner.manager if you wired it in __init__)
    # manager = ExperimentManager(
    #     runner=runner,
    #     experiment_dir=runner.experiment_dir,  # reuse the directory runner created
    #     printer=runner.printer,
    #     verbose=True
    # )

    # # Kick off run (same args as before)
    # manager.run(
    #     validation_strategy=validation_strategy,
    #     pipeline_generator=pipeline_generator,
    #     param_grid=param_grid,
    #     relative_path='../../../data_generation_log/act_data',
    #     file_name='act_data_generated.csv',
    #     dict_filename='act_data_dict_generated.csv',
    #     target_feature='stai_state_score',
    #     responder_criteria='above_median_decrease_in_severe',
    #     group_var_name='Group_tp0',
    #     group_value=[2, 4],
    #     visit_times=['0', '1'],
    #     random_state=random_state,
    #     num_training_repetition=100
    # )



    # J-map
    # AdamBoost
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.kernel_approximation import Nystroem
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import ExtraTreesClassifier
    # XGBoost
    from xgboost import XGBClassifier
    # Extremely Randomized Trees
    from sklearn.ensemble import ExtraTreesClassifier

    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import SGDClassifier
    

    
    # --- experiment setup (Jmap running in severe to moderate baseline state anxiety people) ---
    from .ml_clinical_act_jmap.jmap_act_preprocessWrapper import JmapACTPreprocessor
    from .ml_clinical_act_jmap.pca_with_names import PCAWithNames
    from .ml_clinical_act_jmap.safe_smote import SafeSMOTE
    from .ml_clinical_act_jmap.hetero_selector import DFStandardScaler, WelchTTestSelector, MRMRSelector
    # validation_strategy = "cross_validation_grid_search_opt_in_jmap"
    validation_strategy = "run_internal_cross_validation_experiment_in_severe_state_anxiety_in_jmap"
    random_state = 42
    random.seed(random_state)



    # # Create pipeline factory
    # def pipeline_generator(features, numerical_features, categorical_features, jmap_features, random_state=42, *args, **kwargs):
    #     pipeline = Pipeline([
    #         ("prep", JmapACTPreprocessor(jmap_features=["jmap_tp1"],     # columns containing your volumes
    #                         strategy="flatten",             # "stats", "flatten", or "pca"
    #                         n_components=8,                # used only if strategy="pca"
    #                         keep_channel_axis=True,         # if your data are 4D (X,Y,Z,C)
    #                         random_state=random_state,
    #                         atlas_path="../hammers_atlas/Hammers_mith_atlas_n30r83_SPM5.nii.gz",        # optional for region mapping
    #                         atlas_labels_path="../hammers_atlas/n30r83_id2name_clean.txt",
    #                         scale_volume=True               # <- always StandardScale the features
    #                     )),
            
    #         ('pca', PCAWithNames(n_components=0.95, random_state=random_state)),
    #         # --- mRMR goes here ---
    #         #("scaler_df", DFStandardScaler(with_mean=True, with_std=True)),
    #         ("ttest", WelchTTestSelector(p_thresh=1e-4, min_k_if_empty=2000, cap_after_t=15000)),
    #         ("mrmr",  MRMRSelector(frac_for_topk=0.01, min_topk=10, max_topk=20)),
    #         # replace your LDA stage with:
    #         #("sep", LDA(solver="eigen", shrinkage="auto", n_components=None)),
    #         #("nystroem", Nystroem(kernel="rbf", n_components=20, random_state=42)),
    #         #("fs", SelectKBest(mutual_info_classif, k=10)),   # nonlinear MI-based selection
    #         #("fs", SelectFromModel(ExtraTreesClassifier(n_estimators=100, random_state=random_state), threshold="median")),
    #         # ('pca', PCA(n_components=0.95, random_state=random_state)),  # optional
    #                 # --- AdaBoost replaces RandomForest here ---
    #         # ('clf', AdaBoostClassifier(
    #         #     estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state),
    #         #     n_estimators=200,
    #         #     learning_rate=0.5,
    #         #     #algorithm='SAMME',
    #         #     random_state=random_state
    #         # )),
    #         ("rbf", RBFSampler(gamma=1.0, n_components=10, random_state=random_state)),
    #         ("sgd", SGDClassifier(loss="log_loss", max_iter=1000, random_state=random_state))
            
    #         #('smote', SafeSMOTE(sampling_strategy=1.0, k_neighbors=2, random_state=random_state)),

    #         #('clf', RandomForestClassifier(random_state=random_state, class_weight='balanced')),
    #         # ('clf', XGBClassifier(
    #         # random_state=random_state,
    #         # use_label_encoder=False,
    #         # eval_metric='logloss',
    #         # scale_pos_weight=1  # adjust if your dataset is imbalanced
    #         # ))
            
            
    #         # ('clf', ExtraTreesClassifier(
    #         #     random_state=random_state,
    #         #     n_jobs=-1,              # use all cores
    #         #     bootstrap=False         # ExtraTrees usually best w/o bootstrap
    #         # ))

    #     ])
    #     return pipeline

    # Create pipeline factory
    def pipeline_generator(features, numerical_features, categorical_features, jmap_features, random_state=42, *args, **kwargs):
        pipeline = Pipeline([
            ("prep", JmapACTPreprocessor(jmap_features=["jmap_tp1"],     # columns containing your volumes
                            strategy="flatten",             # "stats", "flatten", or "pca"
                            n_components=8,                # used only if strategy="pca"
                            keep_channel_axis=True,         # if your data are 4D (X,Y,Z,C)
                            random_state=random_state,
                            atlas_path="../hammers_atlas/Hammers_mith_atlas_n30r83_SPM5.nii.gz",        # optional for region mapping
                            atlas_labels_path="../hammers_atlas/n30r83_id2name_clean.txt",
                            scale_volume=True               # <- always StandardScale the features
                        )),
            ('smote', SafeSMOTE(sampling_strategy=1.0, k_neighbors=2, random_state=random_state)),
            ('pca', PCAWithNames(n_components=0.95, random_state=random_state)),
            # --- mRMR goes here ---
            ("ttest", WelchTTestSelector(p_thresh=1e-4, min_k_if_empty=2000, cap_after_t=15000)),
            ("mrmr",  MRMRSelector(frac_for_topk=0.01, min_topk=10, max_topk=20)),
            ("rbf", RBFSampler(gamma=1.0, n_components=10, random_state=random_state)),
            ("sgd", SGDClassifier(loss="log_loss", max_iter=1000, random_state=random_state))
        ])
        return pipeline

    # Hyperparameters to search
    param_grid = {
        # Random Forest
        # 'clf__estimator__max_depth': [1, 2],   # depth-1 stump is classic; 2 can help if features are noisy
        # 'clf__max_depth': [None, 10, 20],
        # 'clf__min_samples_split': [2, 5],
        # 'clf__min_samples_leaf': [1, 2],


        # XGBoost parameters
        'clf__n_estimators': [50],           # overprovision, early_stopping_rounds will cut it down
        'clf__max_depth': [2],             # shallower trees for speed
        'clf__learning_rate': [0.2],     # lower LR works better with early stopping
        'clf__subsample': [0.8],
        'clf__colsample_bytree':[0.8],
        'clf__colsample_bylevel': [0.8],
        'clf__colsample_bynode':[0.8],
        'clf__min_child_weight': [5],    # stronger regularization
        'clf__gamma': [1],                    # pruning strength
        'clf__tree_method': ['hist'],         # fast histogram algorithm (or 'gpu_hist' if GPU)
        'clf__max_bin': [128],                # fewer histogram bins = faster
        'clf__n_jobs': [-1],                  # use all CPU cores  

        # Extremely Randomized Trees parameters
        # 'clf__n_estimators': [20],          # fast & stable; bump to 500 if you can
        # 'clf__max_depth': [None],        # None = fully grown; 10–14 often good
        # 'clf__max_features': ['sqrt'],  # strong stochasticity
        # 'clf__min_samples_split': [2],   # regularization vs speed
        # 'clf__min_samples_leaf': [2],  # regularization vs speed
        # 'clf__criterion': ['gini'],          # keep fast; 'entropy' is slower
        # 'clf__class_weight': ['balanced']  # if classes imbalanced

    }

    # Instantiate runner (data prep + low-level training live here)
    runner = ExperimentRunner(
        experiment_name="rf_grid_search_cv",
        verbose=True
    )

    # Instantiate manager (or use runner.manager if you wired it in __init__)
    manager = ExperimentManager(
        runner=runner,
        experiment_dir=os.path.join(runner.experiment_dir,str(random_state)),  # reuse the directory runner created
        printer=runner.printer,
        verbose=True
    )

    # Kick off run (same args as before)
    manager.run(
        validation_strategy=validation_strategy,
        pipeline_generator=pipeline_generator,
        param_grid=param_grid,
        relative_path='../../../data_generation_log/act_data',
        file_name='act_data_generated.csv',
        dict_filename='act_data_dict_generated.csv',
        target_feature='stai_state_score',
        responder_criteria='above_median_decrease_in_severe',
        group_var_name='Group_tp0',
        group_value=[4],
        visit_times=['0', '1'],
        random_state=random_state,
        num_training_repetition=40
    )
    
    print("Seeds used in the experiments: ", manager.seed_list)