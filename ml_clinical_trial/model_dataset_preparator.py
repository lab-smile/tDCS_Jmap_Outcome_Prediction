from .ml_clinical_act_jmap.ml_clinical_act.ml_clinical.data_spiltter import DataSpiltter

from .ml_clinical_act_jmap.jmap_act_importer_ml import JmapActDataImporterML
from .ml_clinical_act_jmap.jmap_act_feature_type_identifier import JmapActFeatureTypeIdentifier

from .ml_clinical_act_jmap.ml_clinical_act.act_importer_ml import ActDataImporterML
from .ml_clinical_act_jmap.ml_clinical_act.act_data_cleaner import ActDataCleaner
from .ml_clinical_act_jmap.ml_clinical_act.act_feature_selector import ActFeatureSelector
from .ml_clinical_act_jmap.ml_clinical_act.act_feature_type_identifier import ActFeatureTypeIdentifier

from .ml_clinical_act_jmap.ml_clinical_act.ml_clinical.data_labeler import DataLabeler

from .meta_df import get_meta_df
from .data_characterization import characterize_dataset

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

class ModelDatasetPreparator:
    def __init__(self, 
                 relative_path:str, 
                 experiment_dir,
                 file_name = 'act_data_generated.csv',
                 dict_filename='act_data_dict_generated.csv',
                 target_feature = 'stai_state_score',
                 responder_criteria = 'decrease',
                 group_var_name = 'Group_tp0',
                 group_value = [3, 4],
                 visit_times = ['0', '1'],
                 printer = print):

        self.relative_path = relative_path
        self.file_name = file_name
        self.dict_filename = dict_filename

        self.target_feature = target_feature

        self.responder_criteria = responder_criteria
        self.group_var_name = group_var_name
        self.group_value = group_value
        self.experiment_dir = experiment_dir

        self.visit_times = visit_times

        self.printer = printer
     

    def prepare_train_test_data(self,
                                ImporterType,
                                ActDataCleaner,
                                DataLabeler,
                                ActFeatureSelector,
                                ActFeatureTypeIdentifier,
                                new_category_list = ['State and Trait Anxiety Inventory (Adjunct_STAI)'],
                                non_severe_included = True,
                                j_map_included = False):

        
        relative_path = self.relative_path
        experiment_dir = self.experiment_dir
        file_name = self.file_name
        dict_filename = self.dict_filename

        target_feature = self.target_feature

        responder_criteria = self.responder_criteria
        group_var_name = self.group_var_name
        group_value = self.group_value

        visit_times = self.visit_times
        
        printer = self.printer
        
        ### STEP 1 ### 
        # import the ACT data in selected file path
        importer, data, dict_data = ModelDatasetPreparator.import_act(ImporterType, relative_path, file_name, dict_filename)
        print(data.head())
        print(f"Shape of imported data: {data.shape}\n")

        ### STEP 2 ### 
        # sequentially clean the dataset by filtering participants 
        # based on incomplete labels, intervention group, severity, valid jmap (if j_map_included is True), 
        # construct ground-truth responder labels, 
        # and return the cleaned subject IDs with the corresponding responder table.
        responder_df, subject_ids, labeler = ModelDatasetPreparator.clean_data_before_split(importer,
                                                                relative_path,
                                                                target_feature,
                                                                responder_criteria,
                                                                group_var_name,
                                                                group_value,
                                                                ActDataCleaner,
                                                                DataLabeler,
                                                                non_severe_included,
                                                                j_map_included,
                                                                experiment_dir)
        ### STEP 3 ### 
        # splite the data based on cross-site validation manner
        train_df, test_df = ModelDatasetPreparator.data_splitting_cross_site(importer, responder_df, relative_path, experiment_dir, printer)
        ### STEP 4 ###
        # filter features by visit times, drops all-NaN ones, and keeps only those in the given categories.
        features = ModelDatasetPreparator.post_split_feature_selection(
            importer,
            visit_times,
            train_df,
            ActFeatureSelector,
            new_category_list,
            printer
        )
        ### STEP 5 ###
        if j_map_included == True:
            # classifies features (numerical, categorical, JMAP) and builds the cleaned feature matrix X, showing its head and shape
            numerical_features, categorical_features, jmap_features, X = ModelDatasetPreparator.indentify_feature_type_construct_feature_matrix(importer,
                                                                                                                            features,
                                                                                                                            subject_ids,
                                                                                                                            ActFeatureTypeIdentifier,
                                                                                                                            printer,
                                                                                                                            j_map_included)
        if j_map_included == False:
            # classifies features (numerical, categorical) and builds the cleaned feature matrix X, showing its head and shape
            numerical_features, categorical_features, X = ModelDatasetPreparator.indentify_feature_type_construct_feature_matrix(importer,
                                                                                                                features,
                                                                                                                subject_ids,
                                                                                                                ActFeatureTypeIdentifier,
                                                                                                                printer,
                                                                                                                j_map_included)


        ### STEP 6 ###
        # Unified the format of training set and testing set
        X_train, X_test, y_train, y_test = ModelDatasetPreparator.format_dataset(X, train_df, test_df, printer)

        # calculate descriptive statistics for the training set, stratified by responder status
        from .ml_clinical_act_jmap.ml_clinical_act.ml_clinical.spaghetti_plot import stats_aggregate, StatsTimePoint
        tp1_df_full, tp2_df_full, responder_full = labeler.tp1_df, labeler.tp2_df, labeler.responder
        tp1_df, tp2_df, responder = tp1_df_full, tp2_df_full, responder_full
        tp1_df = tp1_df.loc[X_train.index]
        tp2_df = tp2_df.loc[X_train.index]
        responder = responder.loc[X_train.index]
        stats_aggregate(tp1_df, tp2_df, responder, labeler, StatsTimePoint.TRAINING)
        print(f"[ModelDatasetPreparator] stats_aggregate done for TRAINING data set")

        # calculate descriptive statistics for the testing set, stratified by responder status
        from .ml_clinical_act_jmap.ml_clinical_act.ml_clinical.spaghetti_plot import stats_aggregate, StatsTimePoint
        tp1_df, tp2_df, responder = tp1_df_full, tp2_df_full, responder_full
        tp1_df = tp1_df.loc[X_test.index]
        tp2_df = tp2_df.loc[X_test.index]
        responder = responder.loc[X_test.index]
        stats_aggregate(tp1_df, tp2_df, responder, labeler, StatsTimePoint.TESTING)
        print(f"[ModelDatasetPreparator] stats_aggregate done for testing data set")
        

        if j_map_included == False:
            # Characterize the dataset and save the figures and tables
            # Run for both train and test sets
            print("Generating dataset characterization figures and tables...\n")
            metadata_df = get_meta_df()
            characterize_dataset(X_train, metadata_df, prefix="train")
            characterize_dataset(X_test, metadata_df, prefix="test")

        if j_map_included == True:
            return X_train, X_test, y_train, y_test, features, numerical_features, categorical_features, jmap_features
        if j_map_included == False:
            return X_train, X_test, y_train, y_test, features, numerical_features, categorical_features
        
    def prepare_cross_validation_data(self,
                                ImporterType,
                                ActDataCleaner,
                                DataLabeler,
                                ActFeatureSelector,
                                ActFeatureTypeIdentifier,
                                random_state,
                                new_category_list = ['State and Trait Anxiety Inventory (Adjunct_STAI)'],
                                non_severe_included = True,
                                j_map_included = False):

        
        relative_path = self.relative_path
        experiment_dir = self.experiment_dir
        file_name = self.file_name
        dict_filename = self.dict_filename

        target_feature = self.target_feature

        responder_criteria = self.responder_criteria
        group_var_name = self.group_var_name
        group_value = self.group_value

        visit_times = self.visit_times
        
        printer = self.printer
        
        ### STEP 1 ### 
        # import the ACT data in selected file path
        importer, data, dict_data = ModelDatasetPreparator.import_act(ImporterType, relative_path, file_name, dict_filename)
        print(data.head())
        print(f"Shape of imported data: {data.shape}\n")

        ### STEP 2 ### 
        # sequentially clean the dataset by filtering participants 
        # based on incomplete labels, intervention group, severity, valid jmap (if j_map_included is True), 
        # construct ground-truth responder labels, 
        # and return the cleaned subject IDs with the corresponding responder table.
        responder_df, subject_ids, labeler = ModelDatasetPreparator.clean_data_before_split(importer,
                                                                relative_path,
                                                                target_feature,
                                                                responder_criteria,
                                                                group_var_name,
                                                                group_value,
                                                                ActDataCleaner,
                                                                DataLabeler,
                                                                non_severe_included,
                                                                j_map_included,
                                                                experiment_dir)
        ### STEP 3 ### 
        # splite the data in internal cross validation manner
        cv_df = ModelDatasetPreparator.data_splitting_internal_cross_validation(importer, 
                                                                        responder_df, 
                                                                        relative_path, 
                                                                        experiment_dir, 
                                                                        printer,
                                                                        random_state
                                                                        )
        
        
        # ### STEP 4 ###
        # filter features by visit times, drops all-NaN ones, and keeps only those in the given categories.
        features_dfs = ModelDatasetPreparator.get_features_from_cross_validation_df(importer,
                                                                                visit_times,
                                                                                cv_df,
                                                                                ActFeatureSelector,
                                                                                new_category_list,
                                                                                printer)

        # ### STEP 5 ###
        # initialize empty DataFrames
        numerical_features_dfs = pd.DataFrame(columns=["fold", "data"])
        categorical_features_dfs = pd.DataFrame(columns=["fold", "data"])
        jmap_features_dfs = pd.DataFrame(columns=["fold", "data"])
        X_dfs = pd.DataFrame(columns=["fold", "data"])
        print("print(features_df)")
        print(features_dfs)
        all_same = features_dfs["features"].apply(lambda x: x == features_dfs["features"].iloc[0]).all()

        print(f"all_same: {all_same}")
        if all_same == False:
            for _, row in features_dfs.iterrows():
                fold_id = row["fold"]
                features = row["features"]
                if j_map_included:
                    numerical_features, categorical_features, jmap_features, X = (
                        ModelDatasetPreparator.indentify_feature_type_construct_feature_matrix(
                            importer,
                            features,
                            subject_ids,
                            ActFeatureTypeIdentifier,
                            printer,
                            j_map_included
                        )
                    )

                    numerical_features_dfs = pd.concat(
                        [numerical_features_dfs, pd.DataFrame([{"fold": fold_id, "data": numerical_features}])],
                        ignore_index=True
                    )
                    categorical_features_dfs = pd.concat(
                        [categorical_features_dfs, pd.DataFrame([{"fold": fold_id, "data": categorical_features}])],
                        ignore_index=True
                    )
                    jmap_features_dfs = pd.concat(
                        [jmap_features_dfs, pd.DataFrame([{"fold": fold_id, "data": jmap_features}])],
                        ignore_index=True
                    )
                    X_dfs = pd.concat(
                        [X_dfs, pd.DataFrame([{"fold": fold_id, "data": X}])],
                        ignore_index=True
                    )

                else:
                    numerical_features, categorical_features, X = (
                        ModelDatasetPreparator.indentify_feature_type_construct_feature_matrix(
                            importer,
                            features,
                            subject_ids,
                            ActFeatureTypeIdentifier,
                            printer,
                            j_map_included
                        )
                    )

                    numerical_features_dfs = pd.concat(
                        [numerical_features_dfs, pd.DataFrame([{"fold": fold_id, "data": numerical_features}])],
                        ignore_index=True
                    )
                    categorical_features_dfs = pd.concat(
                        [categorical_features_dfs, pd.DataFrame([{"fold": fold_id, "data": categorical_features}])],
                        ignore_index=True
                    )
                    X_dfs = pd.concat(
                        [X_dfs, pd.DataFrame([{"fold": fold_id, "data": X}])],
                        ignore_index=True
                    )
        else:
            from copy import deepcopy
            num_records, cat_records, jmap_records, X_records = [], [], [], []
            # Compute ONCE using the first row, then replicate for all folds
            first = features_dfs.iloc[0]
            first_fold = first["fold"]
            features = first["features"]

            if j_map_included:
                numerical_features, categorical_features, jmap_features, X = (
                    ModelDatasetPreparator.indentify_feature_type_construct_feature_matrix(
                        importer,
                        features,
                        subject_ids,
                        ActFeatureTypeIdentifier,
                        printer,
                        j_map_included,
                    )
                )
            else:
                numerical_features, categorical_features, X = (
                    ModelDatasetPreparator.indentify_feature_type_construct_feature_matrix(
                        importer,
                        features,
                        subject_ids,
                        ActFeatureTypeIdentifier,
                        printer,
                        j_map_included,
                    )
                )
                jmap_features = None  # not used

            # Duplicate results for each fold id.
            # deepcopy so later per-fold mutations won't alias the same object.
            fold_ids = features_dfs["fold"].tolist()
            for fid in fold_ids:
                num_records.append({"fold": fid, "data": deepcopy(numerical_features)})
                cat_records.append({"fold": fid, "data": deepcopy(categorical_features)})
                if j_map_included:
                    jmap_records.append({"fold": fid, "data": deepcopy(jmap_features)})
                X_records.append({"fold": fid, "data": deepcopy(X)})
            # Build DataFrames once
            numerical_features_dfs = pd.DataFrame(num_records, columns=["fold", "data"])
            categorical_features_dfs = pd.DataFrame(cat_records, columns=["fold", "data"])
            if j_map_included:
                jmap_features_dfs = pd.DataFrame(jmap_records, columns=["fold", "data"])
            else:
                jmap_features_dfs = pd.DataFrame(columns=["fold", "data"])
            X_dfs = pd.DataFrame(X_records, columns=["fold", "data"])


        # ### STEP 6 ###
        # Unified the format of training set and testing set
        new_rows = []
        # iterate only through the 'train' rows to pair with corresponding 'test' rows
        for i, train_row in cv_df.query("split == 'train'").reset_index(drop=True).iterrows():
            fold_id = i + 1
            train_df = train_row["fold"]  # in your old cv_df, 'fold' stores the actual DataFrame
            # find the matching test row (same index position)
            test_row = cv_df.query("split == 'test'").reset_index(drop=True).iloc[i]
            test_df = test_row["fold"]
            new_rows.append({
                "fold": fold_id,
                "train_df": train_df,
                "test_df": test_df
            })
        cv_df = pd.DataFrame(new_rows)
        X_train_dfs = pd.DataFrame(columns = ["fold", "data"])
        X_test_dfs = pd.DataFrame(columns = ["fold", "data"])
        y_train_dfs = pd.DataFrame(columns = ["fold", "data"])
        y_test_dfs = pd.DataFrame(columns = ["fold", "data"])
        for (_, cv_df_row), (_, X_df_row) in zip(cv_df.iterrows(),X_dfs.iterrows()):
            if cv_df_row["fold"] != X_df_row["fold"]:
                raise ValueError("Fold IDs do not match between cv_df and X_dfs.")
            fold_id = cv_df_row["fold"]
            
            X_train, X_test, y_train, y_test = ModelDatasetPreparator.format_dataset(X, train_df, test_df, printer)
            X_train_dfs = pd.concat(
                [X_train_dfs, pd.DataFrame([{"fold": fold_id, "data": X_train}])],
                ignore_index = True
            )
            X_test_dfs = pd.concat(
                [X_test_dfs, pd.DataFrame([{"fold": fold_id, "data": X_test}])],
                ignore_index = True
            )
            y_train_dfs = pd.concat(
                [y_train_dfs, pd.DataFrame([{"fold": fold_id, "data": y_train}])],
                ignore_index = True
            )
            y_test_dfs = pd.concat(
                [y_test_dfs, pd.DataFrame([{"fold": fold_id, "data": y_test}])],
                ignore_index = True
            )

        # # calculate descriptive statistics for the training set, stratified by responder status
        # from .ml_clinical_act_jmap.ml_clinical_act.ml_clinical.spaghetti_plot import stats_aggregate, StatsTimePoint
        # tp1_df_full, tp2_df_full, responder_full = labeler.tp1_df, labeler.tp2_df, labeler.responder
        # tp1_df, tp2_df, responder = tp1_df_full, tp2_df_full, responder_full
        # tp1_df = tp1_df.loc[X_train.index]
        # tp2_df = tp2_df.loc[X_train.index]
        # responder = responder.loc[X_train.index]
        # stats_aggregate(tp1_df, tp2_df, responder, labeler, StatsTimePoint.TRAINING)
        # print(f"[ModelDatasetPreparator] stats_aggregate done for TRAINING data set")

        # # calculate descriptive statistics for the testing set, stratified by responder status
        # from .ml_clinical_act_jmap.ml_clinical_act.ml_clinical.spaghetti_plot import stats_aggregate, StatsTimePoint
        # tp1_df, tp2_df, responder = tp1_df_full, tp2_df_full, responder_full
        # tp1_df = tp1_df.loc[X_test.index]
        # tp2_df = tp2_df.loc[X_test.index]
        # responder = responder.loc[X_test.index]
        # stats_aggregate(tp1_df, tp2_df, responder, labeler, StatsTimePoint.TESTING)
        # print(f"[ModelDatasetPreparator] stats_aggregate done for testing data set")
        

        # if j_map_included == False:
        #     # Characterize the dataset and save the figures and tables
        #     # Run for both train and test sets
        #     print("Generating dataset characterization figures and tables...\n")
        #     metadata_df = get_meta_df()
        #     characterize_dataset(X_train, metadata_df, prefix="train")
        #     characterize_dataset(X_test, metadata_df, prefix="test")

        if j_map_included == True:
            return X_train_dfs, X_test_dfs, y_train_dfs, y_test_dfs, features_dfs, numerical_features_dfs, categorical_features_dfs, jmap_features_dfs
        if j_map_included == False:
            return X_train_dfs, X_test_dfs, y_train_dfs, y_test_dfs, features_dfs, numerical_features_dfs, categorical_features_dfs

    def prepare_train_test_data_in_severe_state_anxiety(self,
                                                        ImporterType = ActDataImporterML,
                                                        ActDataCleaner = ActDataCleaner,
                                                        DataLabeler = DataLabeler,
                                                        ActFeatureSelector = ActFeatureSelector,
                                                        ActFeatureTypeIdentifier = JmapActFeatureTypeIdentifier):
        new_category_list = ['State and Trait Anxiety Inventory (Adjunct_STAI)',
        'Beck Depression Inventory-II (Adjunct_BDI)',
        'demographics']
        non_severe_included = False
        j_map_included = False

        return self.prepare_train_test_data(ImporterType,
                                            ActDataCleaner,
                                            DataLabeler,
                                            ActFeatureSelector,
                                            ActFeatureTypeIdentifier,
                                            new_category_list=new_category_list,
                                            non_severe_included=non_severe_included,
                                            j_map_included=j_map_included)
    
    def prepare_train_test_data_in_severe_state_anxiety_in_jmap(self,
                                                                ImporterType = JmapActDataImporterML,
                                                                ActDataCleaner = ActDataCleaner,
                                                                DataLabeler = DataLabeler,
                                                                ActFeatureSelector = ActFeatureSelector,
                                                                ActFeatureTypeIdentifier = JmapActFeatureTypeIdentifier):
        new_category_list = ['jmap']
        non_severe_included = False
        j_map_included = True

        return self.prepare_train_test_data(ImporterType,
                                    ActDataCleaner,
                                    DataLabeler,
                                    ActFeatureSelector,
                                    ActFeatureTypeIdentifier,
                                    new_category_list=new_category_list,
                                    non_severe_included=non_severe_included,
                                    j_map_included = j_map_included)
    
    def prepare_cross_validation_data_in_severe_state_anxiety_in_jmap(self, random_state,
                                                                ImporterType = JmapActDataImporterML,
                                                                ActDataCleaner = ActDataCleaner,
                                                                DataLabeler = DataLabeler,
                                                                ActFeatureSelector = ActFeatureSelector,
                                                                ActFeatureTypeIdentifier = JmapActFeatureTypeIdentifier):
        new_category_list = ['jmap']
        non_severe_included = False
        j_map_included = True
        return self.prepare_cross_validation_data(ImporterType,
                                            ActDataCleaner,
                                            DataLabeler,
                                            ActFeatureSelector,
                                            ActFeatureTypeIdentifier,
                                            random_state,
                                            new_category_list=new_category_list,
                                            non_severe_included=non_severe_included,
                                            j_map_included=j_map_included)
    
    #############################
    # helper
    #############################
    @staticmethod
    def get_features_from_cross_validation_df(importer,
                                            visit_times,
                                            cv_df,
                                            ActFeatureSelector,
                                            new_category_list,
                                            printer):
        records = []
        def iter_train_dfs(cv_df):
            for i, row in cv_df.query("split == 'train'").reset_index(drop=True).iterrows():
                yield i + 1, row["fold"]     # (fold_id, train_df)
        for fold_id, train_df in iter_train_dfs(cv_df):
            # filter features by visit times, drops all-NaN ones, and keeps only those in the given categories.
            feature = ModelDatasetPreparator.post_split_feature_selection(
                importer,
                visit_times,
                train_df,
                ActFeatureSelector,
                new_category_list,
                printer
            )
            records.append({
                "fold": fold_id,
                "features": feature
            })
        # Convert the list of dicts into a DataFrame
        features_df = pd.DataFrame(records)
        return features_df



    @staticmethod
    def import_act( ImporterType, relative_path, file_name, dict_filename):
        # import the ACT data in selected file path
        # If relative_path is a tuple, convert it to a proper path
        if isinstance(relative_path, tuple):
            relative_path = Path(*relative_path)
        else:
            relative_path = Path(relative_path)
        importer = ImporterType(relative_path)
        data = importer.load_data( filename = file_name)
        dict_data = importer.load_feature_dictionary(dict_filename = dict_filename)
        return importer, data, dict_data
    
    @staticmethod
    def data_splitting_cross_site( importer, responder_df, relative_path, experiment_dir, printer):
        # splite the data based on cross-site validation manner
        splitter = DataSpiltter(importer, responder_df, relative_path, experiment_dir)
        train_df, test_df = splitter.split_data_by_site()
        splitter.visualize_distribution(train_df, test_df)
        printer(f"head of training set:\n{train_df.to_string()}\n")
        printer(f"shape of training set: {train_df.shape}\n")
        printer(f"head of testprinter=printing set:\n{test_df.to_string()}\n")
        printer(f"shape of testing set: {test_df.shape}\n")

        return train_df, test_df
    
    @staticmethod
    def data_splitting_internal_cross_validation( importer, responder_df, relative_path, experiment_dir, printer, random_state):
        # splite the data based on cross-site validation manner
        #splitter = DataSpiltter(importer, responder_df, relative_path, experiment_dir)
        #train_df, test_df = splitter.split_data_by_site()
        # splitter.visualize_distribution(train_df, test_df)
        # printer(f"head of training set:\n{train_df.head()}\n")
        # printer(f"shape of training set: {train_df.shape}\n")
        # printer(f"head of testprinter=printing set:\n{test_df.head()}\n")
        # printer(f"shape of testing set: {test_df.shape}\n")

        # 5-fold stratified CV (reproducible)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        fold_summaries = []
        fold_paths = []
        for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(responder_df.index.tolist(), responder_df["responder"]), start=1
        ):
            train_df = responder_df.iloc[train_idx]
            test_df = responder_df.iloc[test_idx]
            # Summaries
            tr_pos = int(train_df["responder"].sum())
            tr_neg = int((train_df["responder"]==0).sum())
            te_pos = int(test_df["responder"].sum())
            te_neg = int((test_df["responder"]==0).sum())

            fold_summaries.append({
                "fold_idx": fold_idx,
                "fold": train_df,
                "split": "train",
                "n": len(train_df),
                "pos": tr_pos,
                "neg": tr_neg,
                "pos_ratio": round(tr_pos/len(train_df), 3)
            })
            fold_summaries.append({
                "fold_idx": fold_idx,
                "fold": train_df,
                "split": "test",
                "n": len(test_df),
                "pos": te_pos,
                "neg": te_neg,
                "pos_ratio": round(te_pos/len(test_df), 3)
            })
        # Show the per-fold class balance
        cv_df = pd.DataFrame(fold_summaries)
        return cv_df
    
    @staticmethod
    def format_dataset( X, train_df, test_df, printer):
        # Unified the format of training set and testing set
        # Ensure subject_id is the index (participant ids)
        X = X.copy()
        train_df = train_df.copy()
        test_df = test_df.copy()
        # Align indices (participant ids) and extract data
        X_train = X.loc[train_df.index]
        y_train = train_df['responder']
        X_test = X.loc[test_df.index]
        y_test = test_df['responder']
        printer(f"Shape of training set: {X_train.shape}, Labels: {y_train.shape}\n")
        printer(f"Shape of testing set: {X_test.shape}, Labels: {y_test.shape}\n")
        
        return X_train, X_test, y_train, y_test
    


    
    @staticmethod
    def clean_data_before_split(importer,
                                relative_path,
                                target_feature,
                                responder_criteria,
                                group_var_name,
                                group_value,
                                ActDataCleaner,
                                DataLabeler,
                                non_severe_included,
                                j_map_included,
                                experiment_dir):
        
        # sequentially clean the dataset by filtering participants 
        # (incomplete labels, intervention group, severity, valid jmap), 
        # construct ground-truth responder labels, 
        # and return the cleaned subject IDs with the corresponding responder table.

        # define the target feature used to construct ground truth labels
        target_feature_tp1 = f"{target_feature}_tp1"
        target_feature_tp2 = f"{target_feature}_tp2"
        
        cleaner = ActDataCleaner(importer, relative_path)



        # clean the data set by removing participants without complete target feature record #
        # (no valid ground truth label is constructed for participants without complete target feature record)
        subject_ids = cleaner.get_cleaned_subject_ids_by_selected_labeler([target_feature_tp1, target_feature_tp2], 
                                                                        responder_criteria)
        # subject 116036 should be excluded due to BDI outside acceptable range
        # https://pmc.ncbi.nlm.nih.gov/articles/PMC11110843/
        # Individuals who responded ‘yes’ to ever having anxiety 
        # or depression could be included in the trial 
        # as long as they did not describe currently 
        # experiencing significant symptoms negatively interfering 
        # with daily function upon further query by study staff.
        # Furthermore, participants were excluded if they had 
        # a total Beck Depression Inventory – Second Edition (BDI-II) score ≥ 20 
        # (indicative of at least moderate depression symptoms) and 
        # were encouraged to seek further evaluation for depression.
        # The ACT trial primarily targeted healthy, cognitively intact older adults from the community, 
        # many of whom may experience subclinical symptoms of depression or anxiety.
        #subject_ids = cleaner.clean_data_by_ids(subject_ids, [116036])

        print(f"[ModelDatasetPreparator] Number of subjects after cleaning by available target feature ({target_feature_tp1}, {target_feature_tp2}): {len(subject_ids)}\n")
        print(f"[ModelDatasetPreparator] ID list of subjects after cleaning by available target feature:")
        print(subject_ids)
        
        
        # ground truth label construction, 
        # construct a table called responder_df with ground truth labels for each participant
        # participant id is the index of the table called responder_df with ground truth labels
        labeler = DataLabeler(importer, subject_ids, relative_path, experiment_dir = experiment_dir)
        responder_df_origin = labeler.label_data(label_columns=[target_feature_tp1, target_feature_tp2],
                                            responder_criteria=responder_criteria)
        responder_df = responder_df_origin.dropna(subset=['responder'])

        print(f"[ModelDatasetPreparator] Number of subjects after constructing responder labels: {responder_df.shape[0]}\n")
        print(f"[ModelDatasetPreparator] ID list of subjects after constructing responder labels:")
        print(responder_df.index.tolist())

        # clean the data set by removing participants who were not in selected intervention group
        # Category of Intervention Groups:
        # 1 = Education Control Training + Sham tDCS; 
        # 2 = Education Control Training + tDCS; 
        # 3 = Cognitive Training + Sham tDCS; 
        # 4 = Cognitive Training + tDCS
        subject_ids = cleaner.clean_data_by_site_and_group(subject_ids, group_var_name, group_value)

        print(f"[ModelDatasetPreparator] Number of subjects after cleaning by intervention group ({group_var_name} in {group_value}): {len(subject_ids)}\n")
        print(f"[ModelDatasetPreparator] ID list of subjects after cleaning by intervention group:")
        print(subject_ids)

        if non_severe_included == False:
            # clean the data set by selecting the severe state anxiety people
            # only including the participant who has severe state anxiety at baseline time point right before the intervention
            # when a participant has state anxiety score in STAI greater than or equal to 39, the participant has severe state anxiety
            subject_ids = cleaner.clean_data_by_feature_values(
                subject_ids, 
                'stai_state_score_tp1', 
                39, 
                'greater than or equal to')
            print(f"[ModelDatasetPreparator] Number of subjects after cleaning by severe state anxiety (stai_state_score_tp1 >= 39): {len(subject_ids)}\n")
            print(f"[ModelDatasetPreparator] ID list of subjects after cleaning by severe state anxiety:")
            print(subject_ids)
            if len(subject_ids) == 0:
                raise ValueError("No subjects left after cleaning by severe state anxiety criteira. Please check the data.")
            
        if j_map_included == True:
            # clean the data set by selecting the participant with valid registrated jmap
            subject_ids = cleaner.clean_data_by_feature_values(
                subject_ids, 
                'jmap_tp1', 
                np.nan, 
                'valid')
            if len(subject_ids) == 0:
                raise ValueError("No subjects left after cleaning by jmap availability. Please check the data.")
            print(f"[ModelDatasetPreparator] Number of subjects after cleaning by jmap availability (valid jmap_tp1): {len(subject_ids)}\n")
            print(f"[ModelDatasetPreparator] ID list of subjects after cleaning by jmap availability:")
            print(subject_ids)


        # calculate descriptive statistics for the target feature at each time point, stratified by responder status
        # and inferential statistics to compare the target feature between responders and non-responders at each time point
        from .ml_clinical_act_jmap.ml_clinical_act.ml_clinical.spaghetti_plot import stats_aggregate, StatsTimePoint
        tp1_df, tp2_df, responder = labeler.tp1_df, labeler.tp2_df, labeler.responder
        tp1_df = tp1_df.loc[subject_ids]
        tp2_df = tp2_df.loc[subject_ids]
        responder = responder.loc[subject_ids]
        stats_aggregate(tp1_df, tp2_df, responder, labeler, StatsTimePoint.POST)
        print(f"[ModelDatasetPreparator] stats_aggregate done after the data cleaning")
            
          
        # renew the respodner_df to only include participants in cleaned dataset
        responder_df = responder_df.loc[subject_ids]
        
        return responder_df, subject_ids, labeler
    
    @staticmethod
    def post_split_feature_selection(importer, 
                                     visit_times,
                                     train_df, 
                                     FeatureSelector,
                                     new_categories,
                                     printer):
        # filter features by visit times, drops all-NaN ones, and keeps only those in the given categories.

        # Feature Selection
        # Select features corresponding to screening and baseline visit and of type 'num'
        feature_selector = FeatureSelector(importer)
        printer(f"Number of features before selection: {len(importer.get_feature_names())}\n")
        features = feature_selector.select_features(visit_times = visit_times)
        printer(f"Number of features after selection by visits: {len(features)}\n")
        
        # remove the feature with all nan values in the training set
        subject_id_in_training_set = train_df.index.tolist()
        features = feature_selector.rm_nan_features(subject_id_in_training_set, features)
        printer(f"Number of features after removing NaN features: {len(features)}\n")
        
        # remove features with is not in selected feature categories
        features = feature_selector.select_feature_by_new_category_from_features(features, 
                                                                                 new_categories = new_categories)
        printer(f"Number of selected features by new_categories: {len(features)}\n")
        return features
    
    @staticmethod
    def indentify_feature_type_construct_feature_matrix(importer,
                                                        features,
                                                        subject_ids,
                                                        FeatureTypeIdentifier,
                                                        printer,
                                                        include_jmap):
        # classifies features (numerical, categorical, JMAP) and builds the cleaned feature matrix X, showing its head and shape

        # indentify the type of the feature categories
        feature_type_identifier = FeatureTypeIdentifier( importer)
        if isinstance(feature_type_identifier, JmapActFeatureTypeIdentifier):
            print("It's a JmapActFeatureTypeIdentifier")
            numerical_features, categorical_features, jmap_features = feature_type_identifier.get_feature_type_lists(features)
        elif isinstance(feature_type_identifier, ActFeatureTypeIdentifier):
            print("It's an ActFeatureTypeIdentifier")
            numerical_features, categorical_features = feature_type_identifier.get_feature_type_lists(features)

        printer(f"Numerical features: {len(numerical_features)}\n")
        printer(f"list of numerical features: {numerical_features}\n")
        printer(f"Categorical features: {len(categorical_features)}\n")
        printer(f"list of categorical features: {categorical_features}\n")
        printer(f'Number of rows {len(subject_ids)} and columns {len(features)} in the feature matrix X\n')

        # construct the cleaned data set based on selected participant id and selected features
        X = importer.get_filtered_data(subject_ids, features)
        printer("Head of feature matrix X:\n")
        printer(f"{X.head()}\n")
        printer("\n")
        printer("Shape of feature matrix X:")
        printer(f"{X.shape}\n")
        printer("\n")

        if include_jmap == False:
            return numerical_features, categorical_features, X
        else:
            return numerical_features, categorical_features, jmap_features, X

if __name__ == "__main__":
    relative_path='../../../data_generation_log/act_data',
    file_name='act_data_generated.csv',
    dict_filename='act_data_dict_generated.csv',
    target_feature='stai_state_score',
    responder_criteria='above_median_decrease_in_severe',
    group_var_name='Group_tp0',
    group_value=[4],
    visit_times=['0', '1'],

    preparator = ModelDatasetPreparator(
        relative_path = relative_path,
        experiment_dir = './experiments',
        file_name = file_name,
        dict_filename = dict_filename,
        target_feature = target_feature,
        responder_criteria = responder_criteria,
        group_var_name = group_var_name,
        group_value = group_value,
        visit_times = visit_times,
        printer = print
    )
    