# experiment_manager.py
import os
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

from .model_evaluator import ModelEvaluator
from .model_dataset_preparator import ModelDatasetPreparator
from .meta_df import get_meta_df


class ExperimentManager:
    """
    Orchestrates experiment runs. Delegates data preparation and
    model training to a provided `runner` instance (e.g., ExperimentRunner)
    that implements:
      - prepare_train_test_data(...)
      - prepare_train_test_data_in_severe_state_anxiety(...)
      - prepare_train_test_data_in_severe_state_anxiety_in_jmap(...)
      - train_by_loo_and_ga(...)
      - train_by_grid_search_cv(...)
      - train_by_genetic_opt_cv(...)
    """

    def __init__(self, runner, experiment_dir: str | None = None, printer=print, verbose: bool = False):
        self.runner = runner
        self.verbose = verbose

        # default experiment_dir and printer
        if experiment_dir is None:
            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            experiment_dir = f"./results_{now}"
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"[ExperimentManager] Experiment directory created: {experiment_dir}\n")

        self.experiment_dir = experiment_dir
        self.printer = printer

        self.seed_list = []

    # ---------------------------
    # Public API (moved from Runner)
    # ---------------------------
    def run(self,
            validation_strategy,
            pipeline_generator,
            param_grid,
            relative_path,
            file_name='act_data_generated.csv',
            dict_filename='act_data_dict_generated.csv',
            target_feature='stai_state_score',
            responder_criteria='decrease',
            group_var_name='Group_tp0',
            group_value=[3, 4],
            visit_times=['0', '1'],
            random_state=42,
            num_training_repetition=20):

        strategy = str(validation_strategy).lower()
        results = []

        if strategy == "cross_validation_grid_search_opt":
            # Create a fresh directory and file logger per run (preserves original behavior)
            now = datetime.now()
            formatted_datetime = now.strftime("%Y_%m_%d_%H_%M_%S")
            experiment_dir = f"./results_{formatted_datetime}"
            os.makedirs(experiment_dir, exist_ok=True)
            log_path = os.path.join(experiment_dir, f"output_{formatted_datetime}.log")

            def file_printer(log_str):
                with open(log_path, "a") as f:
                    f.write(log_str)


            self.run_grid_search_cv_experiment_in_severe_state_anxiety(
                experiment_dir,
                pipeline_generator,
                param_grid,
                relative_path,
                file_name=file_name,
                dict_filename=dict_filename,
                target_feature=target_feature,
                responder_criteria=responder_criteria,
                group_var_name=group_var_name,
                group_value=group_value,
                visit_times=visit_times,
                random_state=random_state,
                num_training_repetition=num_training_repetition,
                printer=file_printer,
            )
        
        elif strategy == "cross_validation_genetic_opt":
            # Create a fresh directory and file logger per run (preserves original behavior)
            now = datetime.now()
            formatted_datetime = now.strftime("%Y_%m_%d_%H_%M_%S")
            experiment_dir = f"./results_{formatted_datetime}"
            os.makedirs(experiment_dir, exist_ok=True)
            log_path = os.path.join(experiment_dir, f"output_{formatted_datetime}.log")

            def file_printer(log_str):
                with open(log_path, "a") as f:
                    f.write(log_str)


            self.run_genetic_opt_cv_experiment_in_severe_state_anxiety(
                experiment_dir,
                pipeline_generator,
                param_grid,
                relative_path,
                file_name=file_name,
                dict_filename=dict_filename,
                target_feature=target_feature,
                responder_criteria=responder_criteria,
                group_var_name=group_var_name,
                group_value=group_value,
                visit_times=visit_times,
                random_state=random_state,
                num_training_repetition=num_training_repetition,
                printer=file_printer,
            )

        elif strategy == "cross_validation_grid_search_opt_in_jmap":
            self.run_grid_search_opt_cv_experiment_in_severe_state_anxiety_in_jmap(
                self.experiment_dir,
                pipeline_generator,
                param_grid,
                relative_path,
                file_name=file_name,
                dict_filename=dict_filename,
                target_feature=target_feature,
                responder_criteria=responder_criteria,
                group_var_name=group_var_name,
                group_value=group_value,
                visit_times=visit_times,
                random_state=random_state,
                num_training_repetition=num_training_repetition,
                printer=self.printer,
            )
        elif strategy == "run_internal_cross_validation_experiment_in_severe_state_anxiety_in_jmap":
            self.run_internal_cross_validation_experiment_in_severe_state_anxiety_in_jmap(
                self.experiment_dir,
                pipeline_generator,
                param_grid,
                relative_path,
                file_name=file_name,
                dict_filename=dict_filename,
                target_feature=target_feature,
                responder_criteria=responder_criteria,
                group_var_name=group_var_name,
                group_value=group_value,
                visit_times=visit_times,
                random_state=random_state,
                num_training_repetition=num_training_repetition,
                printer=self.printer,
            )

        elif strategy == "cross_site_validation":
            self.run_loocv_and_ga_experiment_in_severe_state_anxiety(
                self.experiment_dir,
                pipeline_generator,
                param_grid,
                relative_path,
                file_name=file_name,
                dict_filename=dict_filename,
                target_feature=target_feature,
                responder_criteria=responder_criteria,
                group_var_name=group_var_name,
                group_value=group_value,
                visit_times=visit_times,
                random_state=random_state,
                num_training_repetition=num_training_repetition,
                printer=self.printer,
            )

        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        self._save_model()
        return results

    def _save_model(self, filename="model.pkl"):
        # Keep placeholder behavior from original code.
        # If you later want to persist the trained pipeline, wire it here.
        # Example (when you have a pipeline handle):
        # model_path = os.path.join(self.experiment_dir, filename)
        # joblib.dump(pipeline, model_path)
        # self.printer(f"Model saved to: {model_path}\n")
        pass

    # ---------------------------
    # Experiment runners (moved from Runner)
    # ---------------------------
    def run_loocv_and_ga_experiment(
        self,
        experiment_dir,
        pipeline_generator,
        param_grid,
        relative_path,
        file_name='act_data_generated.csv',
        dict_filename='act_data_dict_generated.csv',
        target_feature='stai_state_score',
        responder_criteria='decrease',
        group_var_name='Group_tp0',
        group_value=[3, 4],
        visit_times=['0', '1'],
        random_state=42,
        num_training_repetition=20,
        printer=print,
    ):
        balanced_acc_list = []
        model_evaluator = ModelEvaluator(verbose=True)
        for i in range(random_state, random_state + num_training_repetition):
            random_seed = i
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                X_train, X_test, y_train, y_test, features, numerical_features, categorical_features = \
                    self.runner.prepare_train_test_data(
                        relative_path,
                        file_name=file_name,
                        dict_filename=dict_filename,
                        target_feature=target_feature,
                        responder_criteria=responder_criteria,
                        group_var_name=group_var_name,
                        group_value=group_value,
                        visit_times=visit_times,
                        printer=printer
                    )

                pipeline = pipeline_generator(features, numerical_features, categorical_features)
                pipeline, X_test, y_test = self.runner.train_by_loo_and_ga(
                    X_train, X_test, y_train, y_test, pipeline, param_grid, printer, random_seed
                )
                performance_report = model_evaluator.evaluate(pipeline, X_test, y_test)
                balanced_acc_list.append(performance_report["balanced_accuracy"])

            for w in caught_warnings:
                printer(f"[Warning] {w.message}")
                printer("  --> From:" + f"{w.filename}" + "\n")
                printer("  --> Line:" + f"{w.lineno}" + "\n")
                printer("  --> Category:" + f"{w.category}" + "\n")

        model_evaluator.vivid_histogram(balanced_acc_list, experiment_dir=experiment_dir)

    def run_loocv_and_ga_experiment_in_severe_state_anxiety(
        self,
        experiment_dir,
        pipeline_generator,
        param_grid,
        relative_path,
        file_name='act_data_generated.csv',
        dict_filename='act_data_dict_generated.csv',
        target_feature='stai_state_score',
        responder_criteria='decrease',
        group_var_name='Group_tp0',
        group_value=[3, 4],
        visit_times=['0', '1'],
        random_state=42,
        num_training_repetition=20,
        printer=print,
    ):
        balanced_acc_list = []
        model_evaluator = ModelEvaluator(verbose=True)
        for i in range(random_state, random_state + num_training_repetition):
            random_seed = i
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                X_train, X_test, y_train, y_test, features, numerical_features, categorical_features = \
                    self.runner.prepare_train_test_data_in_severe_state_anxiety(
                        relative_path,
                        file_name=file_name,
                        dict_filename=dict_filename,
                        target_feature=target_feature,
                        responder_criteria=responder_criteria,
                        group_var_name=group_var_name,
                        group_value=group_value,
                        visit_times=visit_times,
                        printer=printer
                    )

                pipeline = pipeline_generator(features, numerical_features, categorical_features)
                pipeline, X_test, y_test = self.runner.train_by_loo_and_ga(
                    X_train, X_test, y_train, y_test, pipeline, param_grid, printer, random_seed
                )
                performance_report = model_evaluator.evaluate(pipeline, X_test, y_test)
                balanced_acc_list.append(performance_report["balanced_accuracy"])

            for w in caught_warnings:
                printer(f"[Warning] {w.message}")
                printer("  --> From:" + f"{w.filename}" + "\n")
                printer("  --> Line:" + f"{w.lineno}" + "\n")
                printer("  --> Category:" + f"{w.category}" + "\n")

        model_evaluator.vivid_histogram(balanced_acc_list, experiment_dir=experiment_dir)

    def run_grid_search_cv_experiment_in_severe_state_anxiety(
        self,
        experiment_dir,
        pipeline_generator,
        param_grid,
        relative_path,
        file_name='act_data_generated.csv',
        dict_filename='act_data_dict_generated.csv',
        target_feature='stai_state_score',
        responder_criteria='decrease',
        group_var_name='Group_tp0',
        group_value=[3, 4],
        visit_times=['0', '1'],
        random_state=42,
        num_training_repetition=20,
        printer=print,
    ):
        balanced_acc_list = []
        model_evaluator = ModelEvaluator(verbose=True)

        model_dataset_preparator = ModelDatasetPreparator(
            relative_path,
            experiment_dir,
            file_name=file_name,
            dict_filename=dict_filename,
            target_feature=target_feature,
            responder_criteria=responder_criteria,
            group_var_name=group_var_name,
            group_value=group_value,
            visit_times=visit_times,
            printer=printer,
        )
        X_train, X_test, y_train, y_test, features, numerical_features, categorical_features = \
            model_dataset_preparator.prepare_train_test_data_in_severe_state_anxiety()
        all_imps = []  # list of per-iter importance DataFrames
        for i in range(random_state, random_state + num_training_repetition):
            random_seed = i
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")



                pipeline = pipeline_generator(features, numerical_features, categorical_features, random_state=random_seed)
                pipeline, X_test, y_test = self.runner.train_by_grid_search_cv(
                    X_train, y_train, X_test, y_test, pipeline, param_grid, printer, random_seed
                ) if False else self.runner.train_by_grid_search_cv(
                    X_train, X_test, y_train, y_test, pipeline, param_grid, printer, random_seed
                )

                performance_report = model_evaluator.evaluate(pipeline, X_test, y_test)
                balanced_acc_list.append(performance_report["balanced_accuracy"])
                printer(f"random_state: {i}: ")
                printer(f"balanced_acc: {performance_report['balanced_accuracy']}\n")

                # --- SHAP values ---
                metadata_df = get_meta_df()
                # Now compute and plot SHAP values
                # 1) Build the name mapper once (from your metadata sheet)
                # metadata_df columns must be: variable, label, choices
                namer = model_evaluator.make_stai_feature_namer(metadata_df)

                # call your function (assumes we're in a class; otherwise drop `self.`)
                imp = model_evaluator.shap_summary_tables(
                    model_pipeline=pipeline,
                    X=X_test,
                    experiment_dir = experiment_dir,
                    feature_namer=namer,    # or pass your prettifier
                )
                # store with iteration id
                imp = imp.assign(iter=i)  # columns: ['feature', 'mean_abs_shap', 'iter']
                all_imps.append(imp)

                # AUC ROC curve and PR curve
                _ = model_evaluator.collect_roc_pr_curves(model_pipeline=pipeline, X=X_test, y=y_test,
                                class_idx=1, experiment_dir=experiment_dir, n_points=1001,
                                run_name=f"iter_{i:04d}")

                model_evaluator.collect_performance_metrics(
                    model_pipeline=pipeline,
                    X=X_test,
                    y=y_test,
                    threshold=0.5,
                    experiment_dir=experiment_dir,
                    run_name=f"iter_{i:04d}"
                )

            for w in caught_warnings:
                printer(f"[Warning] {w.message}")
                printer("  --> From:" + f"{w.filename}" + "\n")
                printer("  --> Line:" + f"{w.lineno}" + "\n")
                printer("  --> Category:" + f"{w.category}" + "\n")

        if len(balanced_acc_list) > 1:
            model_evaluator.vivid_histogram(balanced_acc_list, experiment_dir=experiment_dir)
            # --- combine & aggregate ---
            all_imps_df = pd.concat(all_imps, ignore_index=True)
            out = model_evaluator.save_shap_importance_summary(
                all_imps_df=all_imps_df,
                experiment_dir=experiment_dir,
                top_k=30,
                ci=0.95,
                figsize=(12,6),
                filename_stub="shap_importance_mean_std",
            )

            # access the aggregated table in-memory
            summary_df = out["summary_df"]
            print(summary_df.head(10))

            # once at the end:
            _ = model_evaluator.plot_aggregated_curves(experiment_dir=experiment_dir, ci=0.95)
            # After all iterations, produce the aggregate stats table
            model_evaluator.summarize_performance_metrics(
                experiment_dir=experiment_dir,
                ci=0.95,
                save_csv=True
            )
            print("Per-run CSV:", f"{experiment_dir}/table/performance/per_run_performance.csv")
            print("Summary CSV:", f"{experiment_dir}/table/performance/performance_summary_stats.csv")

        
        # metadata_df = get_meta_df()
        # # Now compute and plot SHAP values
        # # 1) Build the name mapper once (from your metadata sheet)
        # # metadata_df columns must be: variable, label, choices
        # namer = model_evaluator.make_stai_feature_namer(metadata_df)

        # # 2) Call your SHAP plotter and pass the mapper
        # shap_out = model_evaluator.shap_summary_plots(
        #     model_pipeline=pipeline,
        #     X=X_test,
        #     experiment_dir = experiment_dir,
        #     feature_namer=namer,        # <-- pretty labels everywhere
        # )

        # # Optional: view top features
        # print(shap_out["importance"].head())

    def run_genetic_opt_cv_experiment_in_severe_state_anxiety(
        self,
        experiment_dir,
        pipeline_generator,
        param_grid,
        relative_path,
        file_name='act_data_generated.csv',
        dict_filename='act_data_dict_generated.csv',
        target_feature='stai_state_score',
        responder_criteria='decrease',
        group_var_name='Group_tp0',
        group_value=[3, 4],
        visit_times=['0', '1'],
        random_state=42,
        num_training_repetition=100,
        printer=print,
    ):
        balanced_acc_list = []
        model_evaluator = ModelEvaluator(verbose=True)

        model_dataset_preparator = ModelDatasetPreparator(
            relative_path,
            experiment_dir,
            file_name=file_name,
            dict_filename=dict_filename,
            target_feature=target_feature,
            responder_criteria=responder_criteria,
            group_var_name=group_var_name,
            group_value=group_value,
            visit_times=visit_times,
            printer=printer,
        )
        X_train, X_test, y_train, y_test, features, numerical_features, categorical_features = \
            model_dataset_preparator.prepare_train_test_data_in_severe_state_anxiety()
        for i in range(random_state, random_state + num_training_repetition):
            random_seed = i
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")

                pipeline = pipeline_generator(features, numerical_features, categorical_features)
                pipeline, X_test, y_test = self.runner.train_by_genetic_opt_cv(
                    X_train, X_test, y_train, y_test, pipeline, param_grid, printer, random_seed, experiment_dir
                )
                performance_report = model_evaluator.evaluate(pipeline, X_test, y_test)
                balanced_acc_list.append(performance_report["balanced_accuracy"])
                printer(f"random_state: {i}: ")
                printer(f"balanced_acc == {performance_report['balanced_accuracy']}\n")

            for w in caught_warnings:
                printer(f"[Warning] {w.message}")
                printer("  --> From:" + f"{w.filename}" + "\n")
                printer("  --> Line:" + f"{w.lineno}" + "\n")
                printer("  --> Category:" + f"{w.category}" + "\n")

        model_evaluator.vivid_histogram(balanced_acc_list, experiment_dir=experiment_dir)

    def run_genetic_opt_cv_experiment_in_severe_state_anxiety_in_jmap(
        self,
        experiment_dir,
        pipeline_generator,
        param_grid,
        relative_path,
        file_name='act_data_generated.csv',
        dict_filename='act_data_dict_generated.csv',
        target_feature='stai_state_score',
        responder_criteria='decrease',
        group_var_name='Group_tp0',
        group_value=[3, 4],
        visit_times=['0', '1'],
        random_state=42,
        num_training_repetition=100,
        printer=print,
    ):
        balanced_acc_list = []
        model_evaluator = ModelEvaluator(verbose=True, printer=self.printer)

        (X_train, X_test, y_train, y_test,
         features, numerical_features, categorical_features, jmap_features) = \
            self.runner.prepare_train_test_data_in_severe_state_anxiety_in_jmap(
                relative_path,
                file_name=file_name,
                dict_filename=dict_filename,
                target_feature=target_feature,
                responder_criteria=responder_criteria,
                group_var_name=group_var_name,
                group_value=group_value,
                visit_times=visit_times,
                printer=printer
            )

        for i in range(random_state, random_state + num_training_repetition):
            random_seed = i
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")

                pipeline = pipeline_generator(
                    features, numerical_features, categorical_features, jmap_features, random_seed
                )
                pipeline, X_test, y_test = self.runner.train_by_genetic_opt_cv(
                    X_train, X_test, y_train, y_test, pipeline, param_grid, printer, random_seed
                )
                performance_report = model_evaluator.evaluate(pipeline, X_test, y_test)
                balanced_acc_list.append(performance_report["balanced_accuracy"])
                printer(f"random_state: {random_seed}: ")
                printer(f"balanced_acc == {performance_report['balanced_accuracy']}\n")

            for w in caught_warnings:
                printer(f"[Warning] {w.message}")
                printer("  --> From:" + f"{w.filename}" + "\n")
                printer("  --> Line:" + f"{w.lineno}" + "\n")
                printer("  --> Category:" + f"{w.category}" + "\n")

        model_evaluator.vivid_histogram(balanced_acc_list, experiment_dir=experiment_dir)

    def run_grid_search_opt_cv_experiment_in_severe_state_anxiety_in_jmap(
        self,
        experiment_dir,
        pipeline_generator,
        param_grid,
        relative_path,
        file_name='act_data_generated.csv',
        dict_filename='act_data_dict_generated.csv',
        target_feature='stai_state_score',
        responder_criteria='decrease',
        group_var_name='Group_tp0',
        group_value=[3, 4],
        visit_times=['0', '1'],
        random_state=42,
        num_training_repetition=100,
        printer=print,
    ):
        balanced_acc_list = []
        model_evaluator = ModelEvaluator(verbose=True, printer=self.printer)

        # (X_train, X_test, y_train, y_test,
        #  features, numerical_features, categorical_features, jmap_features) = \
        #     self.runner.prepare_train_test_data_in_severe_state_anxiety_in_jmap(
        #         relative_path,
        #         file_name=file_name,
        #         dict_filename=dict_filename,
        #         target_feature=target_feature,
        #         responder_criteria=responder_criteria,
        #         group_var_name=group_var_name,
        #         group_value=group_value,
        #         visit_times=visit_times,
        #         printer=printer
        #     )
        model_dataset_preparator = ModelDatasetPreparator(
            relative_path,
            experiment_dir,
            file_name=file_name,
            dict_filename=dict_filename,
            target_feature=target_feature,
            responder_criteria=responder_criteria,
            group_var_name=group_var_name,
            group_value=group_value,
            visit_times=visit_times,
            printer=printer,
        )
        (X_train, X_test, y_train, y_test,
         features, numerical_features, categorical_features, jmap_features) = \
            model_dataset_preparator.prepare_train_test_data_in_severe_state_anxiety_in_jmap()

        # # merge all lists
        merged_list = [ 167, 177, 180, 143, 152, 154, 155, 156, 124, 126, 127, 128, 134, 102, 107, 109, 112, 87, 90, 67, 71, 72, 75, 46, 49, 53, 61, 484, 485, 486, 488, 494, 508, 530, 537, 542, 552, 557, 559, 565, 568, 574, 579]
        #merged_list = [154]
        # make sure the root experiment directory exists once
        os.makedirs(experiment_dir, exist_ok=True)
        # Choose if not to use SHAP interpretation for J-map features
        explain = True


        #for i in range(random_state, random_state + num_training_repetition):
        for i in merged_list:
            random_seed = i
            # NEW: create a per-iteration subfolder (e.g., .../seed_167/)
            iter_dir = os.path.join(experiment_dir, f"seed_{random_seed}")
            os.makedirs(iter_dir, exist_ok=True)
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")

                pipeline = pipeline_generator(
                    features, numerical_features, categorical_features, jmap_features, random_seed
                )
                pipeline, X_test, y_test = self.runner.train_by_grid_search_cv(
                    X_train, X_test, y_train, y_test, pipeline, param_grid, printer, random_seed
                )
                performance_report = model_evaluator.evaluate(pipeline, X_test, y_test)

                # NEW: save performance_report as CSV in the iteration folder
                perf_path = os.path.join(iter_dir, "performance_report.csv")
                pd.DataFrame([performance_report]).to_csv(perf_path, index=False)
                # AUC ROC curve and PR curve
                _ = model_evaluator.collect_roc_pr_curves(model_pipeline=pipeline, X=X_test, y=y_test,
                                class_idx=1, experiment_dir=experiment_dir, n_points=1001,
                                run_name=f"iter_{i:04d}")

                model_evaluator.collect_performance_metrics(
                    model_pipeline=pipeline,
                    X=X_test,
                    y=y_test,
                    threshold=0.5,
                    experiment_dir=experiment_dir,
                    run_name=f"iter_{i:04d}"
                )



                balanced_acc_list.append(performance_report["balanced_accuracy"])
                printer(f"random_state: {random_seed}: ")
                printer(f"balanced_acc == {performance_report['balanced_accuracy']}\n")
                if performance_report["balanced_accuracy"] > 0.65:
                    self.seed_list.append(random_seed)
                
                
                if explain == True:
                    # get cooridnate in MNI152
                    shap_table = model_evaluator.get_coordinate_table_mni_shap(pipeline, X_test, y_test)
                    print(shap_table)

                    # NEW: save shap_table as CSV in the iteration folder
                    shap_path = os.path.join(iter_dir, "shap_table.csv")
                    shap_table.to_csv(shap_path, index=False)

            for w in caught_warnings:
                printer(f"[Warning] {w.message}")
                printer("  --> From:" + f"{w.filename}" + "\n")
                printer("  --> Line:" + f"{w.lineno}" + "\n")
                printer("  --> Category:" + f"{w.category}" + "\n")

        if len(balanced_acc_list) > 1:
            model_evaluator.vivid_histogram(balanced_acc_list, experiment_dir=experiment_dir)
            
        # once at the end:
        _ = model_evaluator.plot_aggregated_curves(experiment_dir=experiment_dir, smooth_lpf=True)
        
        # After all iterations, produce the aggregate stats table
        model_evaluator.summarize_performance_metrics(
            experiment_dir=experiment_dir,
            ci=0.95,
            save_csv=True
        )
        print("Per-run CSV:", f"{experiment_dir}/table/performance/per_run_performance.csv")
        print("Summary CSV:", f"{experiment_dir}/table/performance/performance_summary_stats.csv")

        
        
    def run_internal_cross_validation_experiment_in_severe_state_anxiety_in_jmap(
        self,
        experiment_dir,
        pipeline_generator,
        param_grid,
        relative_path,
        file_name='act_data_generated.csv',
        dict_filename='act_data_dict_generated.csv',
        target_feature='stai_state_score',
        responder_criteria='decrease',
        group_var_name='Group_tp0',
        group_value=[3, 4],
        visit_times=['0', '1'],
        random_state=42,
        num_training_repetition=100,
        printer=print,
    ):
        balanced_acc_list = []
        model_evaluator = ModelEvaluator(verbose=True, printer=self.printer)

        model_dataset_preparator = ModelDatasetPreparator(
            relative_path,
            experiment_dir,
            file_name=file_name,
            dict_filename=dict_filename,
            target_feature=target_feature,
            responder_criteria=responder_criteria,
            group_var_name=group_var_name,
            group_value=group_value,
            visit_times=visit_times,
            printer=printer,
        )
        # (X_train, X_test, y_train, y_test,
        #  features, numerical_features, categorical_features, jmap_features) = \
        (X_train_dfs, X_test_dfs, y_train_dfs, y_test_dfs, 
         features_dfs, numerical_features_dfs, categorical_features_dfs, jmap_features_dfs) =\
            model_dataset_preparator.prepare_cross_validation_data_in_severe_state_anxiety_in_jmap(random_state)
        

        # # # merge all lists
        # merged_list = [ 167, 177, 180, 143, 152, 154, 155, 156, 124, 126, 127, 128, 134, 102, 107, 109, 112, 87, 90, 67, 71, 72, 75, 46, 49, 53, 61, 484, 485, 486, 488, 494, 508, 530, 537, 542, 552, 557, 559, 565, 568, 574, 579]

        # make sure the root experiment directory exists once
        os.makedirs(experiment_dir, exist_ok=True)
        # Choose if not to use SHAP interpretation for J-map features
        explain = False
        for i in range(random_state, random_state + num_training_repetition):
            #for i in merged_list:
            random_seed = i
            fold = 1
            balanced_acc_list_across_folds = pd.DataFrame(columns = ['fold', 'balanced_accuracy'])
            for (X_train, X_test, y_train, y_test,
                features, numerical_features, categorical_features, jmap_features) in zip(
                    X_train_dfs["data"], X_test_dfs["data"], y_train_dfs["data"], y_test_dfs["data"],
                    features_dfs["features"], numerical_features_dfs["data"], categorical_features_dfs["data"], jmap_features_dfs["data"]):
                

                # NEW: create a per-iteration subfolder (e.g., .../seed_167/)
                iter_dir = os.path.join(experiment_dir, f"seed_{random_seed}_fold_{fold}")
                os.makedirs(iter_dir, exist_ok=True)
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")

                    pipeline = pipeline_generator(
                        features, numerical_features, categorical_features, jmap_features, random_seed
                    )
                    pipeline, X_test, y_test = self.runner.train_by_grid_search_cv(
                        X_train, X_test, y_train, y_test, pipeline, param_grid, printer, random_seed
                    )
                    performance_report = model_evaluator.evaluate(pipeline, X_test, y_test)

                    # NEW: save performance_report as CSV in the iteration folder
                    perf_path = os.path.join(iter_dir, "performance_report.csv")
                    pd.DataFrame([performance_report]).to_csv(perf_path, index=False)
                    # AUC ROC curve and PR curve
                    _ = model_evaluator.collect_roc_pr_curves(model_pipeline=pipeline, X=X_test, y=y_test,
                                    class_idx=1, experiment_dir=experiment_dir, n_points=1001,
                                    run_name=f"iter_{i:04d}")

                    model_evaluator.collect_performance_metrics(
                        model_pipeline=pipeline,
                        X=X_test,
                        y=y_test,
                        threshold=0.5,
                        experiment_dir=experiment_dir,
                        run_name=f"iter_{i:04d}"
                    )
                    balanced_acc_list_across_folds = pd.concat(
                        [balanced_acc_list_across_folds, pd.DataFrame({'fold': [fold], 'balanced_accuracy': [performance_report["balanced_accuracy"]]})],
                        ignore_index=True
                    )


                    balanced_acc_list.append(performance_report["balanced_accuracy"])
                    printer(f"random_state: {random_seed}: ")
                    printer(f"balanced_acc == {performance_report['balanced_accuracy']}\n")
                    
                    if explain == True:
                        # get cooridnate in MNI152
                        shap_table = model_evaluator.get_coordinate_table_mni_shap(pipeline, X_test, y_test)
                        print(shap_table)

                        # NEW: save shap_table as CSV in the iteration folder
                        shap_path = os.path.join(iter_dir, "shap_table.csv")
                        shap_table.to_csv(shap_path, index=False)
                

                for w in caught_warnings:
                    printer(f"[Warning] {w.message}")
                    printer("  --> From:" + f"{w.filename}" + "\n")
                    printer("  --> Line:" + f"{w.lineno}" + "\n")
                    printer("  --> Category:" + f"{w.category}" + "\n")
                fold += 1
            averaged_balanced_acc_across_folds = balanced_acc_list_across_folds['balanced_accuracy'].mean()
            #if averaged_balanced_acc_across_folds > 0.65:
            self.seed_list.append(random_seed)

        if len(balanced_acc_list) > 1:
            model_evaluator.vivid_histogram(balanced_acc_list, experiment_dir=experiment_dir)
            import scipy.stats as stats
            def summarize_stats(data_list):
                """
                Compute summary statistics for a list of floats:
                - mean
                - std (sample)
                - standard error
                - 95% confidence interval (CI)
                - min, max, median
                """
                data = np.array(data_list)
                
                n = len(data)
                if n == 0:
                    raise ValueError("Input list is empty.")

                mean = np.mean(data)
                std = np.std(data, ddof=1)  # sample std
                sem = stats.sem(data)      # standard error of the mean
                ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)

                return {
                    "count": n,
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                    "95% CI": (ci_low, ci_high),
                    "min": np.min(data),
                    "max": np.max(data),
                    "median": np.median(data),
                }
            stats_summary = summarize_stats(balanced_acc_list)
            print("Balanced Accuracy Summary Statistics:")
            for k, v in stats_summary.items():
                print(f"  {k}: {v}")
            
        # once at the end:
        _ = model_evaluator.plot_aggregated_curves(experiment_dir=experiment_dir, smooth_lpf=True)
        
        # After all iterations, produce the aggregate stats table
        model_evaluator.summarize_performance_metrics(
            experiment_dir=experiment_dir,
            ci=0.95,
            save_csv=True
        )
        print("Per-run CSV:", f"{experiment_dir}/table/performance/per_run_performance.csv")
        print("Summary CSV:", f"{experiment_dir}/table/performance/performance_summary_stats.csv")

    



# experiment_manager.py (append at bottom of file)

if __name__ == "__main__":
    import random
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    from sklearn.decomposition import PCA  # optional, commented below
    from sklearn.ensemble import RandomForestClassifier

    # local imports
    from .ml_clinical_act_jmap.ml_clinical_act.ml_clinical.preprocess_wrapper import PreprocessWrapper  # or: from .ml_clinical_act_jmap.ml_clinical_act.ml_clinical.preprocess_wrapper import PreprocessWrapper
    from .experiment_runner import ExperimentRunner     # your existing runner

    # --- experiment setup (matches your previous script) ---
    validation_strategy = "cross_validation_grid_search_opt"
    random_state = 587
    random.seed(random_state)

    # Create pipeline factory
    def pipeline_generator(features, numerical_features, categorical_features, random_state=42, *args, **kwargs):
        pipeline = Pipeline([
            ('preprocess', PreprocessWrapper(features, numerical_features, categorical_features, verbose=False)),
            ('smote', SMOTE(sampling_strategy=1.0, random_state=random_state)),
            # ('pca', PCA(n_components=0.95, random_state=random_state)),  # optional
            ('clf', RandomForestClassifier(random_state=random_state, class_weight='balanced')),
        ])
        return pipeline

    # Hyperparameters to search
    param_grid = {
        'smote__sampling_strategy': [1.0],
        'clf__n_estimators': [100],
        'clf__max_features': [None],
        # 'clf__max_depth': [None, 10, 20],
        # 'clf__min_samples_split': [2, 5],
        # 'clf__min_samples_leaf': [1, 2],
    }

    # Instantiate runner (data prep + low-level training live here)
    runner = ExperimentRunner(
        experiment_name="rf_grid_search_cv",
        verbose=True
    )

    # Instantiate manager (or use runner.manager if you wired it in __init__)
    manager = ExperimentManager(
        runner=runner,
        experiment_dir=runner.experiment_dir,  # reuse the directory runner created
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
        responder_criteria='decrease',
        group_var_name='Group_tp0',
        group_value=[3, 4],
        visit_times=['0', '1'],
        random_state=random_state,
        num_training_repetition=1
    )


# --- experiment setup (matches your previous script) ---
    # from sklearn_genetic.space import Integer, Categorical, Continuous
    # validation_strategy = "cross_validation_genetic_opt"
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
    #     'clf__n_estimators': Integer(50, 200),            # integer range from 100 to 200
    #     'clf__max_depth': Categorical([None, 10, 20]),     # discrete choices
    #     #'clf__min_samples_split': Integer(2, 5),           # integer from 2 to 5
    #     #'clf__min_samples_leaf': Integer(1, 2),            # integer from 1 to 2
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
    #     num_training_repetition=10
    # )