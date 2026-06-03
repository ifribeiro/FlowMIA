"""
FlowMIA: A framework for privacy evaluation of synthetic tabular data.

Supports Membership Inference Attacks (MIA) via:
    - FlowMIA-GAN: discriminator-based MIA using a trained GAN
    - DOMIAS: density-ratio MIA using normalizing flows (BNAF)
    - DCR: distance-to-closest-record MIA using nearest neighbors
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.privacy.flowmia_gan import FlowMIA_GAN
from src.privacy.domias_bnaf import domias_bnaf
from src.privacy.dcr_mia import dcr_mia

from src.utility.utility import Utility
from src.fidelity.fidelity import fidelity_compute, plotFidelity


class FlowMIA:
    """
    Orchestrates privacy evaluation of a synthetic dataset against real training data.

    Preprocessing is handled once here and shared across all attack methods.
    Numerical columns are scaled with RobustScaler, categorical columns are
    one-hot encoded, and IP address columns are normalized to [0, 1] by
    dividing by 2**32 - 1.

    Args:
        config (dict): Configuration dictionary with the following keys:

            - member_path (str): Path to the training (member) CSV.
            - non_member_path (str): Path to the held-out (non-member) CSV.
            - synth_path (str): Path to the synthetic data CSV.
            - test_path (str): Path to the utility test CSV.
            - save_path (str): Directory where outputs and checkpoints are saved.
            - categorical_cols (list[str]): Names of categorical feature columns.
            - numerical_cols (list[str]): Names of numerical feature columns.
            - ip_cols (list[str]): Names of IP address columns.
            - label_col (str): Name of the label/target column.
            - use_wgan (bool): Whether to use WGAN-GP instead of standard GAN.
            - batch_size (int): Batch size for GAN training.
            - num_epochs (int): Number of GAN training epochs.
            - fcheckpoint (int): Checkpoint save frequency (in epochs).
    """

    def __init__(self, config: dict):
        self.config = config


        os.makedirs(config["save_path"], exist_ok=True)
        self.save_path = config["save_path"]

        self.categorical_cols = config["categorical_cols"]
        self.numerical_cols = config["numerical_cols"]
        self.ip_cols = config["ip_cols"]
        self.label_col = config["label_col"]
        self.use_wgan = config["use_wgan"]

        features = [self.ip_cols, self.numerical_cols, self.categorical_cols,]        
        if self.label_col is None:
            features.append([])
        else:
            features.append([self.label_col])
        self.all_features = []
        for f in features:
            self.all_features+=f

        self.member = pd.read_csv(config["member_path"])[self.all_features]
        self.non_member = pd.read_csv(config["non_member_path"])[self.all_features]
        self.synth = pd.read_csv(config["synth_path"])[self.all_features]
        self.util_test = pd.read_csv(config["test_path"])[self.all_features]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        # --------------- preprocessor ---------------
        # Fits on synthetic data so the attack sees the same feature space
        # as the generator. IP columns are handled separately (see _transform).
        all_cat = self.categorical_cols 
        all_categories = []
        for col in all_cat:
            # pega categorias do REAL (não do synth)
            cats = pd.concat([self.member[col], self.non_member[col]]).unique()
            all_categories.append(sorted(cats))

        # 2. Construir os transformers dinamicamente
        transformers = []
        if self.numerical_cols:
            transformers.append(("num", StandardScaler(),  self.numerical_cols))
        if all_cat:
            # Só adicionamos o 'cat' se houver colunas categóricas
            transformers.append(("cat", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                categories=all_categories
            ), all_cat))

        

        self.preprocessor = ColumnTransformer(transformers=transformers)
        fit_data = pd.concat([self.synth, self.member], axis=0)
        self.preprocessor.fit(fit_data)

        # Pre-transform every split once so downstream methods receive arrays
        self.X_member = self._transform(self.member)
        self.X_non_member = self._transform(self.non_member)
        self.X_synth = self._transform(self.synth)
        print(self.X_member.shape, self.X_non_member.shape, self.X_synth.shape)
        # --------------- GAN ---------------
        self.flowmia_gan = FlowMIA_GAN(
            X_member=self.X_member,
            X_non_member=self.X_non_member,
            X_synth=self.X_synth,
            batch_size=config["batch_size"],
            use_wgan=self.use_wgan,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted preprocessor and append normalized IP columns.

        Args:
            df: Raw input dataframe.

        Returns:
            2-D float32 array ready to feed into attack models.
        """
        X = self.preprocessor.transform(df).astype(np.float32)
        if self.ip_cols:
            ips = (df[self.ip_cols].values / (2**32 - 1)).astype(np.float32)
            X = np.hstack([X, ips])
        return X

    # ------------------------------------------------------------------
    # Attack methods
    # ------------------------------------------------------------------

    def flowmiagan(self, pre_trained_model: str = None, plot: bool = False, test_size=1000) -> dict:
        """
        Run the FlowMIA-GAN membership inference attack.

        Trains (or loads) a GAN on the synthetic data, then uses the
        discriminator's confidence scores to separate members from non-members.

        Args:
            pre_trained_model: Path to a saved checkpoint. When provided,
                training is skipped and the checkpoint is loaded directly.
            plot: If True, saves score distribution, ROC/PR, and confusion
                matrix plots to ``save_path/plots/``.

        Returns:
            Dictionary of MIA metrics including AUC, accuracy, precision,
            recall, F1, score statistics, and Wasserstein distances.
        """
        print(f"Starting FlowMIA-GAN privacy evaluation {pre_trained_model}...")

        if pre_trained_model:
            self.flowmia_gan.load_model(pre_trained_model)
        else:
            self.flowmia_gan.fit(
                epochs=self.config["num_epochs"],
                fcheckpoint=self.config["fcheckpoint"],
                save_path=self.save_path,
            )

        self.mia_results = self.flowmia_gan.membership_inference(test_size=test_size)
        print(f"FlowMIA-GAN results: AUC={self.mia_results['auc']:.4f}")

        if plot:
            colors = {
                "members": "#e74c3c",
                "non_members": "#3498db",
                "synthetic": "#2ecc71",
                "random": "#95a5a6",
            }
            self.flowmia_gan.plot_all(
                results=self.mia_results,
                colors=colors,
                save_path=self.save_path,
            )

        return self.mia_results

    def domias(self, test_size: int = 1000, epochs: int = 10, save_path: str = "", load: bool = False) -> tuple[np.ndarray, float]:
        """
        Run the DOMIAS membership inference attack.

        Estimates the log-density of the synthetic distribution at each
        query point using a Block Neural Autoregressive Flow (BNAF). A
        higher density score implies the sample is more likely a member.

        Args:
            test_size: Number of members and non-members to sample for
                evaluation (balanced split).

        Returns:
            Tuple of (scores, auc) where ``scores`` is a 1-D array of
            log-density values and ``auc`` is the ROC-AUC of the attack.
        """
        rng = np.random.default_rng(42)
        idx_m = rng.choice(len(self.X_member), size=test_size, replace=False)
        idx_nm = rng.choice(len(self.X_non_member), size=test_size, replace=False)

        scores, auc = domias_bnaf(
            train_members=self.X_member[idx_m],
            non_members=self.X_non_member[idx_nm],
            synthetic_data=self.X_synth,
            device=self.device,
            save_path=save_path,
            epochs=epochs,
            load=load
        )
        print(f"DOMIAS results: AUC={auc:.4f}")
        return scores, auc

    def compute_dcr(self, test_size: int = 1000) -> tuple[np.ndarray, float]:
        """
        Run the DCR (Distance to Closest Record) membership inference attack.

        Assigns each query point a score equal to the negative distance to
        its nearest neighbor in the synthetic dataset. Members are expected
        to have smaller distances (higher scores) than non-members.

        Args:
            test_size: Number of members and non-members to sample for
                evaluation (balanced split).

        Returns:
            Tuple of (scores, auc) where ``scores`` is a 1-D array of
            negative-distance values and ``auc`` is the ROC-AUC of the attack.
        """
        rng = np.random.default_rng(42)
        idx_m = rng.choice(len(self.X_member), size=test_size, replace=False)
        idx_nm = rng.choice(len(self.X_non_member), size=test_size, replace=False)

        scores, auc = dcr_mia(
            train_members=self.X_member[idx_m],
            non_members=self.X_non_member[idx_nm],
            synthetic_data=self.X_synth,
        )
        print(f"DCR results: AUC={auc:.4f}")
        return scores, auc

    def evaluate_privacy(
        self,
        plot: bool = False,
        run_domias: bool = False,
        run_dcr: bool = False,
        test_size: int = 1000,
    ) -> dict:
        """
        Run all enabled privacy attacks and collect results.

        Args:
            plot: Forward to :meth:`flowmiagan` to save diagnostic plots.
            run_domias: Whether to include the DOMIAS attack.
            run_dcr: Whether to include the DCR attack.
            test_size: Evaluation sample size for DOMIAS and DCR.

        Returns:
            Dictionary with keys ``"flowmiagan"``, ``"domias"``, and ``"dcr"``.
            Values are ``None`` when the corresponding attack was not run.
        """
        results = {
            "flowmiagan": self.flowmiagan(plot=plot),
            "domias": None,
            "dcr": None,
        }

        if run_domias:
            scores, auc = self.domias(test_size=test_size)
            results["domias"] = {"scores": scores, "auc": auc}

        if run_dcr:
            scores, auc = self.compute_dcr(test_size=test_size)
            results["dcr"] = {"scores": scores, "auc": auc}

        return results
    
    # ------------------------------------------------------------------
    # Fidelity
    # ------------------------------------------------------------------
 
    def evaluate_fidelity(self, plot: bool = False) -> dict:
        """
        Measure per-feature distributional similarity between real and synthetic data.
 
        Computes KL divergence, Jensen-Shannon divergence, and Wasserstein distance
        for every column. Categorical columns use probability mass functions over
        observed categories; numerical columns use histogram-based estimates.
 
        Args:
            plot: If True, saves a grouped bar chart (log scale) of all three
                divergences per feature to ``save_path/plots/fidelity.pdf``.
 
        Returns:
            Dictionary mapping each column name to a (1, 3) numpy array
            containing ``[kl, js, wasserstein]`` divergence values.
        """
        self.fidelity_results = fidelity_compute(
            self.member,
            self.synth,
            categorical=self.categorical_cols + [self.label_col],
        )
 
        if plot:
            plotFidelity(self.fidelity_results, self.save_path)
 
        return self.fidelity_results
 
    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
 
    def evaluate_utility(self, classifiers: list, plot: bool = False) -> dict:
        """
        Benchmark synthetic data utility via RTR and TSTR protocols.
 
        For each classifier in ``classifiers``, two pipelines are evaluated:
 
        - **RTR** (Real-Train / Real-Test): trained on real member data, tested
          on the held-out test set. Serves as an upper-bound reference.
        - **TSTR** (Synthetic-Train / Real-Test): trained on synthetic data,
          tested on the same held-out test set. A TSTR score close to RTR
          indicates high utility.
 
        Each pipeline includes its own RobustScaler + OneHotEncoder fit, so
        no pre-transformed arrays are needed here.
 
        Args:
            classifiers: List of scikit-learn compatible classifier instances.
            plot: If True, saves a 2×2 grid of metric bar charts to
                ``save_path/plots/utility.pdf``.
 
        Returns:
            Nested dictionary of the form
            ``{classifier_name: {"RTR": metrics_dict, "TSTR": metrics_dict}}``
            where each ``metrics_dict`` contains Accuracy, Precision, Recall,
            and F1-Score.
        """
        utility = Utility(
            classifiers,
            self.member,
            self.util_test,
            self.synth,
            self.categorical_cols,
            self.numerical_cols,
            self.ip_cols,
            self.label_col,
        )
        utility_dict = utility.evaluate()
 
        if plot:
            utility.plot_utility(utility_dict, self.save_path)
 
        return utility_dict
 
    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------
 
    def evaluate_all(
        self,
        classifiers: list,
        plot: bool = False,
        run_domias: bool = False,
        run_dcr: bool = False,
        test_size: int = 1000,
    ) -> dict:
        """
        Run the complete evaluation pipeline: privacy, fidelity, and utility.
 
        This is a convenience method that calls :meth:`evaluate_privacy`,
        :meth:`evaluate_fidelity`, and :meth:`evaluate_utility` in sequence
        and bundles their outputs into a single dictionary.
 
        Args:
            classifiers: List of scikit-learn classifiers for utility evaluation.
            plot: If True, all diagnostic plots are saved under ``save_path/plots/``.
            run_domias: Whether to include the DOMIAS attack in privacy evaluation.
            run_dcr: Whether to include the DCR attack in privacy evaluation.
            test_size: Evaluation sample size for DOMIAS and DCR.
 
        Returns:
            Dictionary with keys ``"privacy"``, ``"fidelity"``, and ``"utility"``,
            each containing the results of the corresponding evaluation.
        """
        return {
            "privacy": self.evaluate_privacy(
                plot=plot,
                run_domias=run_domias,
                run_dcr=run_dcr,
                test_size=test_size,
            ),
            "fidelity": self.evaluate_fidelity(plot=plot),
            "utility": self.evaluate_utility(classifiers, plot=plot),
        }