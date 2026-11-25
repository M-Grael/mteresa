from pathlib import Path
import pickle
import warnings
import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# Default priors. Each entry is (mu, sd) for Normal priors and a
# single sd for the HalfNormal priors of the sigma_* parameters.
# Sampling is done in the time unit of the data (e.g., seconds):
#   L_group : logit scale (q = 1 / (1 + exp(L)))
#   w_group : log scale, rate in 1 / (time unit of the data)
#   b_group : log scale, rate in 1 / (time unit of the data)
# The priors below assume IRTs in seconds.
# ---------------------------------------------------------------
DEFAULT_PRIORS = {
    'L_group': (1.0, 1.5),
    'w_group': (0.5, 1.0),
    'b_group': (-3.0, 1.5),
    'sigma_L': 1.0,
    'sigma_w': 1.0,
    'sigma_b': 1.0,
}


# ---------------------------------------------------------------
# likelihood functions (order of parameters: L, w, b)
# L is the logit of the "within" probability: q = 1 / (1 + exp(L))
# ---------------------------------------------------------------
def random_irt(L, w, b, rng=None, size=None):
    """Yield IRTs (Y - d scale) of two exponentials."""
    if rng is None:
        rng = np.random.default_rng()
    L, w, b = np.broadcast_arrays(L, w, b)
    shape = size if size is not None else L.shape
    q = np.exp(-np.logaddexp(0.0, L))  # numerically stable 1 / (1 + exp(L))
    is_between = rng.binomial(1, q, size=shape).astype(bool)
    between_draw = rng.exponential(1.0 / b, size=shape)  # numpy uses scale = 1/rate
    within_draw = rng.exponential(1.0 / w, size=shape)
    return np.where(is_between, between_draw, within_draw)


def biexp_logp(value, L, w, b):
    """Log-density of the mixture: q*Exp(b) + (1-q)*Exp(w), with q = sigmoid(-L)."""
    import pytensor.tensor as pt

    # log(q) = log_sigmoid(-L) = -softplus(L); log(1-q) = log_sigmoid(L) = -softplus(-L)
    log_between = -pt.softplus(L) + pt.log(b) - b * value
    log_within = -pt.softplus(-L) + pt.log(w) - w * value
    return pt.logaddexp(log_between, log_within)


def _require(df, columns, why):
    """Fails early, with a clear error message."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns ({why}): {missing}. "
            f"Available columns: {list(df.columns)}"
        )


class BiExponentialRefractoryModel:
    """
    Models the inter-response times (IRTs) as a mixture of two
    exponentials. Performs a hierarchical Bayesian analysis at
    the group and subject levels. The class receives a dataframe
    of operant events and can return both descriptive and
    inferential outputs, such as log-survivor plots (for model
    checking) and summaries of the posterior distributions.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of events. Each row corresponds to an event,
        such as 'left lever on', 'dipper on' or 'light off'. The
        dataframe must contain a column with a unique ID for each
        session, a column with timestamps of each event, and a
        column with the group of each event. Alternatively, ``df``
        can already contain the IRTs (see ``irt_column``).

    timestamp_column : str, default='timestamp'
        The column of the dataframe that contains the timestamp
        of each event. The default value, ``timestamp``, follows the
        heuristics of the mteresa package (see ``Notes`` section).
        If the dataframe already contains a column with IRTs,
        ``timestamp_column`` can be set to `None`. In that case,
        users must provide ``irt_column``.

    subject_column : str, default='subject'
        The column of the dataframe that contains the subject
        label of each event. The default value is ``subject``.
        All events must have a subject and group label (see
        ``grouping_column``).

    grouping_column : str, default='group'
        The column of the dataframe that contains the group label
        of each event. The default value is ``group``. The parameter
        provides flexibility to work with the events. The group
        label could refer to the group in which the subjects
        vary. However, as the label refers to the group of the
        event, one subject could have events with different labels.
        For example, the group label could refer to the side of the
        operant box in which the lever was pressed (``'left'`` or ``'right'``).
        Additionally, for longitudinal experimental designs, in which
        all subjects are exposed to the same interventions, the group
        label could refer to the condition of each event.
        During model fitting, subject parameters are the group-level
        value plus a subject-level deviation. The deviation (and its
        standard deviation, ``sigma_*``) is shared across groups, i.e.,
        it works as a random intercept per subject (see ``Notes`` section).

    event_column : str, default='event_alias'
        The column of the dataframe that contains the names of the
        events. The default value is ``event_alias``.

    session_index : str, default='session_id'
        The column of the dataframe that contains the unique
        identifier for sessions. The default value is
        ``session_id``.

    irt_column : str, default=None
        The column of the dataframe that contains the IRTs. The
        default value is ``None``. Only used if the dataframe was
        processed before. Rows with missing IRTs are dropped.

    refractory_period_column : str, default=None
        The column of the dataframe that contains the refractory
        period of each IRT. The default value is ``None``. It
        requires that the dataframe was processed before.

    filter_events : str or list of str, default=None
        Only events whose name appears in the provided string or
        list are kept. If set to the default value, ``None``, no
        filter is applied. A single string keeps only one event,
        and a list of strings keeps multiple events.

    Other Parameters
    ----------------
    irt_within_group : bool, default=True
        Only used when IRTs are computed from timestamps. If ``True``,
        IRTs are computed between consecutive events of the same
        session *and* the same group label (e.g., only between presses
        of the same lever). If ``False``, IRTs are computed between
        consecutive events of a session, and each IRT receives the group
        label of the second event.

    w_mode : {'offset', 'fmax'}, default='offset'
        Constraint type applied to the `w` parameter during fitting.
        For theoretical reasons, `w` must be greater than `b`:

            ``'offset'`` : `w` is parameterized as a positive offset
            added to `b`(i.e., ``b + exp(w_subj_cond)``).

            ``'fmax'`` : `w` is computed as the element-wise maximum
            of `b` and ``exp(w_subj_cond)``. This is not smooth and
            can hinder NUTS sampling.

    priors : dict, default=None
        Overrides of the default priors (see ``DEFAULT_PRIORS``).
        Only the keys provided are changed. Valid keys:

            ``'L_group'``, ``'w_group'``, ``'b_group'`` : ``(mu, sd)``
            of a Normal prior.

            ``'sigma_L'``, ``'sigma_w'``, ``'sigma_b'`` : sd of a
            HalfNormal prior.

        Sampling is done in the time unit of the data, and the default
        priors assume seconds. If the data are in another unit, rescale
        the data or adjust ``b_group`` and ``w_group``.
        Example: ``priors={'b_group': (-2.0, 1.0)}``.

    b_time_scale : float, default=60.0
        Factor used only to *report* ``b`` in ``summary_natural()``.
        With data in seconds, the default (60) reports ``b`` in bouts
        per minute. Use 1 to report ``b`` in the unit of the data
        (1/s). The model itself always samples ``b`` in 1/(time unit
        of the data).

    Notes
    -----
    The default parameter values follow the heuristics of the
    mteresa package. For example, when a MED-PC output folder is
    cleaned with ``read_medpc()``, each session is automatically
    saved with a unique ID in the ``session_id`` column, and the
    timestamp of each event in the ``timestamp`` column.

    Subject × group cells without observations are not identified
    by the data and keep only the prior.

    References
    ----------
    Brackney, R.J., Cheung, T.H.C., Neisewander, J.L. and
    Sanabria, F. (2011), THE ISOLATION OF MOTIVATIONAL, MOTORIC,
    AND SCHEDULE EFFECTS ON OPERANT PERFORMANCE: A MODELING APPROACH.
    Journal of the Experimental Analysis of Behavior, 96: 17-38.
    https://doi.org/10.1901/jeab.2011.96-17

    Brackney RJ, Cheung THC, Sanabria F. A bout analysis of operant
    response disruption. Behav Processes. 2017 Aug;141(Pt 1):42-49.
    doi: 10.1016/j.beproc.2017.04.008. Epub 2017 Apr 19. PMID:
    28431947; PMCID: PMC5501476.

    Falligant JM, Hagopian LP, Newland MC. Bouts, Pauses, and Units
    of Operant Performance: A Primer. Perspect Behav Sci. 2024 Aug
    30;47(3):643-674. doi: 10.1007/s40614-024-00419-z. PMID:
    39309239; PMCID: PMC11411050.

    Contact me
    ----------
    m.g.grael@gmail.com
    """

    def __init__(
        self,
        df,
        timestamp_column='timestamp',
        subject_column='subject',
        grouping_column='group',
        event_column='event_alias',
        session_index='session_id',
        irt_column=None,
        refractory_period_column=None,
        filter_events=None,
        w_mode='offset',
        irt_within_group=True,
        priors=None,
        b_time_scale=60.0,
    ):
        if w_mode not in ('fmax', 'offset'):
            raise ValueError("w_mode should be 'fmax' or 'offset'.")
        self.w_mode = w_mode

        priors = {} if priors is None else dict(priors)
        unknown = set(priors) - set(DEFAULT_PRIORS)
        if unknown:
            raise KeyError(
                f"Unknown prior(s): {sorted(unknown)}. Valid keys: {list(DEFAULT_PRIORS)}"
            )
        self.priors = {**DEFAULT_PRIORS, **priors}

        if b_time_scale <= 0:
            raise ValueError("b_time_scale must be positive.")
        self.b_time_scale = float(b_time_scale)

        try:
            import pymc  # noqa: F401
            import arviz  # noqa: F401
        except ImportError:
            raise ImportError(
                "Bayesian libraries not found. Run in your terminal:\n"
                '    pip install "mteresa[bayes]"'
            ) from None

        df = df.copy()
        df.columns = df.columns.str.lower()

        self.timestamp_column = timestamp_column.lower() if timestamp_column else None
        self.subject_column = subject_column.lower()
        self.grouping_column = grouping_column.lower()
        self.event_column = event_column.lower()
        self.session_index = session_index.lower()
        self.irt_column = irt_column.lower() if irt_column else None
        self.refractory_period_column = (
            refractory_period_column.lower() if refractory_period_column else None
        )

        if self.irt_column is None and self.timestamp_column is None:
            raise ValueError("Provide either timestamp_column or irt_column.")

        _require(df, [self.subject_column, self.grouping_column], 'always necessary')

        if filter_events is not None:
            _require(df, [self.event_column], 'used in filter_events')
            if isinstance(filter_events, str):
                filter_events = [filter_events]
            df = df[df[self.event_column].isin(filter_events)].copy()

        # ---- IRT ----
        if self.irt_column is None:
            _require(df, [self.timestamp_column, self.session_index], "IRT calculation")
            df = df.sort_values([self.session_index, self.timestamp_column])
            keys = [self.session_index]
            if irt_within_group:
                keys.append(self.grouping_column)
            df['irt'] = df.groupby(keys)[self.timestamp_column].diff()
            df = df.dropna(subset=['irt']).copy()
        else:
            _require(df, [self.irt_column], 'irt_column')
            df['irt'] = df[self.irt_column].astype(float)
            df = df.dropna(subset=['irt']).copy()

        # ---- Refractory period (d) ----
        if self.refractory_period_column is not None:
            _require(df, [self.refractory_period_column], 'refractory_period_column')
            df['d'] = df[self.refractory_period_column].astype(float)
        elif 'd' not in df.columns:
            # If no column is given and 'd' does not exist, assume 0.0
            df['d'] = 0.0

        self.df = df

        self.y = df['irt'].to_numpy(dtype=float)
        self.d = df['d'].to_numpy(dtype=float)
        y_shift = self.y - self.d
        if np.any(y_shift < -1e-9):
            n_bad = int((y_shift < -1e-9).sum())
            raise ValueError(f'{n_bad} observations with IRT < d.')
        self.y_shift = np.maximum(y_shift, 0.0)

        self.subject_idx, self.subject_labels = pd.factorize(df[self.subject_column])
        self.group_idx, self.group_labels = pd.factorize(df[self.grouping_column], sort=True)
        if (self.subject_idx < 0).any() or (self.group_idx < 0).any():
            raise ValueError("NaN found in subject_column or grouping_column.")

        S, G = len(self.subject_labels), len(self.group_labels)
        self.cell_idx = self.subject_idx * G + self.group_idx
        if self.cell_idx.min() < 0 or self.cell_idx.max() >= S * G:
            raise ValueError("Cell index out of range (subject x group).")

        self.coords = {
            'subject': np.asarray(self.subject_labels),
            'group': np.asarray(self.group_labels),
            'obs': np.arange(len(df)),
        }

        self.model = None
        self.idata = None

    def _build_model(self):
        import pymc as pm
        import pytensor.tensor as pt

        p = self.priors

        with pm.Model(coords=self.coords) as model:
            cell = pm.Data('cell', self.cell_idx, dims='obs')

            # --- L  ---
            sigma_L = pm.HalfNormal('sigma_L', p['sigma_L'])
            L_group = pm.Normal('L_group', *p['L_group'], dims='group')
            L_offset = pm.Normal('L_offset', 0, 1, dims='subject')
            L_subj_cond = pm.Deterministic(
                'L_subject_group',
                L_group[None, :] + L_offset[:, None] * sigma_L,
                dims=('subject', "group")
            )

            # --- w ---
            sigma_w = pm.HalfNormal('sigma_w', p['sigma_w'])
            w_group = pm.Normal('w_group', *p['w_group'], dims='group')
            w_offset = pm.Normal('w_offset', 0, 1, dims='subject')
            w_subj_cond = pm.Deterministic(
                'w_subject_group',
                w_group[None, :] + w_offset[:, None] * sigma_w,
                dims=('subject', 'group'),
            )

            # --- b ---
            sigma_b = pm.HalfNormal('sigma_b', p['sigma_b'])
            b_group = pm.Normal('b_group', *p['b_group'], dims='group')
            b_offset = pm.Normal('b_offset', 0, 1, dims='subject')
            b_subj_cond = pm.Deterministic(
                'b_subject_group',
                b_group[None, :] + b_offset[:, None] * sigma_b,
                dims=('subject', 'group'),
            )
            # --- Subjects parameters in natural scale ---
            # q = sigmoid(-L) = 1 / (exp(L) + 1); L is passed as logit to the
            # likelihood for numerical stability.
            b_s = pt.exp(b_subj_cond)
            if self.w_mode == 'fmax':
                w_s = pt.maximum(b_s, pt.exp(w_subj_cond))
            else:
                w_s = b_s + pt.exp(w_subj_cond)

            pm.CustomDist(
                'y',
                L_subj_cond.flatten()[cell],
                w_s.flatten()[cell],
                b_s.flatten()[cell],
                logp=biexp_logp,
                random=random_irt,
                observed=self.y_shift,
                dims='obs',
            )

        self.model = model
        return model

    def _save_idata(self, path):
        """Save idata as netCDF, falling back to pickle. Never raises."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.idata.to_netcdf(path)
            print(f"idata saved to {path}")
            return path
        except Exception as e:
            warnings.warn(f"netCDF save failed ({e}); trying pickle.")

        pkl_path = path.with_suffix('.pkl')
        try:
            with open(pkl_path, 'wb') as f:
                pickle.dump(self.idata, f)
            print(f"idata saved to {pkl_path} (pickle)")
            return pkl_path
        except Exception as e2:
            warnings.warn(f"pickle save also failed ({e2}); idata kept only in memory.")
            return None
    
    def fit(
        self,
        draws=2000,
        tune=2000,
        chains=4,
        cores=4,
        target_accept=0.95,
        max_treedepth=15,
        nuts_sampler='pymc',
        random_seed=42,
        compute_log_likelihood=False,
        posterior_predictive=False,
        initvals=None,
        save_path=None
    ):
        """
        Parameters
        ----------
        initvals : None, False or dict, default=None
            Starting point of the group parameters.
            ``None``: the mean of each prior. ``False``: no initial
            values are passed (PyMC chooses). dict: overrides only the
            keys given, e.g. ``{'b_group': -2.0}``. Valid keys:
            ``'L_group'``, ``'w_group'``, ``'b_group'``. Values are
            scalars applied to all groups.

        save_path : str or Path, default=None
            If given, the InferenceData is saved right after sampling
            and before ``compute_log_likelihood`` and
            ``posterior_predictive``, so a failure in those steps does
            not lose the posterior. Saved as netCDF, with pickle as
            fallback.
        """
        import pymc as pm

        if self.model is None:
            self._build_model()

        kwargs = {}
        if nuts_sampler == "pymc":
            kwargs["max_treedepth"] = max_treedepth

        if initvals is False:
            init = None
        else:
            G = len(self.group_labels)
            keys = ('L_group', 'w_group', 'b_group')
            base = {k: self.priors[k][0] for k in keys}
            user = {} if initvals is None else dict(initvals)
            unknown = set(user) - set(keys)
            if unknown:
                raise KeyError(f"Unknown initval(s): {sorted(unknown)}. Valid keys: {list(keys)}")
            base.update(user)
            init = {k: np.full(G, float(v)) for k, v in base.items()}

        with self.model:
            self.idata = pm.sample(
                draws=draws, tune=tune, chains=chains, cores=cores,
                target_accept=target_accept, nuts_sampler=nuts_sampler,
                random_seed=random_seed, initvals=init, **kwargs,
            )

            if save_path is not None:
                self._save_idata(save_path)

            if compute_log_likelihood:
                pm.compute_log_likelihood(self.idata)
            if posterior_predictive:
                pm.sample_posterior_predictive(
                    self.idata, extend_inferencedata=True, random_seed=random_seed
                )

        return self

    def summary(self):
        import arviz as az

        if self.idata is None:
            raise ValueError("Run .fit() before .summary().")
        var_names = [
            "L_group", "w_group", "b_group",
            "sigma_L", "sigma_w", "sigma_b",
        ]
        return az.summary(self.idata, var_names=var_names)

    def summary_natural(self):
        import arviz as az
        import xarray as xr

        if self.idata is None:
            raise ValueError("Run .fit() before .summary_natural().")
        post = self.idata.posterior

        # Group-level b and w in the unit of the data (1/s), consistent with w_mode
        b_rate = np.exp(post['b_group'])
        w_exp = np.exp(post['w_group'])
        if self.w_mode == 'offset':
            w_rate = b_rate + w_exp
        else:
            w_rate = np.maximum(b_rate, w_exp)

        # b is only *reported* in bouts per time (b_time_scale = 60 -> bouts/min)
        b_reported = b_rate * self.b_time_scale

        data = {
            'q_between': 1.0 / (1.0 + np.exp(post['L_group'])),
            'L_length': 1 + np.exp(post['L_group']),
            'b_bouts_per_min': b_reported,
            'w_rate': w_rate,
            'sigma_L': post['sigma_L'],
            'sigma_w_fator': np.exp(post["sigma_w"]),
            'sigma_b_fator': np.exp(post["sigma_b"]),
        }
        table = az.summary(xr.Dataset(data))

        if post.sizes["group"] > 1:
            ratios = {}
            for name, var in [("b", b_rate), ("w", w_rate)]:
                ratios[f"{name}_ratio_vs_ref"] = (
                    var.isel(group=slice(1, None)) / var.isel(group=0, drop=True)
                )
            table = pd.concat([table, az.summary(xr.Dataset(ratios))])

        return table

    def predictive_irt(self):
        """Posterior predictive samples on the original scale (adds d back)."""
        import xarray as xr

        if self.idata is None or not hasattr(self.idata, "posterior_predictive"):
            raise ValueError("Run .fit() with posterior_predictive=True first.")
        y_rep = self.idata.posterior_predictive["y"]
        return y_rep + xr.DataArray(self.d, dims='obs')

    def plot_survivor_function(self, scale='log_linear', xlim=None):
        """Plot the survivor function of IRTs for each group label in 'grouping_column'."""
        from math import ceil
        import matplotlib.pyplot as plt

        ncols = ceil(np.sqrt(len(self.group_labels)))
        nrows = ceil(len(self.group_labels) / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        axes_flat = np.atleast_1d(axes).flatten()

        for idx, grp in enumerate(self.group_labels):
            ax = axes_flat[idx]
            plot = self.df[self.df[self.grouping_column] == grp].copy()
            plot = plot.sort_values('irt', ascending=False).reset_index(drop=True)
            plot['i'] = plot.index + 1
            plot['surv'] = plot['i'] / len(plot)
            ax.scatter(x=plot['irt'], y=plot['surv'], facecolors='none', edgecolors='black')
            ax.set_yscale('log')
            ax.set_xlim(left=0)
            ax.text(0.98, 0.98, grp, transform=ax.transAxes, va='top', ha='right')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(0.001, 1)

            if scale == 'log_log':
                ax.set_xlim(left=0.1)
                ax.set_xscale('log')
            elif scale == 'log_linear':
                ax.set_xlim(left=0)
                ax.set_xscale('linear')

            if xlim is not None:
                ax.set_xlim(xlim)

        # Hide extra subplots if nrows * ncols > number of groups
        for idx in range(len(self.group_labels), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.supylabel('Relative Frequency')
        fig.supxlabel('IRTs')
        fig.tight_layout()
        plt.show()
        return None


if __name__ == '__main__':
    df = pd.read_csv(Path('sandbox') / 'matthew.csv')
    print(df.head())
    model = BiExponentialRefractoryModel(df, irt_column='IRT', grouping_column='g')
    model.plot_survivor_function()
    # results = model.fit(tune=400, draws=400)
    # print(results.summary())
    # print(results.summary_natural())