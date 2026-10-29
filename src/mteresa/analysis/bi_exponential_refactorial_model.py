import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# Funções da verossimilhança (ordem dos parâmetros: q, w, b)
# ---------------------------------------------------------------
def random_irt(q, w, b, rng=None, size=None):
    """Sorteia IRTs (na escala de Y - D) da mistura de duas exponenciais."""
    if rng is None:
        rng = np.random.default_rng()
    q, w, b = np.broadcast_arrays(q, w, b)
    shape = size if size is not None else q.shape
    is_between = rng.binomial(1, q, size=shape).astype(bool)
    between_draw = rng.exponential(1.0 / b, size=shape)  # numpy usa escala = 1/taxa
    within_draw = rng.exponential(1.0 / w, size=shape)
    return np.where(is_between, between_draw, within_draw)


def biexp_logp(value, q, w, b):
    """Log-densidade da mistura: q*Exp(b) + (1-q)*Exp(w)."""
    import pytensor.tensor as pt

    log_between = pt.log(q) + pt.log(b) - b * value
    log_within = pt.log1p(-q) + pt.log(w) - w * value
    return pt.logaddexp(log_between, log_within)


# ---------------------------------------------------------------
# Classe do modelo
# ---------------------------------------------------------------
class BiExponentialRefactorialModel:
    def __init__(
        self,
        df,
        timestamp_column="timestamp",
        subject_column="subject",
        grouping_column="group",
        event_column="event_alias",
        session_index="session_id",
        filter_events=None,
    ):
        df = df.copy()

        # Filtro de eventos (opcional)
        if filter_events is not None:
            if isinstance(filter_events, str):
                filter_events = [filter_events]
            df = df[df[event_column].isin(filter_events)]

        # Timestamp numérico (segundos), caso venha como data/hora
        if pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            df[timestamp_column] = (
                df[timestamp_column] - df[timestamp_column].min()
            ).dt.total_seconds()

        # IRT = diferença entre eventos consecutivos dentro da sessão
        df = df.sort_values([session_index, timestamp_column])
        df["IRT"] = df.groupby(session_index)[timestamp_column].diff()
        df = df.dropna(subset=["IRT"]).reset_index(drop=True)

        # D = menor IRT de cada sujeito (estimado dos dados, tratado como conhecido)
        df["D"] = df.groupby(subject_column)["IRT"].transform("min")

        self.df = df
        self.y = df["IRT"].to_numpy(dtype=float)
        self.d = df["D"].to_numpy(dtype=float)
        self.y_shift = self.y - self.d
        assert np.all(self.y_shift >= 0), "Existem observações com IRT < D."

        # Índice de sujeito por observação (tamanho N)
        self.subject_idx, self.subject_labels = pd.factorize(df[subject_column])

        # Índice de grupo por sujeito (tamanho S), alinhado a subject_labels
        n_groups_per_subject = df.groupby(subject_column)[grouping_column].nunique()
        if n_groups_per_subject.max() > 1:
            raise ValueError("Há sujeitos associados a mais de um grupo.")
        subject_group = (
            df.groupby(subject_column)[grouping_column]
            .first()
            .reindex(self.subject_labels)
        )
        self.group_idx, self.group_labels = pd.factorize(subject_group, sort=True)

        self.coords = {
            "subject": np.asarray(self.subject_labels),
            "group": np.asarray(self.group_labels),
            "obs": np.arange(len(df)),
        }

        self.model = None
        self.idata = None

    def _build_model(self):
        import pymc as pm
        import pytensor.tensor as pt

        with pm.Model(coords=self.coords) as model:
            subj = pm.Data("subj", self.subject_idx, dims="obs")
            grp = pm.Data("grp", self.group_idx, dims="subject")

            # --- L (log-odds de within sobre between) ---
            sigma_L = pm.HalfNormal("sigma_L", 5)
            L_group = pm.Normal("L_group", 0, 10, dims="group")
            L_offset = pm.Normal("L_offset", 0, 1, dims="subject")
            L_subject = pm.Deterministic(
                "L_subject", L_group[grp] + L_offset * sigma_L, dims="subject"
            )

            # --- w (log da taxa within-bout) ---
            sigma_w = pm.HalfNormal("sigma_w", 5)
            w_group = pm.Normal("w_group", 0, 10, dims="group")
            w_offset = pm.Normal("w_offset", 0, 1, dims="subject")
            w_subject = pm.Deterministic(
                "w_subject", w_group[grp] + w_offset * sigma_w, dims="subject"
            )

            # --- b (log da taxa between-bout) ---
            sigma_b = pm.HalfNormal("sigma_b", 5)
            b_group = pm.Normal("b_group", 0, 10, dims="group")
            b_offset = pm.Normal("b_offset", 0, 1, dims="subject")
            b_subject = pm.Deterministic(
                "b_subject", b_group[grp] + b_offset * sigma_b, dims="subject"
            )

            # --- Parâmetros por sujeito na escala natural ---
            q_s = pm.math.sigmoid(-L_subject)  # = 1 / (exp(L) + 1)
            b_s = pt.exp(b_subject)
            w_s = b_s + pt.exp(w_subject)

            # --- Verossimilhança (ordem: q, w, b, igual às funções) ---
            pm.CustomDist(
                "y",
                q_s[subj],
                w_s[subj],
                b_s[subj],
                logp=biexp_logp,
                random=random_irt,
                observed=self.y_shift,
                dims="obs",
            )

        self.model = model
        return model

    def fit(
        self,
        draws=2000,
        tune=2000,
        chains=4,
        cores=4,
        target_accept=0.95,
        nuts_sampler="pymc",
        random_seed=42,
    ):
        import pymc as pm

        if self.model is None:
            self._build_model()

        with self.model:
            self.idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                nuts_sampler=nuts_sampler,
                idata_kwargs={"log_likelihood": True},
                random_seed=random_seed,
            )

            # Alguns amostradores não guardam o log_likelihood: calcula se faltar
            if "log_likelihood" not in self.idata.groups():
                pm.compute_log_likelihood(self.idata)

            pm.sample_posterior_predictive(
                self.idata, extend_inferencedata=True, random_seed=random_seed
            )
        return self

    def summary(self):
        import arviz as az

        if self.idata is None:
            raise ValueError("Rode .fit() antes de .summary().")
        var_names = [
            "L_group", "w_group", "b_group",
            "sigma_L", "sigma_w", "sigma_b",
        ]
        return az.summary(self.idata, var_names=var_names)

    def predictive_irt(self):
        """Amostras preditivas na escala original (soma D de volta)."""
        if self.idata is None:
            raise ValueError("Rode .fit() antes.")
        y_rep = self.idata.posterior_predictive["y"]
        return y_rep + self.d


if __name__ == "__main__":
    df = pd.read_csv(r"sandbox\graja.csv")
    model = BiExponentialRefactorialModel(df=df[df['session_id'] > 1200])
    print(model.df[["session_id", 'group', "subject", "IRT", "D"]])
    print(model.df["IRT"].describe())
    print(model.y_shift.min())

    model.fit(chains=4, tune=200, draws=200)
    print(model.summary())