from __future__ import annotations
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _as_bool(value) -> bool:
    """Interpret CSV flags without treating the string 'False' as truthy."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "1.0"}
    return pd.notna(value) and bool(value)


def _as_list(value) -> list:
    """Restore collections written by DataFrame.to_csv using safe literal parsing."""
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return list(value)
    return [] if pd.isna(value) else [value]


def _normalize_plot_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Give native extraction rows and CSV-loaded rows the same chart input types."""
    result = df.copy()
    if "phd_match_by" in result:
        result["phd_match_by"] = result["phd_match_by"].fillna("").astype(str).str.strip()
    if "phd_id" in result:
        result["phd_id"] = result["phd_id"].replace(r"^\s*$", None, regex=True)
    for column in ("affiliation_match", "near_exact_match", "exact_match", "contributor_confirmed"):
        if column in result:
            result[column] = result[column].map(_as_bool).astype(bool)
    for column in ("name_matches_open_alex", "contributor_name", "contributor_id", "contributor_orcid"):
        if column in result:
            result[column] = result[column].map(_as_list)
    for column in ("n_shared_pubs", "n_shared_inst_grad"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)
    return result


class PhDMatchPlotter:
    """
    Encapsulates the data wrangling and plotting logic for the
    PhD-matching confirmation chart.

    Parameters
    ----------
    df : pd.DataFrame
        Raw extraction frame containing at least the columns listed
        in `_REQUIRED_COLS`.
    intra_gap : float, default 0.6
        Horizontal space between bars that belong to the *same* group.
    inter_gap : float, default 1.2
        Horizontal space between the two groups (confirmed / not confirmed).
    bar_width : float, default 0.5
        Thickness of each bar.
    color_confirmed : str, default "tab:blue"
        Face color for confirmed bars.
    color_not_confirmed : str, default "tab:red"
        Face color for not-confirmed bars.
    """

    _REQUIRED_COLS: tuple[str, ...] = (
        "phd_name", "phd_id", "phd_match_by", "phd_match_score",
        "affiliation_match", "near_exact_match", "exact_match"
    )

    _BAR_CATEGORIES: list[str] = [
        "Not found in Open Alex",
        "Found, but no other match",
        "Fuzzy title match only",
        "Fuzzy title match + affiliation",
        "Near exact title match only",
        "Near exact title match + affiliation",
        "Exact title match only",
        "Exact title match + affiliation",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        intra_gap: float = 0.6,
        inter_gap: float = 1.2,
        bar_width: float = 0.5,
        color_confirmed: str = "tab:blue",
        color_not_confirmed: str = "tab:red",
    ) -> None:
        if not set(self._REQUIRED_COLS).issubset(df.columns):
            missing = set(self._REQUIRED_COLS) - set(df.columns)
            raise ValueError(f"DataFrame missing columns: {missing}")

        self.df = _normalize_plot_frame(df)
        self.intra_gap = intra_gap
        self.inter_gap = inter_gap
        self.bar_width = bar_width
        self.color_confirmed = color_confirmed
        self.color_not_confirmed = color_not_confirmed

        self._prepare()

    # -----------------------------------------------------------------
    # public helpers
    # -----------------------------------------------------------------
    @property
    def n_phds(self) -> int:
        return len(self.df_unique)

    @property
    def n_confirmed(self) -> int:
        return self.df_unique["match_category"].isin(self.confirmed_categories).sum()

    def plot(self, *, ax: plt.Axes | None = None) -> plt.Axes:
        """Render the bar chart; return the `Axes` used."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        bars = ax.bar(
            self.x_pos,
            self.match_counts.values,
            width=self.bar_width,
            color=self.colors,
            edgecolor="black",
        )
        ax.bar_label(bars, label_type="edge")

        # ticks / labels
        ax.set_xticks(self.x_pos, self._BAR_CATEGORIES, rotation=90)
        ax.set_ylabel("Number of PhDs")
        ax.set_xlabel("Match Category")
        ax.set_title(
            f"Confirmed {self.n_confirmed} out of {self.n_phds} PhDs", fontsize=10
        )
        fig.suptitle("PhD Matching Confirmation by Category", fontsize=12)

        # legend
        ax.legend(
            handles=[
                plt.Line2D([], [], marker="s", linestyle="", color=self.color_not_confirmed,
                           label="Not confirmed"),
                plt.Line2D([], [], marker="s", linestyle="", color=self.color_confirmed,
                           label="Confirmed"),
            ],
            loc="upper left",
        )

        return ax

    # -----------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------
    def _prepare(self) -> None:
        
        self.df_unique = (
            self.df.loc[:, self._REQUIRED_COLS] # select the required columns only
                .drop_duplicates(subset=["phd_name", "phd_id"]) # unique rows per PhD. With this subset, we can distinguish between PhDs with the same name, but different ids (if we found them) 
                .copy()
        )
        # categorize
        self.df_unique["match_category"] = self.df_unique.apply(
            self._determine_category_row, axis=1
        )

        # counts & colors
        self.match_counts = (
            self.df_unique["match_category"]
            .value_counts()
            .reindex(self._BAR_CATEGORIES, fill_value=0)
        )
        self.confirmed_categories = self._BAR_CATEGORIES[2:]

        self.colors = (
            [self.color_not_confirmed] * 2 + [self.color_confirmed] * 6
        )
        self.x_pos = self._make_positions()

    def _make_positions(self) -> list[float]:
        """Custom x positions with narrower intra-group spacing."""
        x_vals: list[float] = []
        x = 0.0
        for idx in range(len(self._BAR_CATEGORIES)):
            x_vals.append(x)
            x += self.inter_gap if idx == 1 else self.intra_gap
        return x_vals

    @staticmethod
    def _determine_category_row(row: pd.Series) -> str:
        if pd.isna(row["phd_id"]) :
            return "Not found in Open Alex"
        if not row["phd_match_by"]:
            return "Found, but no other match"
        if not row["exact_match"] and not row["affiliation_match"] and not row["near_exact_match"]:
            return "Fuzzy title match only"
        if not row["exact_match"] and row["affiliation_match"] and not row["near_exact_match"]:
            return "Fuzzy title match + affiliation"
        if row["near_exact_match"] and not row["affiliation_match"] and not row["exact_match"]:
            return "Near exact title match only"
        if row["near_exact_match"] and row["affiliation_match"] and not row["exact_match"]:
            return "Near exact title match + affiliation"
        if row["exact_match"] and not row["affiliation_match"]:
            return "Exact title match only"
        if row["exact_match"] and row["affiliation_match"]:
            return "Exact title match + affiliation"
        return "Other"


# ---------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------
# plotter = PhDMatchPlotter(extraction_df)
# ax = plotter.plot()
# plt.show()


class ContributorMatchPlotter:
    """
    Encapsulates the data wrangling and plotting logic for the
    contributor-matching confirmation chart (one row = one potential
    contributor–PhD link).

    Counts and colors use the exported ``contributor_confirmed`` decision.
    Stacked bars show confirmation separately from publication and shared-
    affiliation evidence; ``same_grad_inst`` is not a confirmation rule.

    Parameters
    ----------
    df : pd.DataFrame
        Extraction frame; must contain the columns listed in
        `_REQUIRED_COLS`. PhDs are grouped by source `integer_id` when
        available, otherwise by `phd_id`.
    intra_gap : float, default 0.6
        Horizontal space between bars that belong to the *same* group.
    inter_gap : float, default 1.2
        Horizontal space between missing-match and evidence categories.
    bar_width : float, default 0.5
        Thickness of each bar.
    color_confirmed : str, default "tab:orange"
        Face color for confirmed bars.
    color_not_confirmed : str, default "tab:red"
        Face color for not-confirmed bars.
    """

    _REQUIRED_COLS: tuple[str, ...] = (
        "phd_id",
        "phd_match_by",
        "contributor_name",
        "name_matches_open_alex",
        "contributor_confirmed",
        "n_shared_pubs",
        "n_shared_inst_grad",
    )

    _BAR_CATEGORIES: list[str] = [
        "PhD not confirmed",
        "No contributor name match in OpenAlex",
        "No shared publications or affiliation in target years",
        "Shared affiliation in target years only",
        "Shared publications only",
        "Shared publications and affiliation in target years",
    ]

    # -----------------------------------------------------------------
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        intra_gap: float = 0.6,
        inter_gap: float = 1.2,
        bar_width: float = 0.5,
        color_confirmed: str = "tab:orange",
        color_not_confirmed: str = "tab:red",
    ) -> None:
        if not set(self._REQUIRED_COLS).issubset(df.columns):
            missing = set(self._REQUIRED_COLS) - set(df.columns)
            raise ValueError(f"DataFrame missing columns: {missing}")

        self.df = _normalize_plot_frame(df)
        self.intra_gap = intra_gap
        self.inter_gap = inter_gap
        self.bar_width = bar_width
        self.color_confirmed = color_confirmed
        self.color_not_confirmed = color_not_confirmed

        self._prepare()

    # -----------------------------------------------------------------
    # public helpers
    # -----------------------------------------------------------------
    @property
    def n_contributors(self) -> int:
        return len(self.contrib_df)

    @property
    def n_confirmed_contributors(self) -> int:
        return self.contrib_df["contributor_confirmed"].sum()

    @property
    def n_phds_with_confirmed(self) -> int:
        return self.phd_flags["has_confirmed_contributor"].sum()

    # -----------------------------------------------------------------
    # main plot
    # -----------------------------------------------------------------
    def plot(self, *, ax: plt.Axes | None = None) -> plt.Axes:
        """Render the bar chart; return the `Axes` used."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.bar(
            self.x_pos,
            self.unconfirmed_counts.values,
            width=self.bar_width,
            color=self.color_not_confirmed,
            edgecolor="black",
        )
        bars = ax.bar(
            self.x_pos,
            self.confirmed_counts.values,
            bottom=self.unconfirmed_counts.values,
            width=self.bar_width,
            color=self.color_confirmed,
            edgecolor="black",
        )
        ax.bar_label(bars, labels=self.match_counts.values, label_type="edge")

        ax.set_xticks(self.x_pos, self._BAR_CATEGORIES, rotation=90)
        ax.set_ylabel("Number of contributors")
        ax.set_xlabel("Match Type")
        ax.set_title(
            f"Confirmed {self.n_confirmed_contributors} out of "
            f"{self.n_contributors} contributors\n"
            f"{self.n_phds_with_confirmed} PhDs have ≥1 confirmed contributor",
            fontsize=10,
        )
        fig.suptitle("Contributor Matching Confirmation by Type", fontsize=12)
        fig.subplots_adjust(top=0.85, bottom=0.28, right=0.97)


        ax.legend(
            handles=[
                plt.Line2D([], [], marker="s", linestyle="", color=self.color_not_confirmed,
                           label="Not confirmed"),
                plt.Line2D([], [], marker="s", linestyle="", color=self.color_confirmed,
                           label="Confirmed"),
            ],
            loc="upper left",
        )

        # ensure edge bars are fully visible
        ax.set_xlim(self.x_pos[0] - self.bar_width,
                    self.x_pos[-1] + self.bar_width)
        fig.subplots_adjust(right=0.97, bottom=0.28)
        return ax

    # -----------------------------------------------------------------
    # internals
    # -----------------------------------------------------------------
    def _determine_category_row(self, row: pd.Series) -> str:
        if not row["phd_match_by"]:
            return "PhD not confirmed"
        if not row["name_matches_open_alex"]:
            return "No contributor name match in OpenAlex"
        has_publications = pd.notna(row["n_shared_pubs"]) and row["n_shared_pubs"] > 0
        has_affiliation = (
            pd.notna(row["n_shared_inst_grad"]) and row["n_shared_inst_grad"] > 0
        )
        if not has_publications and not has_affiliation:
            return "No shared publications or affiliation in target years"
        if not has_publications and has_affiliation:
            return "Shared affiliation in target years only"
        if has_publications and not has_affiliation:
            return "Shared publications only"
        return "Shared publications and affiliation in target years"

    def _make_positions(self) -> list[float]:
        """Custom x positions: narrow gaps within a group, wider gap between."""
        x_vals: list[float] = []
        x = 0.0
        for idx in range(len(self._BAR_CATEGORIES)):
            x_vals.append(x)
            x += self.inter_gap if idx == 1 else self.intra_gap
        return x_vals

    def _prepare(self) -> None:
        # -----------------------------------------------------------------
        # 1) categorise each row
        # -----------------------------------------------------------------
        self.df["match_category"] = pd.Series(
            [self._determine_category_row(row) for _, row in self.df.iterrows()],
            index=self.df.index, dtype=object,
        )

        # -----------------------------------------------------------------
        # 2) contributor-level frame
        # -----------------------------------------------------------------
        phd_key = "integer_id" if "integer_id" in self.df.columns else "phd_id"
        self.contrib_df = self.df[list(dict.fromkeys([
            phd_key, "phd_id", "match_category", "contributor_name", "contributor_confirmed",
        ]))].copy()

        # -----------------------------------------------------------------
        # 3) flags per PhD: any confirmed contributor?
        # -----------------------------------------------------------------
        row_is_confirmed = self.contrib_df["contributor_confirmed"]
        self.phd_flags = (
            row_is_confirmed.groupby(self.contrib_df[phd_key], dropna=False).any()
            .rename("has_confirmed_contributor")
            .reset_index()
        )

        # -----------------------------------------------------------------
        # 4) bar heights
        # -----------------------------------------------------------------
        self.confirmed_counts = (
            self.contrib_df.loc[row_is_confirmed, "match_category"]
            .value_counts()
            .reindex(self._BAR_CATEGORIES, fill_value=0)
        )
        self.unconfirmed_counts = (
            self.contrib_df.loc[~row_is_confirmed, "match_category"]
            .value_counts()
            .reindex(self._BAR_CATEGORIES, fill_value=0)
        )
        self.match_counts = self.confirmed_counts + self.unconfirmed_counts

        # -----------------------------------------------------------------
        # 5) x-positions (confirmation colors are applied to each stack)
        # -----------------------------------------------------------------
        self.x_pos = self._make_positions()


# -----------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------
# contrib_plotter = ContributorMatchPlotter(extraction_df)
# ax = contrib_plotter.plot()
# plt.show()
