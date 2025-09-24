from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

        self.df = df.copy()
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
        self.df_unique["affiliation_match"] = (
            self.df_unique["affiliation_match"]
                .fillna(False)
                .astype(bool)
        )
        self.df_unique["exact_match"] = (
            self.df_unique["exact_match"]
                .fillna(False)
                .astype(bool)
        )
        
        # categorize
        self.df_unique["match_category"] = self.df_unique.apply(
            self._determine_category_row, axis=1
        )

        # counts & colors
        self.match_counts = (
            self.df_unique["match_category"]
            .value_counts()
            .reindex(self._BAR_CATEGORIES)
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

    Parameters
    ----------
    df : pd.DataFrame
        Extraction frame; must contain the columns listed in
        `_REQUIRED_COLS`.
    intra_gap : float, default 0.6
        Horizontal space between bars that belong to the *same* group.
    inter_gap : float, default 1.2
        Horizontal space between the two groups (confirmed / not confirmed).
    bar_width : float, default 0.5
        Thickness of each bar.
    color_confirmed : str, default "tab:orange"
        Face color for confirmed bars.
    color_not_confirmed : str, default "tab:red"
        Face color for not-confirmed bars.
    """

    _REQUIRED_COLS: tuple[str, ...] = (
        "phd_id",
        "phd_name",
        "contributor_id",
        "contributor_name",
        "n_shared_pubs",
        "same_grad_inst",
    )

    _BAR_CATEGORIES: list[str] = [
        "Not found",
        "No shared publications or affiliation at graduation",
        "Shared affiliation at graduation only",
        "Shared publications only",
        "Shared publications and affiliation at graduation",
        "Other",
    ]
    _FIRST_CONFIRMED = 3  # bars 0-2  = not confirmed, 3-5 = confirmed

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

        self.df = df.copy()
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
        return self.contrib_df["match_category"].isin(self.confirmed_categories).sum()

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

        bars = ax.bar(
            self.x_pos,
            self.match_counts.values,
            width=self.bar_width,
            color=self.colors,
            edgecolor="black",
        )
        ax.bar_label(bars, label_type="edge")

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
    @staticmethod
    def _missing_or_empty(val) -> bool:
        """True if val is NaN / None or an empty iterable/ndarray."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return True
        if isinstance(val, (list, tuple, set, np.ndarray)):
            return len(val) == 0
        return False

    def _determine_category_row(self, row: pd.Series) -> str:
        if self._missing_or_empty(row["contributor_id"]):
            return "Not found"
        if not row["n_shared_pubs"] and not row["same_grad_inst"]:
            return "No shared publications or affiliation at graduation"
        if not row["n_shared_pubs"] and row["same_grad_inst"]:
            return "Shared affiliation at graduation only"
        if row["n_shared_pubs"] and not row["same_grad_inst"]:
            return "Shared publications only"
        if row["n_shared_pubs"] and row["same_grad_inst"]:
            return "Shared publications and affiliation at graduation"
        return "Other"

    def _make_positions(self) -> list[float]:
        """Custom x positions: narrow gaps within a group, wider gap between."""
        x_vals: list[float] = []
        x = 0.0
        for idx in range(len(self._BAR_CATEGORIES)):
            x_vals.append(x)
            x += self.inter_gap if idx == self._FIRST_CONFIRMED - 1 else self.intra_gap
        return x_vals

    def _prepare(self) -> None:
        # -----------------------------------------------------------------
        # 1) categorise each row
        # -----------------------------------------------------------------
        self.df["match_category"] = self.df.apply(self._determine_category_row, axis=1)

        # -----------------------------------------------------------------
        # 2) contributor-level frame (ignore rows without phd_id)
        # -----------------------------------------------------------------
        self.contrib_df = self.df.loc[self.df["phd_id"].notna(),
                                      ["phd_id", "match_category", "contributor_name"]]

        # -----------------------------------------------------------------
        # 3) flags per PhD: any confirmed contributor?
        # -----------------------------------------------------------------
        self.confirmed_categories = self._BAR_CATEGORIES[self._FIRST_CONFIRMED:]
        row_is_confirmed = self.contrib_df["match_category"].isin(self.confirmed_categories)
        self.phd_flags = (
            row_is_confirmed.groupby(self.contrib_df["phd_id"]).any()
            .rename("has_confirmed_contributor")
            .reset_index()
        )

        # -----------------------------------------------------------------
        # 4) bar heights
        # -----------------------------------------------------------------
        self.match_counts = (
            self.contrib_df["match_category"]
            .value_counts()
            .reindex(self._BAR_CATEGORIES)
            .fillna(0)
        )

        # -----------------------------------------------------------------
        # 5) colours & x-positions
        # -----------------------------------------------------------------
        self.colors = (
            [self.color_not_confirmed] * self._FIRST_CONFIRMED +
            [self.color_confirmed] * (len(self._BAR_CATEGORIES) - self._FIRST_CONFIRMED)
        )
        self.x_pos = self._make_positions()


# -----------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------
# contrib_plotter = ContributorMatchPlotter(extraction_df)
# ax = contrib_plotter.plot()
# plt.show()