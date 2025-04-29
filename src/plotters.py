from __future__ import annotations
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
        if pd.isna(row["phd_match_by"]):
            return "Not found in Open Alex"
        if not row["phd_match_score"]:
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
