"""Offline, thesis-aligned comparison of extracted and manually validated identities."""

from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd


EMPTY = frozenset()
MISSING_TEXT = {"", "nan", "none", "null", "<na>"}
PHD_OUTCOMES = [
    "Correct confirmed ID", "Wrong confirmed ID", "Not confirmed",
    "Missing extraction", "No manual PhD ID (unscored)",
]


def text(value) -> str:
    if value is None or (not isinstance(value, (list, tuple, set, dict)) and pd.isna(value)):
        return ""
    value = str(value).strip()
    return "" if value.lower() in MISSING_TEXT else value


def sequence(value) -> list:
    """Restore CSV list/dict literals without executing their contents."""
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    value = text(value)
    if not value:
        return []
    if value.startswith(("[", "(")):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Invalid serialized collection: {value!r}") from exc
        if not isinstance(parsed, (list, tuple, set)):
            raise ValueError(f"Expected a collection: {value!r}")
        return list(parsed)
    return [value]


def author_ids(value) -> frozenset[str]:
    """Accept bare IDs and OpenAlex URLs, including /authors/a... variants."""
    result = set()
    for item in sequence(value):
        item = text(item)
        if not item:
            continue
        match = re.fullmatch(
            r"(?:https?://(?:www\.)?openalex\.org/(?:authors/)?)?(A\d+)/?", item, re.I,
        )
        if not match:
            raise ValueError(f"Invalid OpenAlex author ID: {item!r}")
        result.add(match.group(1).upper())
    return frozenset(result)


def orcids(value) -> frozenset[str]:
    result = set()
    for item in sequence(value):
        item = text(item)
        if not item:
            continue
        match = re.fullmatch(
            r"(?:https?://(?:www\.)?orcid\.org/)?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])/?",
            item, re.I,
        )
        if not match:
            raise ValueError(f"Invalid ORCID: {item!r}")
        result.add(match.group(1).upper())
    return frozenset(result)


def flag(value) -> bool:
    value = text(value).lower()
    if value in {"true", "1", "1.0"}:
        return True
    if value in {"", "false", "0", "0.0"}:
        return False
    raise ValueError(f"Unexpected boolean flag: {value!r}")


def source_id(value) -> str:
    value = text(value)
    if re.fullmatch(r"\d+(?:\.0+)?", value):
        return value.split(".")[0]
    raise ValueError(f"Expected an integer thesis integer_id, got {value!r}")


def _require(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def recover_manual_keys(raw: pd.DataFrame, automated: pd.DataFrame):
    """Recover absent source keys only from unique, exact thesis URL matches.

    Wholly blank spreadsheet rows are removed. An unresolved populated row
    raises an error rather than silently disappearing from the denominator.
    """
    _require(raw, ["integer_id", "thesis_identifier"])
    _require(automated, ["integer_id", "thesis_identifier"])
    result = raw.copy().fillna("")
    result = result.loc[result.apply(lambda col: col.map(text)).ne("").any(axis=1)].copy()
    lookup = automated[["thesis_identifier", "integer_id"]].copy()
    lookup["thesis_identifier"] = lookup.thesis_identifier.map(text)
    lookup["integer_id"] = lookup.integer_id.map(source_id)
    lookup = lookup.loc[lookup.thesis_identifier.ne("")].drop_duplicates()
    ambiguous = set(lookup.loc[lookup.thesis_identifier.duplicated(False), "thesis_identifier"])
    lookup = lookup.loc[~lookup.thesis_identifier.isin(ambiguous)].set_index("thesis_identifier").integer_id
    audit = []
    for index, row in result.loc[result.integer_id.map(text).eq("")].iterrows():
        identifier = text(row.thesis_identifier)
        if identifier not in lookup:
            raise ValueError(f"Manual row {index} lacks integer_id and has no unique exact thesis URL match.")
        result.at[index, "integer_id"] = lookup[identifier]
        audit.append({"csv_row": index + 2, "integer_id": lookup[identifier],
                      "thesis_identifier": identifier, "method": "Unique exact thesis URL"})
    return result, pd.DataFrame(audit, columns=["csv_row", "integer_id", "thesis_identifier", "method"])


def clean_manual_identifiers(raw: pd.DataFrame):
    """Audit unambiguous URL repairs and exclude malformed secondary ORCIDs.

    Truncated ORCIDs are never guessed. Invalid author IDs still raise an error,
    since they define the primary comparison rather than an auxiliary check.
    """
    result = raw.copy()
    audit = []
    for column in result:
        is_author = "open_alex_id" in column
        is_orcid = "orcid" in column
        if not (is_author or is_orcid):
            continue
        for index, value in result[column].items():
            original = text(value)
            cleaned = original
            if cleaned.startswith("ttps://"):
                cleaned = "h" + cleaned
            if is_orcid and cleaned.startswith('orcid":"'):
                cleaned = cleaned[len('orcid":"'):]
            action = "Repaired URL text"
            try:
                (author_ids if is_author else orcids)(cleaned)
            except ValueError:
                if is_author:
                    raise
                cleaned = ""
                action = "Excluded malformed ORCID; no digits inferred"
            if cleaned != original:
                result.at[index, column] = cleaned
                audit.append({"csv_row": index + 2, "integer_id": result.at[index, "integer_id"],
                              "column": column, "original": original,
                              "normalized": cleaned, "action": action})
    return result, pd.DataFrame(audit, columns=[
        "csv_row", "integer_id", "column", "original", "normalized", "action",
    ])


def prepare_manual(raw: pd.DataFrame, alternative_supervisor_slots=None):
    """Keep original annotations, and apply only explicitly mapped corrections.

    alternative_supervisor_slots maps source integer_id to supervisor slot (1..5).
    These alternatives replace the slot's superseded ID; they are never assumed
    to be alternative PhD identities. Unmapped alternatives remain audit items.
    """
    _require(raw, ["integer_id", "author_name", "open_alex_id_candidate"] + [
        column for slot in range(1, 6)
        for column in (f"supervisor_{slot}", f"open_alex_id_supervisor_{slot}")
    ])
    manual = raw.copy().fillna("")
    manual = manual.loc[manual.apply(lambda col: col.map(text)).ne("").any(axis=1)].copy()
    manual["integer_id"] = manual.integer_id.map(source_id)
    if manual.integer_id.duplicated().any():
        raise ValueError("Manual validation must contain one row per integer_id.")

    corrections = {source_id(key): slot for key, slot in (alternative_supervisor_slots or {}).items()}
    unknown = set(corrections) - set(manual.integer_id)
    if unknown:
        raise ValueError(f"Alternative mappings refer to absent theses: {sorted(unknown)}")
    records, audit = [], []
    for row in manual.to_dict("records"):
        key = row["integer_id"]
        phd_ids = author_ids(row["open_alex_id_candidate"])
        sup_ids, sup_orcids, unresolved = set(), set(), 0
        named = 0
        slot_override = corrections.get(key)
        if slot_override is not None and slot_override not in range(1, 6):
            raise ValueError(f"Invalid supervisor slot for {key}: {slot_override}")
        alternative = text(row.get("alternative_open_alex_id"))
        if slot_override is not None and not alternative:
            raise ValueError(f"No alternative OpenAlex ID available for {key}")
        for slot in range(1, 6):
            ids = author_ids(row[f"open_alex_id_supervisor_{slot}"])
            slot_orcids = orcids(row.get(f"orcid_supervisor_{slot}"))
            name = text(row[f"supervisor_{slot}"])
            if slot == slot_override:
                ids = author_ids(alternative)
                slot_orcids = orcids(row.get("alternative_orcid"))
            named += bool(name or ids)
            unresolved += bool(name and not ids)
            sup_ids.update(ids)
            sup_orcids.update(slot_orcids)
        row.update(
            reference_phd_ids=phd_ids,
            reference_phd_orcids=orcids(row.get("orcid_candidate")),
            reference_supervisor_ids=frozenset(sup_ids),
            reference_supervisor_orcids=frozenset(sup_orcids),
            n_manual_supervisors=named,
            n_unresolved_supervisors=unresolved,
            reference_supervisors_complete=bool(named and not unresolved),
        )
        records.append(row)
        if alternative:
            audit.append({
                "integer_id": key, "supervisor_slot": slot_override,
                "original_id": row.get(f"open_alex_id_supervisor_{slot_override}", ""),
                "alternative_id": alternative,
                "action": "Replaced supervisor ID" if slot_override else "Unmapped: review required",
                "notes": row.get("notes", ""),
            })
    return pd.DataFrame(records), pd.DataFrame(audit, columns=[
        "integer_id", "supervisor_slot", "original_id", "alternative_id", "action", "notes",
    ])


def prepare_automated(raw: pd.DataFrame) -> pd.DataFrame:
    _require(raw, ["integer_id", "phd_name", "phd_id", "phd_match_by",
                   "contributor_id", "contributor_confirmed", "contributor_rank"])
    auto = raw.copy().fillna("")
    auto["integer_id"] = auto.integer_id.map(source_id)
    auto["phd_match_by"] = auto.phd_match_by.map(text)
    auto["contributor_confirmed"] = auto.contributor_confirmed.map(flag)
    auto["candidate_phd_ids"] = auto.phd_id.map(author_ids)
    auto["candidate_phd_orcids"] = auto.get("phd_orcid", pd.Series("", index=auto.index)).map(orcids)
    auto["candidate_supervisor_ids"] = auto.contributor_id.map(author_ids)
    auto["candidate_supervisor_orcids"] = auto.get("contributor_orcid", pd.Series("", index=auto.index)).map(orcids)
    auto["phd_confirmed"] = auto.phd_match_by.ne("")
    # Refuse to choose an arbitrary first row when repeated PhD metadata disagree.
    for column in ["candidate_phd_ids", "candidate_phd_orcids", "phd_match_by",
                   "phd_match_score", "max_title_similarity", "exact_match",
                   "near_exact_match", "affiliation_match"]:
        if column in auto and auto.groupby("integer_id")[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Conflicting {column} values within an automated thesis.")
    if auto.duplicated(["integer_id", "contributor_rank"]).any():
        raise ValueError("Duplicate automated (integer_id, contributor_rank) rows.")
    return auto


def union_ids(values) -> frozenset[str]:
    return frozenset().union(*values)


def compare_identities(manual: pd.DataFrame, auto: pd.DataFrame):
    """Return thesis and contributor diagnostics; missing manual IDs stay unscored."""
    grouped = {key: group for key, group in auto.groupby("integer_id", sort=False)}
    records = []
    for ref in manual.to_dict("records"):
        group = grouped.get(ref["integer_id"])
        found = group is not None
        first = group.iloc[0] if found else {}
        confirmed = bool(first.get("phd_confirmed", False))
        candidate = first.get("candidate_phd_ids", EMPTY)
        predicted = candidate if confirmed else EMPTY
        truth = ref["reference_phd_ids"]
        correct = bool(predicted and predicted.issubset(truth)) if truth else False
        if not truth:
            outcome = PHD_OUTCOMES[4]
        elif not found:
            outcome = PHD_OUTCOMES[3]
        elif not confirmed:
            outcome = PHD_OUTCOMES[2]
        else:
            outcome = PHD_OUTCOMES[0] if correct else PHD_OUTCOMES[1]
        predicted_sups = union_ids(group.loc[group.contributor_confirmed, "candidate_supervisor_ids"]) if found else EMPTY
        candidate_sups = union_ids(group.candidate_supervisor_ids) if found else EMPTY
        truth_sups = ref["reference_supervisor_ids"]
        identity_hits = predicted_sups & truth_sups
        link_hits = identity_hits if correct else EMPTY
        row = dict(ref)
        row.update(
            in_automated=found, phd_confirmed=confirmed, phd_correct=correct,
            phd_outcome=outcome, candidate_phd_ids=candidate, predicted_phd_ids=predicted,
            phd_candidate_hit=bool(candidate & truth),
            phd_orcid_comparable=bool(ref["reference_phd_orcids"] and first.get("candidate_phd_orcids", EMPTY)),
            phd_orcid_agrees=bool(ref["reference_phd_orcids"] & first.get("candidate_phd_orcids", EMPTY)),
            predicted_supervisor_ids=predicted_sups, candidate_supervisor_ids=candidate_sups,
            correct_supervisor_ids=identity_hits, missed_supervisor_ids=truth_sups - identity_hits,
            extra_supervisor_ids=predicted_sups - truth_sups,
            correct_link_ids=link_hits, missed_link_ids=truth_sups - link_hits,
            n_reference_ids=len(truth_sups), n_predicted_ids=len(predicted_sups),
            n_identity_hits=len(identity_hits), n_link_hits=len(link_hits),
            n_contributor_rows=len(group) if found else 0,
            n_confirmed_contributor_rows=int(group.contributor_confirmed.sum()) if found else 0,
            exact_supervisor_set=bool(truth_sups and predicted_sups == truth_sups),
            all_reference_supervisors_found=bool(truth_sups and truth_sups <= predicted_sups),
            processing_error=text(first.get("processing_error")),
        )
        for column in ["phd_id", "phd_name", "phd_match_score", "max_title_similarity", "phd_match_by"]:
            row[f"auto_{column}"] = first.get(column, "")
        records.append(row)
    theses = pd.DataFrame(records)
    # many_to_one prevents accidental multiplication by manual annotation rows.
    contributors = auto.merge(manual[[
        "integer_id", "reference_supervisor_ids", "reference_supervisors_complete",
    ]], on="integer_id", how="inner", validate="many_to_one")
    contributors["n_candidate_ids"] = contributors.candidate_supervisor_ids.map(len)
    contributors["matched_reference_ids"] = contributors.apply(
        lambda row: row.candidate_supervisor_ids & row.reference_supervisor_ids, axis=1,
    )
    contributors["extra_candidate_ids"] = contributors.apply(
        lambda row: row.candidate_supervisor_ids - row.reference_supervisor_ids, axis=1,
    )
    def contributor_outcome(row):
        if not row.contributor_confirmed:
            return "Not confirmed"
        if not row.candidate_supervisor_ids:
            return "Confirmed without ID"
        if row.matched_reference_ids:
            return "Reference hit + extra IDs" if row.extra_candidate_ids else "Reference IDs only"
        return "No reference hit" if row.reference_supervisors_complete else "Incomplete reference (unscored)"
    contributors["identity_outcome"] = contributors.apply(contributor_outcome, axis=1)
    contributors["phd_correct"] = contributors.integer_id.map(theses.set_index("integer_id").phd_correct)
    return theses, contributors


def coverage_table(manual, auto):
    coverage = manual[["integer_id", "author_name", "institution", "year"]].merge(
        auto[["integer_id", "phd_name"]].drop_duplicates("integer_id"),
        on="integer_id", how="outer", indicator="coverage", validate="one_to_one",
    )
    aliases = {}
    if "duplicate_phds" in auto:
        for row in auto.drop_duplicates("integer_id").to_dict("records"):
            for alias in sequence(row["duplicate_phds"]):
                if isinstance(alias, dict):
                    key = source_id(alias["integer_id"])
                    if key != row["integer_id"]:
                        aliases[key] = row["integer_id"]
    coverage["deduplicated_into_integer_id"] = coverage.integer_id.map(aliases).fillna("")
    return coverage


def ratio(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def metric_summary(theses: pd.DataFrame) -> pd.Series:
    """Observed ID agreement; undefined ratios are NaN, never perfect scores."""
    phd = theses.loc[theses.reference_phd_ids.map(bool)]
    # Precision needs a resolved list; unknown labels are not negative examples.
    relations = phd.loc[phd.reference_supervisors_complete]
    correct = int(phd.phd_correct.sum())
    confirmed = int(phd.phd_confirmed.sum())
    identity_tp = int(relations.n_identity_hits.sum())
    link_tp = int(relations.n_link_hits.sum())
    predicted = int(relations.n_predicted_ids.sum())
    reference = int(relations.n_reference_ids.sum())
    return pd.Series({
        "n_theses": len(theses), "n_extracted": int(theses.in_automated.sum()),
        "n_phd_labeled": len(phd), "n_phd_confirmed": confirmed,
        "n_phd_correct": correct, "n_phd_wrong": confirmed - correct,
        "phd_precision": ratio(correct, confirmed),
        "phd_recall": ratio(correct, len(phd)),
        "phd_f1": ratio(2 * correct, confirmed + len(phd)),
        "phd_candidate_recall": ratio(int(phd.phd_candidate_hit.sum()), len(phd)),
        "n_relation_theses": len(relations),
        "n_reference_relations": reference, "n_predicted_relations": predicted,
        "n_identity_hits": identity_tp, "n_link_hits": link_tp,
        "supervisor_id_precision": ratio(identity_tp, predicted),
        "supervisor_id_recall": ratio(identity_tp, reference),
        "supervisor_id_f1": ratio(2 * identity_tp, predicted + reference),
        "link_precision": ratio(link_tp, predicted),
        "link_recall": ratio(link_tp, reference),
        "link_f1": ratio(2 * link_tp, predicted + reference),
        "exact_supervisor_set_rate": ratio(int(relations.exact_supervisor_set.sum()), len(relations)),
        "all_supervisors_found_rate": ratio(int(relations.all_reference_supervisors_found.sum()), len(relations)),
    })


def group_metrics(theses: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = {str(key): metric_summary(group) for key, group in theses.groupby(column, dropna=False, observed=True)}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(column).sort_values("n_theses", ascending=False)


def export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Make set-valued diagnostics readable and deterministic in CSV exports."""
    result = frame.copy()
    for column in result:
        result[column] = result[column].map(
            lambda value: "; ".join(sorted(value)) if isinstance(value, (set, frozenset)) else value,
        )
    return result
