"""
clean.py
========

Cleaning pipeline for raw MED-PC data, already processed by
`mteresa.io.medpc.parse_med`.

Data model
----------
Each raw MED-PC file becomes a "session": header metadata (Subject,
Box, MSN, ...), scalar/indexed variables (experiment configuration)
and -- when an identifiable event array exists -- a sequence of
(timestamp, event type) pairs.

We adopt a **long format**: a session does not become a single "wide"
row with dozens of sparse columns; it becomes a dictionary with the
scalar metadata plus two parallel lists (`timestamp`, `event_type`).
Run ``DataFrame.explode(['timestamp', 'event_type', 'event_alias'])``
afterwards to get one row per event, with the session metadata
repeated on each row.

All sessions -- regardless of which MSN (program) produced them -- are
combined into a single DataFrame. This is convenient (one table to
filter, group, explode), but it reopens a correctness risk: a variable
with no mapping in the `.MPC` file (e.g. a raw letter "K") can mean
completely different things in two different programs. If both ended
up in a plain column named "K", their values would silently collide.

To keep a single table *safe*, unmapped variables are namespaced with
the session's MSN name (e.g. ``K__MY_PROGRAM``) instead of being left
as a bare letter. Mapped variables (the vast majority, coming from the
`.MPC` file) keep their clean, human-readable name -- namespacing is
only applied as a fallback, exactly where the collision risk actually
exists.
"""

from __future__ import annotations

import logging
import warnings
from os import walk
from os.path import join, dirname
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

import pandas as pd

from mteresa.io.medpc.parse_med import (
    parse_mpc_file,
    parse_med_output_file,
    parse_med_output_sessions,
)

logger = logging.getLogger(__name__)

# Extensions that are known to never be raw MED-PC session files: program
# files (.MPC, .MAC), and spreadsheet/exports that other tools may have
# left behind in the same data folder (e.g. a previous run's .xlsx output).
# Raw MED-PC data files themselves typically have no extension at all, so
# this is a denylist rather than an allowlist.
_NON_SESSION_EXTENSIONS = {".mpc", ".mac", ".xlsx", ".xls", ".csv"}


# =============================================================================
# 1. Reading a single raw file + its matching .MPC program
# =============================================================================

def process_data(file: str, msn_folder_path: Optional[str] = None) -> Dict:
    """Read a MED-PC data file (ONE session) and its `.MPC` (MSN) program.

    For files that hold several sessions use `read_med_folder`, or
    `parse_med_output_sessions` plus `build_processed_session`; this
    function raises ``ValueError`` for them instead of keeping only one.

    Parameters
    ----------
    file:
        Path to the raw MED-PC data file.
    msn_folder_path:
        Folder to look for the `.MPC` file of the program used in this
        session (the MSN name is already present in the data file's
        header). If `None`, the MSN name itself is used as a relative
        path (with a ".MPC" extension appended).

    Returns
    -------
    dict with the keys:
        - "header": session metadata (Subject, Box, MSN, ...);
        - "mappings": names/structure extracted from the `.MPC` file
          (events, indexed arrays, scalar variable mapping);
        - "raw_data": the raw arrays/scalars (keys A-Z), as floats;
        - "raw_tokens": the original text of every array value (keys A-Z),
          needed to read ``timestamp.event`` codes without float rounding.
    """
    parsed_output = parse_med_output_file(file_path=file)
    return build_processed_session(parsed_output, file, msn_folder_path)


def build_processed_session(
    parsed_output: Dict, file: str, msn_folder_path: Optional[str] = None
) -> Dict:
    """Attach the `.MPC` mappings to ONE parsed session (see `process_data`).

    `parsed_output` is one item of `parse_med_output_sessions(file)`; `file`
    is only used to locate the `.MPC` and for messages.
    """
    # The MED OUTPUT header already carries the MSN used in the session.
    # If it's missing, this almost certainly isn't a raw MED-PC data file
    # at all (e.g. a leftover .xlsx export, or some other file that ended
    # up in the data folder) -- fail clearly here instead of letting a
    # cryptic TypeError surface later from Path(None).
    msn_file = parsed_output["header"].get("MSN")
    if not msn_file:
        raise ValueError(
            f"'{file}' does not look like a MED-PC data file: no 'MSN' "
            "field found in its header."
        )

    # We try to locate the matching .MPC file by combining that name with
    # the folder provided by the user; if none is provided, we just use
    # the MSN name directly as a path (with a .MPC extension).
    if msn_folder_path is not None:
        msn_name = Path(msn_file).stem
        msn_path = Path(join(msn_folder_path, f"{msn_name}.MPC"))
    else:
        msn_path = Path(join(dirname(file), f"{msn_file}.MPC"))

    if not msn_path.is_file():
        # `parse_mpc_file` can handle a missing path and returns a generic
        # dictionary for the MED-PC keys (A-Z) -- but this needs to be
        # logged, not just silently swallowed.
        logger.warning(
            "MSN file (.MPC) not found for '%s' (expected at '%s'). "
            "Check the msn_folder_path parameter -- this session's "
            "variables will keep their raw letters [A-Z], with no "
            "descriptive names.",
            file,
            msn_path,
        )

    msn_variables_dict = parse_mpc_file(msn_path)

    return {
        "header": parsed_output.get("header", {}),
        "mappings": msn_variables_dict,
        "raw_data": {
            k: v for k, v in parsed_output.items() if k not in ("header", "_tokens")
        },
        "raw_tokens": parsed_output.get("_tokens", {}),
    }


# =============================================================================
# 2. Event-array detection
# =============================================================================

# An array is only accepted as an "event array" if at least this fraction
# of the checked values matches a known event code -- this avoids mistaking
# an arbitrary numeric array for events just because one value happened to
# match by coincidence.
_EVENT_KEY_MIN_MATCH_RATIO = 0.8
# Minimum sample size needed to trust the heuristic. Very short arrays are
# too ambiguous (a few values matching by chance is not unlikely).
_EVENT_KEY_MIN_SAMPLE_SIZE = 5
_EVENT_KEY_MAX_SAMPLE_SIZE = 20


# MED-PC prints array values with three decimals, so an event code such as
# ".010" is always three digits long in the output file.
_EVENT_CODE_DIGITS = 3


def _canonical_code(digits: str) -> str:
    """Canonical form of the fractional digits of an event code.

    ``.21`` and ``.210`` are the same number, so trailing zeros are
    ignored; leading zeros are significant (``.010`` != ``.100`` != ``.001``).
    """
    return digits.rstrip("0") or "0"


def guess_event_key(
    raw_data: Dict[str, object],
    known_event_codes: Iterable[str],
    min_sample_size: int = _EVENT_KEY_MIN_SAMPLE_SIZE,
    max_sample_size: int = _EVENT_KEY_MAX_SAMPLE_SIZE,
    min_match_ratio: float = _EVENT_KEY_MIN_MATCH_RATIO,
) -> Optional[str]:
    """Try to automatically identify which array encodes events.

    An event array encodes each entry as ``timestamp.event`` (e.g.
    ``1250.201``, where ``201`` is the event code). The heuristic looks
    at a sample of the first values of each candidate array and computes
    the *fraction* of them whose decimal part matches a known event code
    -- instead of accepting the array as soon as the first value matches,
    which produces false positives with arrays that just happen to have
    a decimal value resembling an event code.

    The array values should be the original **strings** (``raw_tokens``),
    not floats: a float cannot tell ``1250.210`` from ``1250.21``.

    Returns `None` if no array satisfies the criteria (minimum sample
    size + minimum match ratio).
    """
    known_codes = {
        _canonical_code(str(code).strip().lstrip(".")) for code in known_event_codes
    }
    if not known_codes:
        return None

    for key, value in raw_data.items():
        if not isinstance(value, list) or not value:
            continue

        sample = value[:max_sample_size]
        if len(sample) < min_sample_size:
            continue

        matches = 0
        for item in sample:
            item_str = str(item)
            if "." not in item_str:
                continue
            _, decimal_part = item_str.split(".", 1)
            if _canonical_code(decimal_part) in known_codes:
                matches += 1

        if matches / len(sample) >= min_match_ratio:
            return key

    return None


# =============================================================================
# 3. Per-variable-type transformation
# =============================================================================

def _clean_label(text: str) -> str:
    """Strip spaces/problematic symbols so text can be used as a column name."""
    return (
        text.replace(" ", "_")
        .replace("/", "")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _scalar_column_name(
    key: str,
    scalar_mapping: Dict[str, str],
    msn_name: str,
    unmapped_keys: List[str],
) -> str:
    """Resolve the column name for a scalar (or unindexed-array) variable.

    If the `.MPC` file provides a name for this key, use it (cleaned up).
    Otherwise, fall back to the raw letter -- but namespaced with the MSN
    name, so that "K" from one program never collides with an unrelated
    "K" from a different program in the combined DataFrame. The key is
    also recorded in `unmapped_keys` so the caller can log it.
    """
    mapped_name = scalar_mapping.get(key)
    if mapped_name is not None:
        return _clean_label(mapped_name) if isinstance(mapped_name, str) else mapped_name

    unmapped_keys.append(key)
    return f"{key}__{_clean_label(msn_name) if msn_name else 'UNKNOWN_MSN'}"


def split_event_array(values: List, event_aliases: Dict[str, str]) -> Dict[str, List]:
    """Split a ``timestamp.event`` array into three parallel lists.

    These lists are ready for ``DataFrame.explode(...)``: a session with
    N events becomes N rows, each with a timestamp, the event code and
    its human-readable name.

    ``values`` must be the original **strings** (``raw_tokens``), e.g.
    ``"1250.210"``. Converting to float first would lose trailing zeros
    (``1250.21``), turning code ``.210`` into ``21`` and making ``.100``
    and ``.001`` both collapse into ``1``.

    The event code is the fractional part read as three digits
    (``.010`` -> 10, ``.100`` -> 100, ``.001`` -> 1). Aliases are matched
    by numeric value of the fraction, so ``.21`` in the `.MPC` file
    matches ``.210`` in the data file, while ``.010`` and ``.100`` stay
    distinct.

    When a value has no decimal part, ``event_type = 0`` is assumed
    (same convention as the original parser).
    """
    timestamps: List[int] = []
    event_types: List[int] = []
    event_aliases_list: List[str] = []

    alias_by_code = {
        _canonical_code(str(code).strip().lstrip(".")): name
        for code, name in event_aliases.items()
    }

    for raw_value in values:
        ts_part, _, code_digits = str(raw_value).strip().partition(".")
        timestamps.append(int(ts_part))
        if code_digits:
            event_types.append(int(code_digits.ljust(_EVENT_CODE_DIGITS, "0")))
            label = code_digits
        else:
            event_types.append(0)
            label = "0"
        event_aliases_list.append(
            alias_by_code.get(_canonical_code(label), f"UNKNOWN_{label}")
        )

    return {
        "timestamp": timestamps,
        "event_type": event_types,
        "event_alias": event_aliases_list,
    }

def expand_indexed_array(key: str, value: List, array_info: dict) -> Dict:
    """Expand an array whose positions each have an individual meaning
    mapped in the `.MPC` file (e.g. ``A(0) = condition``, ``A(1) = group``).

    Unlike the event array, these positions do **not** represent a time
    series -- they are fixed session parameters/configuration. That's
    why they become scalar fields (one per mapped position), rather than
    lists to be exploded.
    """
    array_title = _clean_label(array_info.get("title") or key)
    indices_info = array_info.get("indices", {})

    expanded: Dict[str, object] = {}
    for idx, item_value in enumerate(value):
        idx_info = indices_info.get(idx)
        if idx_info is None:
            continue  # position with no mapping: we don't know what it means

        description = _clean_label(idx_info.get("description", f"INDEX_{idx}"))
        value_labels = idx_info.get("values", {})
        mapped_value = value_labels.get(item_value, item_value)

        expanded[f"{key}({idx})_{array_title}_{description}"] = mapped_value

    return expanded



# =============================================================================
# 4. Orchestration: one raw session -> one "clean" dictionary
# =============================================================================


def clean(
    processed_data: Dict,
    event_key: Optional[str] = None,
    guess_event: bool = True,
) -> Dict:
    """Replace raw keys (A-Z) with the names extracted from the `.MPC` file.

    Parameters
    ----------
    processed_data:
        Output of `process_data`.
    event_key:
        Letter of the event array, if already known. If `None` and
        `guess_event=True`, it is detected automatically (see
        `guess_event_key`).
    guess_event:
        If `True` and `event_key` was not provided, try to detect the
        event array automatically.


    Returns
    -------
    A "flattened" dictionary with:
        - the header metadata;
        - scalar/indexed variables, already named (unmapped ones are
          namespaced with the MSN name -- see `_scalar_column_name`);
        - (if an event array was identified) the parallel lists
          ``timestamp``, ``event_type`` and ``event_alias``, ready for
          ``DataFrame.explode(...)``.

    Note
    ----
    Not every array in `raw_data` that isn't the event array and isn't
    an indexed array is necessarily a per-event or per-timestamp
    variable. Such "leftover" arrays are kept as-is, as a list value in
    a single cell -- they are *not* assumed to align with the event
    timestamps. If such an array turns out to be meaningless data left
    over from a scripting mistake in the MED-PC program, it will simply
    show up as a namespaced, unmapped column full of raw list values;
    check the log message for `unmapped_keys` to catch this.
    """
    header = processed_data.get("header", {})
    cleaned: Dict[str, object] = dict(header)
    msn_name = str(header.get("MSN", "UNKNOWN_MSN"))

    raw_data = processed_data.get("raw_data", {})
    # Original text of each array value; falls back to the float values if
    # absent (e.g. a hand-built `processed_data`).
    raw_tokens = processed_data.get("raw_tokens", {})
    mappings = processed_data.get("mappings", {})
    event_aliases = mappings.get("events", {})
    arrays_info = mappings.get("arrays", {})
    scalar_mapping = mappings.get("mapping", {})

    if guess_event and event_key is None:
        candidates = {k: raw_tokens.get(k, v) for k, v in raw_data.items()}
        event_key = guess_event_key(candidates, event_aliases.keys())

    unmapped_keys: List[str] = []

    for key, value in raw_data.items():
        # (a) Event array: becomes parallel lists, ready to be exploded later.
        if event_key is not None and key == event_key:
            cleaned.update(
                split_event_array(raw_tokens.get(key, value), event_aliases)
            )
            continue

        # (b) Array whose positions are individually mapped in the .MPC file.
        if isinstance(value, list) and key in arrays_info:
            array_info = arrays_info[key]
            if array_info.get("indices"):
                cleaned.update(expand_indexed_array(key, value, array_info))
            else:
                column_name = _scalar_column_name(key, scalar_mapping, msn_name, unmapped_keys)
                cleaned[column_name] = value
            continue

        # (c) Scalar, or an array with no known mapping in the .MPC file.
        column_name = _scalar_column_name(key, scalar_mapping, msn_name, unmapped_keys)
        cleaned[column_name] = value

    if unmapped_keys:
        # Not a fatal error (the data is preserved, safely namespaced), but
        # important information for anyone reviewing cleaning quality: these
        # columns exist because the .MPC file had no name for that letter.
        logger.info(
            "Subject=%s, MSN=%s: variables with no .MPC mapping, namespaced "
            "as '<letter>__%s': %s",
            cleaned.get("Subject", "?"),
            msn_name,
            msn_name,
            ", ".join(sorted(set(unmapped_keys))),
        )

    return cleaned


# =============================================================================
# 5. Batch reading of an entire folder
# =============================================================================
def _find_irt_col(df, irt_key):
    """Return the name of the column holding the IRT list, or None.

    Candidates are the column named exactly ``<irt_key>`` (variable with no
    name in the `.MPC`) or starting with ``<irt_key>_`` (columns are already
    lowercased) **and** containing at least one list. The second condition
    avoids picking unrelated scalar columns that merely share the letter
    (e.g. the scalar ``c`` of another MSN in a mixed folder).
    """
    if irt_key is None:
        return None

    key = irt_key.lower()
    prefix = f"{key}_"
    is_list = lambda x: isinstance(x, list)
    matching_cols = [
        col for col in df.columns
        if (col == key or col.startswith(prefix)) and df[col].map(is_list).any()
    ]
    if not matching_cols:
        logger.warning(
            "irt_key=%r: no column named '%s' or starting with '%s' holds "
            "lists. Skipping IRT calculation.",
            irt_key, key, prefix,
        )
        return None
    if len(matching_cols) > 1:
        logger.warning(
            "irt_key=%r matched multiple columns %s; using the first one.",
            irt_key, matching_cols,
        )
    return matching_cols[0]


def _explode_irt(df, irt_col):
    """One row per IRT value, with a 1-based ``irt_index``.

    Only sessions whose ``irt_col`` is a list are expanded; every other
    row (e.g. event sessions in a mixed folder) is kept as a single row
    with ``irt_index`` empty. Trailing zeros of the MED-PC array (unused
    positions) are removed, but only from rows that came from an IRT list.
    """
    if 'timestamp' in df.columns:
        has_events = df['timestamp'].notna()   # already exploded: scalar per row
    else:
        has_events = pd.Series(False, index=df.index)

    pairs = []
    n_both = 0
    for value, is_event_row in zip(df[irt_col], has_events):
        if isinstance(value, list):
            if is_event_row:
                n_both += 1                      # events win; keep the list as is
                pairs.append(([None], [value]))
            else:
                pairs.append((list(range(1, len(value) + 1)), value))
        else:
            pairs.append(([None], [value]))

    if n_both:
        logger.warning(
            "%d event row(s) also hold an IRT list in '%s'; events take "
            "precedence and the list was left unexpanded.", n_both, irt_col,
        )

    df = df.copy()
    df['irt_index'] = pd.Series([p[0] for p in pairs], index=df.index, dtype=object)
    df[irt_col] = pd.Series([p[1] for p in pairs], index=df.index, dtype=object)
    df = df.explode(['irt_index', irt_col]).reset_index(drop=True)

    from_irt_list = df['irt_index'].notna()
    return df[~from_irt_list | (df[irt_col] != 0)]


def _styling_final_dataframe(df, drop_zeros, drop_lists, guess_session, irt_key):
    # ========================================================================
    #                         HANDLE DATE FORMAT
    # ========================================================================
    
    df['Start Date'] = pd.to_datetime(df['Start Date'], format='%m/%d/%y', errors='coerce')
    df['End Date'] = pd.to_datetime(df['End Date'], format='%m/%d/%y', errors='coerce')

    df['Start Time'] = pd.to_datetime(df['Start Time'], format='%H:%M:%S', errors='coerce')
    df['End Time'] = pd.to_datetime(df['End Time'], format='%H:%M:%S', errors='coerce')

    df['Duration'] = df['End Time'] - df['Start Time']
    negative_duration = df['Duration'] < pd.Timedelta(0)
    df.loc[negative_duration, 'Duration'] += pd.Timedelta(days=1)

    df['Duration'] = (
        df['Duration'].dt.components['hours'].astype(str).str.zfill(2) + ':' +
        df['Duration'].dt.components['minutes'].astype(str).str.zfill(2) + ':' +
        df['Duration'].dt.components['seconds'].astype(str).str.zfill(2)
    )

    df['Start Time'] = df['Start Time'].dt.time
    df['End Time'] = df['End Time'].dt.time

    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')

    df = df.sort_values(by='start_date')

    # ========================================================================
    #                          GUESS SESSION
    # ========================================================================
    if guess_session:
        df = df.reset_index(drop=True)
        df['session'] = df.groupby('subject').cumcount() + 1

        first_cols_not_timestamp = [
            'session', 'subject', 'start_date', 'end_date', 
            'start_time', 'end_time', 'msn', 'experiment', 
            'group', 'box', 'duration'
        ]

        first_cols_with_timestamp = [
            'session', 'subject', 'timestamp', 'event_type', 'event_alias', 
            'start_date', 'end_date', 'start_time', 'end_time', 'msn', 
            'experiment', 'group', 'box', 'duration'
        ]
        if 'timestamp' in list(df.columns.values):
            main_cols = [col for col in first_cols_with_timestamp if col in df.columns]
        else:  main_cols = [col for col in first_cols_not_timestamp if col in df.columns]

        cols = main_cols + [col for col in df.columns if col not in main_cols]

        df = df[cols]

    # ========================================================================
    #                             EXPLODE
    # ========================================================================
    # Events and IRT are independent: a mixed folder can hold sessions of
    # both kinds, so neither step may disable the other.
    df = df.reset_index(drop=True)
    df['session_id'] = df.index + 1
    
    irt_col = _find_irt_col(df, irt_key)

    if 'timestamp' in df.columns:
        df = df.explode(['timestamp', 'event_type', 'event_alias'])
        df = df.reset_index(drop=True)

    if irt_col is not None:
        df = _explode_irt(df, irt_col)

    # ========================================================================
    #                          DROP COLUMNS
    # ========================================================================
    if drop_zeros:
        df = df.loc[:, (df != 0).any(axis=0)]

    if drop_lists:
        # Retorna True para colunas que NÃO contêm elementos do tipo lista
        has_list = df.map(lambda x: isinstance(x, list)).all()
        df = df.loc[:, ~has_list]

    # ========================================================================
    #                            SORTING
    # ========================================================================
    try:
        df['subject_number'] = df['subject'].str.extract(r'(\d+)').astype(int)
        df = df.sort_values(by=['subject_number', 'start_date'])
        df = df.drop('subject_number', axis=1)
    except Exception as erro:
        print(f"Error converting subject to number: {erro}")
    df = df.reset_index(drop=True)
    
    # ========================================================================
    #                     IRT INTO TIMESTAMP COLUMN
    # ========================================================================
    if irt_col is not None:
        df = df.sort_values(['session_id', 'irt_index'], kind='stable')
        is_irt = df['irt_index'].notna()
        df['irts'] = pd.to_numeric(df[irt_col], errors='coerce')  
        ts = df.loc[is_irt].groupby('session_id')['irts'].cumsum()
        df.loc[is_irt, 'timestamp'] = ts
        df['event_type'] = '---'
        df['event_alias'] = 'operant generic response'


    return df


def _handle_failure(
    exc: Exception,
    file_path: str,
    session_index: Optional[int],
    n_sessions: int,
    on_error: str,
    issues: List[Dict],
) -> None:
    """Apply the `on_error` policy to one failed file (or one session of it)."""
    where = (
        file_path
        if n_sessions <= 1
        else f"{file_path} [session {session_index} of {n_sessions}]"
    )
    if on_error == "raise":
        raise RuntimeError(f"Failed to process '{where}': {exc}") from exc

    # "warn" and "skip" both continue; they differ in noise.
    log = logger.warning if on_error == "warn" else logger.debug
    log("Failed to process '%s': %s", where, exc)
    issues.append({"file": file_path, "session": session_index, "error": str(exc)})


def read_med_folder(
    folder_path: str,
    msn_folder_path: Optional[str] = None,
    event_key: Optional[str] = None,
    guess_event: bool = True,
    irt_key: Optional[str] = None,
    drop_zeros: bool = True,
    drop_lists: bool = False,
    guess_session: bool = False,
    on_error: Literal["warn", "raise", "skip"] = "warn",
) -> pd.DataFrame:
    """Read all data files in a folder into a single long-format DataFrame.

    Every file in `folder_path` is treated as a candidate data file, except
    those with an extension in `_NON_SESSION_EXTENSIONS` (`.MPC`/`.MAC`
    program files, and spreadsheet exports like `.xlsx`/`.csv` that may
    already sit in the same folder from a previous run). Files that pass
    this filter but still turn out not to be MED-PC sessions (e.g. no
    `MSN` in the header) are handled according to `on_error`.

    A data file may hold several sessions (MED-PC appends them when the
    same file name is reused): each ``Start Date:`` block is read as its own
    session, and a failure in one block does not discard the others.

    Variables with no mapping in their `.MPC` file are namespaced with
    the MSN name (see `clean`), so combining everything into a single
    DataFrame does not risk mixing unrelated data under the same column.

    Parameters
    ----------
    folder_path:
        Folder to walk recursively looking for data files.
    msn_folder_path:
        Folder to look for `.MPC` files. See `process_data`.
    event_key, guess_event:
        Passed through to `clean` for each session.
    drop_zeros:
        If True, drops columns containing all values equal 0.
    drop_lists:
        If True, drops columns containing lists.
    guess_session:
        If True, creates a column called `session` by counting the sessions
        of each subject. The count needs to be continuous for each subject.
    on_error:
        What to do when a file fails to process:

        - ``"warn"`` (default): skip the file and emit one summary warning.
        - ``"raise"``: stop at the first failure (``RuntimeError`` naming
          the file, chained to the original exception).
        - ``"skip"``: skip the file silently.

        With ``"warn"`` and ``"skip"``, the failures are kept in
        ``df.attrs["issues"]`` (see Returns).

    Returns
    -------
    sessions:
        One row per session (**not** yet exploded), across all MSNs.
        To get one row per event, run::

            sessions.explode(["timestamp", "event_type", "event_alias"])

        Use the "MSN" column to filter/group sessions by program when
        needed. Files that failed are listed in ``sessions.attrs["issues"]``
        as a list of ``{"file": ..., "session": ..., "error": ...}`` dicts
        (``session`` is the 1-based block in that file, or ``None`` when
        the whole file could not be read); use
        ``pd.DataFrame(sessions.attrs["issues"])`` for a table. (A plain
        list, not a DataFrame: pandas compares ``attrs`` with ``==`` when
        printing or concatenating, which breaks on DataFrame values.)
    """
    if on_error not in ("warn", "raise", "skip"):
        raise ValueError(
            f"on_error must be 'warn', 'raise' or 'skip', got {on_error!r}"
        )

    rows: List[Dict] = []
    issues: List[Dict[str, str]] = []

    for root, _dirs, files in walk(folder_path):
        for file_name in files:
            if Path(file_name).suffix.lower() in _NON_SESSION_EXTENSIONS:
                continue  # program files or leftover exports, not session data

            file_path = join(root, file_name)

            # A data file can hold several sessions (one per "Start Date:").
            try:
                parsed_sessions = parse_med_output_sessions(file_path)
            except Exception as exc:
                _handle_failure(exc, file_path, None, 0, on_error, issues)
                continue

            n_sessions = len(parsed_sessions)
            for index, parsed in enumerate(parsed_sessions, start=1):
                try:
                    session = clean(
                        build_processed_session(parsed, file_path, msn_folder_path),
                        event_key=event_key,
                        guess_event=guess_event,
                    )
                except Exception as exc:
                    _handle_failure(exc, file_path, index, n_sessions, on_error, issues)
                    continue

                rows.append(session)

    if not rows:
        first = "; ".join(f"{i['file']}: {i['error']}" for i in issues[:3])
        raise ValueError(
            f"No valid MED-PC sessions found in '{folder_path}'. "
            f"{len(issues)} file(s) failed" + (f" (first: {first})" if first else "")
        )

    sessions_df = _styling_final_dataframe(
        pd.DataFrame(rows),
        drop_zeros,
        drop_lists,
        guess_session,
        irt_key,
    )

    sessions_df.attrs["issues"] = issues  # list of dicts; assign AFTER styling

    if issues and on_error == "warn":
        warnings.warn(
            f"{len(issues)} file(s) failed to process; see df.attrs['issues'].",
            stacklevel=2,
        )

    return sessions_df