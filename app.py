import csv
from hashlib import sha256
import io
import itertools
import warnings
from zipfile import BadZipFile

import numpy as np
import streamlit as st
from sympy import Matrix, latex

from src.generate_data import (
    generate_first_order_processes,
    generate_infinite_order_processes,
    generate_k_order_processes,
    generate_zero_order_processes,
    infinite_order_model,
)
from src.minimization import EXAMPLE_4_2, _rational, minimize_representation
from src.rank_one import classify_minimal
from src.stage4_pipeline import (
    classify_hidden_chain, classify_higher_block, classify_symbol_matrices,
)
from src.statistical import estimate_markov_order


def reset_matrix(key, values):
    st.session_state[f"values:{key}"] = [[str(x) for x in row] for row in values]
    st.session_state.pop(f"shape:{key}", None)


def matrix_editor(label, rows, cols, key, default=None, row_labels=None):
    """Keep a fixed editor baseline and save string values across shape changes."""
    st.markdown(f"**{label}** · {rows} × {cols}")
    shape = (rows, cols)
    previous_revision = st.session_state.get(f"revision:{key}", 0)
    if (st.session_state.get(f"shape:{key}") != shape
            or f"grid:{key}:{previous_revision}" not in st.session_state):
        previous = st.session_state.get(f"values:{key}", default or [])
        values = [[str(previous[i][j]) if i < len(previous) and j < len(previous[i])
                   else "0" for j in range(cols)] for i in range(rows)]
        st.session_state[f"base:{key}"] = values
        st.session_state[f"shape:{key}"] = shape
        st.session_state[f"revision:{key}"] = previous_revision + 1
    revision = st.session_state[f"revision:{key}"]
    base = st.session_state[f"base:{key}"]
    if rows * cols > 100:
        st.caption("Large matrix: import a headerless CSV of rational entries, or edit below.")
    with st.expander(f"Import {label} from CSV"):
        upload = st.file_uploader("Headerless CSV; entries may include fractions", type="csv",
                                  key=f"csv:{key}:{revision}")
        if st.button("Use CSV", key=f"import:{key}:{revision}"):
            try:
                if upload is None:
                    raise ValueError("Choose a CSV file first.")
                values = list(csv.reader(io.StringIO(upload.getvalue().decode("utf-8-sig"))))
                if len(values) != rows or any(len(row) != cols for row in values):
                    raise ValueError(f"CSV must have {rows} rows and {cols} columns.")
                for row in values:
                    for value in row:
                        _rational(value.strip())
                reset_matrix(key, values)
                st.rerun()
            except (ValueError, TypeError, ZeroDivisionError, UnicodeError) as exc:
                st.error(str(exc))
    labels = row_labels if row_labels is not None else [str(i) for i in range(rows)]
    data = {"Row": labels}
    data.update({str(j): [base[i][j] for i in range(rows)] for j in range(cols)})
    edited = st.data_editor(
        data, key=f"grid:{key}:{revision}", hide_index=True,
        disabled=["Row"], num_rows="fixed", width="stretch",
        column_config={str(j): st.column_config.TextColumn(str(j), required=True)
                       for j in range(cols)},
    )
    values = [[edited[str(j)][i] for j in range(cols)] for i in range(rows)]
    st.session_state[f"values:{key}"] = values
    return values


def rational_entries(rows):
    """Validate at submission time, preserving fractions in editor state."""
    try:
        return [[_rational(value.strip()) for value in row] for row in rows]
    except (ValueError, TypeError, AttributeError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("Every matrix cell must contain an integer, fraction, or decimal.") from exc


def show_matrix(label, values):
    st.latex(label + " = " + latex(Matrix(values)))


def exact_preset():
    preset = st.session_state.exact_preset
    st.session_state.exact_n = 2
    st.session_state.exact_m = 2
    st.session_state.exact_pi_mode = "Automatic"
    if preset == "Even Process":
        st.session_state.exact_mode = "Symbol matrices"
        # Existing model evaluated at an exactly representable p=1/2.
        matrices, _ = infinite_order_model("even", 0.5)
        for symbol, matrix in enumerate(matrices):
            reset_matrix(f"exact_M{symbol}", matrix.tolist())
    else:
        st.session_state.exact_mode = "Hidden Markov chain"
        reset_matrix("exact_T", [[1, 0], [0, 1]] if preset == "Multiple stationary distributions"
                     else [["4/5", "1/5"], ["3/10", "7/10"]])
        for i in range(2):
            st.session_state[f"mapping:{i}:2"] = i
    st.session_state.pop("exact_result", None)
    st.session_state.pop("exact_chosen_pi", None)


def run_classifier(request, pi=None):
    mode, args = request
    args = dict(args)
    for name in ("T", "Q"):
        if name in args:
            args[name] = rational_entries(args[name])
    if "matrices" in args:
        args["matrices"] = {s: rational_entries(rows) for s, rows in args["matrices"].items()}
    if mode == "General Linear Representation":
        reduced = minimize_representation(
            rational_entries([args["alpha"]])[0], args["matrices"],
            [row[0] for row in rational_entries(args["beta"])])
        result = classify_minimal(reduced["matrices"])
        result["original_dimension"] = reduced["original_dimension"]
        return result
    if pi is None:
        pi = args.pop("pi")
        if pi is not None:
            pi = rational_entries([pi])[0]
    else:
        args.pop("pi")
    classifier = {"Hidden Markov chain": classify_hidden_chain,
                  "Symbol matrices": classify_symbol_matrices,
                  "Higher-block process": classify_higher_block}[mode]
    return classifier(**args, pi=pi)


def classification_output(result):
    order = result["order"]
    if "original_dimension" in result:
        st.metric("Original representation dimension", result["original_dimension"])
    columns = st.columns(3)
    columns[0].metric("Exact Markov order", "∞" if order is None else str(order))
    columns[1].metric("Minimal dimension d", result["minimal_dimension"])
    columns[2].metric("Search bound d(d−1)/2", result["search_bound"])
    if order is None:
        st.info("Infinite order: the finite theoretical search bound was exhausted.")
    elif order == 0:
        st.success("Order 0: the stationary observed process is i.i.d.")
    else:
        st.success(f"The observed process has exact Markov order {order}.")
    if result["diagnostics"]:
        st.dataframe([
            {"Candidate order": item["order"], "Products with rank > 1": item["surviving_products"]}
            for item in result["diagnostics"]
        ], hide_index=True, width="stretch")
        st.caption("The first candidate order with zero surviving products succeeds.")


def exact_tab():
    st.session_state.setdefault("exact_n", 2)
    st.session_state.setdefault("exact_m", 2)
    st.subheader("Classify a known stationary model")
    if st.session_state.get("exact_mode") == "General Linear Representation":
        st.caption("Enter exact rational entries, including negative numbers, fractions, and decimals.")
    else:
        st.caption("Enter exact model probabilities. Use 1/2, 2/3, or decimals; row sums must equal 1 exactly.")
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    left.selectbox("Example", ["First-order chain", "Multiple stationary distributions", "Even Process"],
                   key="exact_preset")
    right.button("Load example", on_click=exact_preset,
                 help="Replace the current inputs with the selected example, then click Classify exact order.")
    mode = st.selectbox("Input mode", ["Hidden Markov chain", "Symbol matrices", "Higher-block process",
                                       "General Linear Representation"],
                        key="exact_mode")
    columns = st.columns(2)
    if mode == "Higher-block process":
        k = int(columns[0].number_input("History length k", 1, 12, 2, key="exact_k"))
        m = int(columns[1].number_input("Alphabet size m", 2, 20, 2, key="block_m"))
        n = m ** k
        if n > 64:
            st.warning("The history table grows exponentially; exact computation may be expensive.")
        if n > 4096:
            st.error("This table exceeds the interactive limit of 4096 histories. Reduce k or m.")
            return
        labels = [" ".join(map(str, history)) if m > 10 else "".join(map(str, history))
                  for history in itertools.product(range(m), repeat=k)]
        q = matrix_editor("Q", n, m, "exact_Q", [[f"1/{m}"] * m] * n, labels)
        st.caption("History length k describes the input table; the actual Markov order may be smaller.")
        args = dict(Q=q, order=k, m=m)
    else:
        n = int(columns[0].number_input("Hidden states n" if mode == "Hidden Markov chain" else
                                       "Representation dimension n", 1, 100, value=None, key="exact_n"))
        m = int(columns[1].number_input("Observable symbols m", 1, 20, value=None, key="exact_m"))
        if n > 6:
            st.warning("Exact minimization and classification may be expensive for larger matrices.")
        if mode == "Hidden Markov chain":
            transition = matrix_editor("T", n, n, "exact_T",
                                       [["4/5", "1/5"], ["3/10", "7/10"]] if n == 2
                                       else [[f"1/{n}"] * n] * n)
            st.markdown("**State-to-symbol mapping f**")
            mapping = []
            columns = st.columns(min(n, 6))
            for i in range(n):
                mapping_key = f"mapping:{i}:{m}"
                st.session_state.setdefault(mapping_key, i % m)
                mapping.append(columns[i % len(columns)].selectbox(
                    f"State {i}", range(m), index=None, key=mapping_key))
            args = dict(T=transition, mapping=mapping)
        elif mode == "General Linear Representation":
            st.info("This mode assumes that the supplied linear representation defines a stationary stochastic process. "
                    "The application does not automatically verify this assumption. The rank-one theorem is only "
                    "guaranteed to classify Markov order under its mathematical hypotheses.")
            alpha = matrix_editor("α", 1, n, "general_alpha", [[f"1/{n}"] * n])[0]
            matrices = {str(symbol): matrix_editor(
                f"M{symbol}", n, n, f"general_M{symbol}", [["0"] * n] * n)
                for symbol in range(m)}
            beta = matrix_editor("β", n, 1, "general_beta", [[1]] * n)
            args = dict(alpha=alpha, matrices=matrices, beta=beta)
        else:
            matrices = {}
            for symbol in range(m):
                default = [[f"1/{n*m}"] * n] * n
                matrices[str(symbol)] = matrix_editor(f"M{symbol}", n, n, f"exact_M{symbol}", default)
            args = dict(matrices=matrices)
    if mode != "General Linear Representation":
        pi_mode = st.radio("Stationary distribution π", ["Automatic", "Manual"], horizontal=True,
                           key="exact_pi_mode")
        args["pi"] = (matrix_editor("π", 1, n, f"exact_pi:{mode}", [[f"1/{n}"] * n])[0]
                      if pi_mode == "Manual" else None)
    request = (mode, args)
    if st.button("Classify exact order", type="primary", key="classify"):
        st.session_state.pop("exact_result", None)
        st.session_state.pop("exact_chosen_pi", None)
        try:
            with st.spinner("Classifying with exact arithmetic…"):
                st.session_state.exact_result = run_classifier(request)
                st.session_state.exact_request = request
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            st.error(str(exc))
    if "exact_result" not in st.session_state:
        return
    if st.session_state.exact_request != request:
        st.caption("Inputs changed. Classify again to update the result.")
        return
    result = st.session_state.exact_result
    if "stationary_family" in result:
        st.info("Stationarity is not unique. Choose real parameter values satisfying the constraints.")
        show_matrix(r"\pi", result["stationary_family"])
        st.latex(r"\quad ".join(latex(constraint) for constraint in result["constraints"]))
        values = {parameter: st.text_input(str(parameter), key=f"parameter:{parameter}")
                  for parameter in result["free_parameters"]}
        if st.button("Use parameters and classify", key="choose_parameters"):
            try:
                chosen = {p: _rational(value.strip()) for p, value in values.items()}
                pi = list(result["stationary_family"].subs(chosen))
                with st.spinner("Classifying the selected stationary model…"):
                    result = run_classifier(request, pi=pi)
                st.session_state.exact_result = result
                st.session_state.exact_chosen_pi = pi
            except (ValueError, TypeError, ZeroDivisionError) as exc:
                st.error(f"Invalid stationary parameters: {exc}")
    if "order" in result:
        if st.session_state.get("exact_chosen_pi") is not None and args.get("pi") is None:
            show_matrix(r"\pi_{\mathrm{selected}}", [st.session_state.exact_chosen_pi])
        classification_output(result)


def minimization_preset():
    st.session_state.min_n = 3
    st.session_state.min_m = 2
    reset_matrix("min_alpha", [EXAMPLE_4_2["alpha"]])
    reset_matrix("min_beta", [[x] for x in EXAMPLE_4_2["beta"]])
    for i, matrix in enumerate(EXAMPLE_4_2["matrices"].values()):
        reset_matrix(f"min_M{i}", matrix)
    st.session_state.pop("min_result", None)


def minimization_tab():
    st.session_state.setdefault("min_n", 3)
    st.session_state.setdefault("min_m", 2)
    st.subheader("Minimize a linear representation")
    st.caption("General rational entries are allowed, including negative values.")
    st.button("Load Example 4.2", on_click=minimization_preset)
    columns = st.columns(2)
    n = int(columns[0].number_input("Original dimension n", 1, 100, value=None, key="min_n"))
    m = int(columns[1].number_input("Number of symbols m", 1, 20, value=None, key="min_m"))
    alpha = matrix_editor("α", 1, n, "min_alpha", [EXAMPLE_4_2["alpha"]])[0]
    matrices = {str(i): matrix_editor(f"M{i}", n, n, f"min_M{i}",
                                    list(EXAMPLE_4_2["matrices"].values())[i] if i < 2 else None)
                for i in range(m)}
    beta = matrix_editor("β", n, 1, "min_beta", [[1]] * n)
    request = (alpha, matrices, beta)
    if st.button("Minimize representation", type="primary", key="minimize"):
        st.session_state.pop("min_result", None)
        try:
            with st.spinner("Minimizing with exact arithmetic…"):
                result = minimize_representation(
                    rational_entries([alpha])[0],
                    {s: rational_entries(rows) for s, rows in matrices.items()},
                    [row[0] for row in rational_entries(beta)])
            st.session_state.min_result = result
            st.session_state.min_request = request
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            st.error(str(exc))
    if "min_result" not in st.session_state:
        return
    if st.session_state.min_request != request:
        st.caption("Inputs changed. Minimize again to update the result.")
        return
    result = st.session_state.min_result
    columns = st.columns(4)
    for column, name in zip(columns, ["original", "reachable", "observable", "minimal"]):
        column.metric(name.capitalize() + " dimension", result[name + "_dimension"])
    show_matrix(r"\alpha'", result["alpha"])
    for symbol, matrix in result["matrices"].items():
        show_matrix(f"M'_{{{symbol}}}", matrix)
    show_matrix(r"\beta'", result["beta"])
    st.caption("Minimal representations are unique up to an invertible change of basis. "
               "Matrices can differ from a reference paper while defining the same word probabilities.")


def generate_dataset(family, count, length, m, order, seed):
    """Adapt existing generator outputs to their existing NPZ field names."""
    common = dict(num_processes=count, sequence_length=length, seed=seed)
    metadata = dict(seed=np.array(seed), num_symbols=np.array(m))
    if family == "i.i.d.":
        sequences, probabilities = generate_zero_order_processes(num_symbols=m, **common)
        return dict(sequences=sequences, probabilities=probabilities,
                    orders=np.zeros(count, dtype=int), **metadata)
    if family == "First-order Markov":
        sequences, initial, transitions, dependence = generate_first_order_processes(num_symbols=m, **common)
        return dict(sequences=sequences, initial_probabilities=initial, transitions=transitions,
                    dependence=dependence, orders=np.ones(count, dtype=int), **metadata)
    if family == "Higher-order Markov":
        sequences, probabilities, orders = generate_k_order_processes(order, num_symbols=m, **common)
        return dict(sequences=sequences, conditional_probabilities=probabilities, orders=orders, **metadata)
    # Backend generates both families. Keep the selected half, including metadata.
    data = generate_infinite_order_processes(num_processes=2 * count, sequence_length=length, seed=seed)
    selected = data["families"] == ("even" if family == "Even Process" else "sns")
    return {name: value[selected] if value.ndim else value for name, value in data.items()}


def generation_tab():
    st.subheader("Generate observed sequences")
    family = st.selectbox("Process family", ["i.i.d.", "First-order Markov", "Higher-order Markov",
                                              "Even Process", "SNS"], key="gen_family")
    columns = st.columns(3)
    count = int(columns[0].number_input("Number of processes", 1, 100, 2, key="gen_count"))
    length = int(columns[1].number_input("Sequence length", 1, 1_000_000, 1000, key="gen_length"))
    seed = int(columns[2].number_input("Random seed", 0, 2**32 - 1, 42, key="gen_seed"))
    infinite = family in ("Even Process", "SNS")
    m = 2 if infinite else int(st.number_input("Alphabet size", 2, 20, 2, key="gen_m"))
    order = int(st.number_input("Generating history length", 2, 12, 2, key="gen_order")) if family == "Higher-order Markov" else 0
    if infinite:
        st.caption("Binary stationary models; the backend samples p in [0.2, 0.8). "
                   "Both families are generated internally; only the selected family is retained.")
    elif family == "First-order Markov":
        st.caption("The backend splits the batch into weak and strong dependence groups; "
                   "Dirichlet parameters and row-distance thresholds are fixed by the generator.")
    else:
        st.caption("Probability rows are sampled from Dirichlet(1, …, 1).")
    too_large = (family == "Higher-order Markov" and m**order * m * count > 1_000_000)
    if too_large:
        st.warning("This conditional table is too large for the interactive app. Reduce history length, alphabet, or count.")
    if st.button("Generate sequences", type="primary", disabled=too_large, key="generate"):
        try:
            with st.spinner("Generating…"):
                data = generate_dataset(family, count, length, m, order, seed)
                st.session_state.generated = data
                st.session_state.generated_family = family
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **data)
                st.session_state.generated_npz = buffer.getvalue()
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
    if "generated" not in st.session_state:
        return
    data = st.session_state.generated
    st.markdown(f"**Generated batch: {st.session_state.generated_family}**")
    index = st.selectbox("Inspect process", range(len(data["sequences"])), format_func=lambda i: f"Process {i + 1}")
    sequence = data["sequences"][index]
    order_value = data["orders"][index]
    st.write(f"Generating order: {'∞' if np.isinf(order_value) else int(order_value)} · Length: {len(sequence)}")
    st.caption("Higher-order tables have their generating order almost surely; this is not an exact certification of stored floats.")
    st.text(" ".join(map(str, sequence[:100])))
    symbols, counts = np.unique(sequence, return_counts=True)
    st.dataframe([{"Symbol": int(s), "Empirical frequency": float(c / len(sequence))}
                  for s, c in zip(symbols, counts)], hide_index=True)
    with st.expander("Generating model parameters"):
        for field, label in [("probabilities", "Symbol probabilities"), ("transitions", "T"),
                             ("conditional_probabilities", "Q"), ("initial_probabilities", "Initial probabilities"),
                             ("stationary_distributions", "π")]:
            if field in data:
                st.write(label)
                table = np.atleast_2d(data[field][index])
                if field == "conditional_probabilities":
                    alphabet_size = table.shape[1]
                    histories = itertools.product(range(alphabet_size), repeat=int(order_value))
                    separator = " " if alphabet_size > 10 else ""
                    labeled = {"History": [separator.join(map(str, history)) for history in histories]}
                    labeled.update({f"Next symbol {symbol}": table[:, symbol]
                                    for symbol in range(alphabet_size)})
                    st.dataframe(labeled, hide_index=True, width="stretch")
                else:
                    st.dataframe(table, width="stretch")
        if "symbol_matrices" in data:
            st.write(f"p = {data['p'][index]}")
            for symbol, matrix in enumerate(data["symbol_matrices"][index]):
                st.write(f"M{symbol}")
                st.dataframe(matrix)
        if "dependence" in data:
            st.write("Dependence group:", data["dependence"][index])
    st.caption("Generator parameters are floating point. Downloads preserve them as model metadata; "
               "they are not silently converted to exact classifier inputs.")
    st.download_button("Download batch (.npz)", st.session_state.generated_npz,
                       file_name="generated_processes.npz", mime="application/octet-stream")


def load_sequence_file(content, filename):
    """Read existing NPZ sequences or headerless integer CSV, without rounding."""
    alphabet = None
    if filename.lower().endswith(".npz"):
        with np.load(io.BytesIO(content), allow_pickle=False) as data:
            if "sequences" not in data:
                raise ValueError("NPZ must contain a 'sequences' array.")
            sequences = data["sequences"]
            if "num_symbols" in data:
                value = data["num_symbols"]
                if value.ndim != 0 or value.dtype.kind not in "iu" or int(value) < 2:
                    raise ValueError("num_symbols must be a scalar integer >= 2.")
                alphabet = int(value)
    else:
        rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
        try:
            sequences = np.array([[int(cell.strip()) for cell in row] for row in rows], dtype=np.int64)
        except (ValueError, OverflowError) as exc:
            raise ValueError("CSV must be a rectangular table of integer symbols without a header.") from exc
        # A single column is one sequence; otherwise each row is a sequence.
        if sequences.ndim == 2 and sequences.shape[1] == 1:
            sequences = sequences.T
    if sequences.ndim == 1:
        sequences = sequences[np.newaxis, :]
    if (sequences.ndim != 2 or not sequences.size
            or sequences.dtype.kind not in "iu" or np.any(sequences < 0)):
        raise ValueError("Sequences must be a nonempty vector or table of nonnegative integer symbols.")
    return sequences, alphabet


def statistical_tab():
    st.subheader("Sequence-only Markov-order estimation")
    st.write("Estimate Markov order from observed symbols using BIC, without knowing the generating model.")
    st.caption("This selects a finite-order approximation. It does not certify finite or infinite memory.")
    sources = ["Generated sequence", "Manual sequence", "CSV upload"]
    if "generated" in st.session_state:
        sources.append("Previous batch")
    source = st.radio("Sequence source", sources, horizontal=True, key="stat_source")
    if source == "Generated sequence":
        family = st.selectbox("Process family", ["i.i.d.", "First-order Markov", "Higher-order Markov",
                                                  "Even Process", "SNS"], key="stat_gen_family")
        columns = st.columns(2)
        length = int(columns[0].number_input("Sequence length", 1, 1_000_000, 1000, key="stat_gen_length"))
        seed = int(columns[1].number_input("Random seed", 0, 2**32 - 1, 42, key="stat_gen_seed"))
        alphabet = (2 if family in ("Even Process", "SNS") else
                    int(st.number_input("Generating alphabet size m", 2, 20, 2, key="stat_gen_m")))
        order = (int(st.number_input("True generating order", 2, 12, 2, key="stat_gen_order"))
                 if family == "Higher-order Markov" else 0)
        generation_request = (family, length, seed, alphabet, order)
        too_large = family == "Higher-order Markov" and alphabet**order * alphabet > 1_000_000
        if too_large:
            st.warning("This conditional table is too large. Reduce the order or alphabet size.")
        if st.button("Generate sequence", key="stat_generate", disabled=too_large):
            st.session_state.pop("stat_result", None)
            try:
                with st.spinner("Generating sequence…"):
                    st.session_state.stat_generated = generate_dataset(family, 1, length, alphabet, order, seed)
                st.session_state.stat_generation_request = generation_request
            except (ValueError, TypeError) as exc:
                st.error(str(exc))
        if ("stat_generated" not in st.session_state
                or st.session_state.stat_generation_request != generation_request):
            st.info("Choose generator settings and click Generate sequence.")
            return
        data = st.session_state.stat_generated
        sequences, index = data["sequences"], 0
        true_order = data["orders"][0]
        st.write(f"Generating order: {'∞' if np.isinf(true_order) else int(true_order)}")
        st.text(" ".join(map(str, sequences[0][:100])))
    elif source == "Manual sequence":
        raw = st.text_area("Sequence of integer symbols", key="stat_manual",
                           placeholder="0 1 1 0 2 1 0 2", help="Separate symbols with spaces, commas, or newlines.")
        if not raw.strip():
            st.info("Paste a sequence to estimate its Markov order.")
            return
        try:
            sequence = np.array([int(token) for token in raw.replace(",", " ").split()], dtype=np.int64)
            if not sequence.size or np.any(sequence < 0):
                raise ValueError
        except (ValueError, OverflowError):
            st.error("Use nonnegative integer symbols separated by spaces, commas, or newlines.")
            return
        sequences, index, alphabet = sequence[np.newaxis, :], 0, None
    elif source == "Previous batch":
        data = st.session_state.generated
        sequences = data["sequences"]
        family_name = st.session_state.generated_family
        index = st.selectbox("Previously generated sequence", range(len(sequences)),
                             format_func=lambda i: f"{family_name} · Process {i + 1}",
                             key="stat_sequence")
        st.caption("Sequences from the Process Generation tab.")
        # Infinite-process batches use the backend's fixed binary alphabet.
        alphabet = int(data.get("num_symbols", 2))
    else:
        upload = st.file_uploader("Sequence file (.npz or .csv)", type=["npz", "csv"], key="stat_upload")
        st.caption("NPZ: 'sequences' array and optional 'num_symbols'. CSV: no header; "
                   "one sequence per row, or one sequence in a single column. Symbols must be integers 0…m−1.")
        if upload is None:
            return
        try:
            sequences, alphabet = load_sequence_file(upload.getvalue(), upload.name)
        except (ValueError, TypeError, OSError, UnicodeError, OverflowError, BadZipFile) as exc:
            st.error(str(exc))
            return
        index = st.selectbox("Uploaded sequence", range(len(sequences)),
                             format_func=lambda i: f"Sequence {i + 1}", key="stat_uploaded_sequence")
    sequence = sequences[index]
    sequence_id = sha256(sequence.tobytes()).hexdigest()
    if alphabet is None:
        context = (sequence_id, sequence.dtype.str)
        if st.session_state.get("stat_alphabet_context") != context:
            st.session_state.stat_m = max(2, int(sequence.max()) + 1)
            st.session_state.stat_alphabet_context = context
        alphabet = int(st.number_input("Alphabet size m", min_value=2, value=None, key="stat_m"))
        st.caption("Include the full alphabet, even symbols absent from this sample.")
    else:
        st.write(f"Alphabet size m: {alphabet}")
    st.write(f"Sequence length: {len(sequence)}")
    max_order = int(st.number_input("Maximum candidate order", min_value=0, value=6, key="stat_max_order"))
    request = (source, sequence_id, sequence.dtype.str, alphabet, max_order)
    if st.button("Estimate Markov order (BIC)", type="primary", key="estimate_bic"):
        st.session_state.pop("stat_result", None)
        try:
            with st.spinner("Comparing candidate orders…"), warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", RuntimeWarning)
                result = estimate_markov_order(sequence, max_order=max_order, m=alphabet)
            st.session_state.stat_result = result
            st.session_state.stat_request = request
            st.session_state.stat_warnings = [str(item.message) for item in captured]
        except (ValueError, TypeError, OverflowError) as exc:
            st.error(str(exc))
    if "stat_result" not in st.session_state:
        return
    if st.session_state.stat_request != request:
        st.caption("Inputs changed. Estimate again to update the result.")
        return
    result = st.session_state.stat_result
    for message in st.session_state.stat_warnings:
        st.warning(message)
    selected = result["estimated_order"]
    columns = st.columns(2)
    columns[0].metric("BIC-selected finite-order estimate", selected)
    columns[1].metric("Effective sample size n_eff", result["n_eff"])
    if selected == max_order:
        st.warning("The selected order is at the search boundary. Longer memory remains unresolved.")
    scores = [{"Candidate order": row["order"], "Maximized log-likelihood": row["log_likelihood"],
               "Parameters": row["parameters"], "BIC": row["bic"]} for row in result["scores"]]
    st.dataframe(scores, hide_index=True, width="stretch")
    chart = [{**row, "Selected minimum": row["Candidate order"] == selected} for row in scores]
    st.vega_lite_chart(chart, {
        "title": "BIC by candidate order (red point = selected minimum)",
        "encoding": {"x": {"field": "Candidate order", "type": "ordinal"},
                     "y": {"field": "BIC", "type": "quantitative", "scale": {"zero": False}},
                     "tooltip": [{"field": "Candidate order"}, {"field": "BIC"}]},
        "layer": [{"mark": {"type": "line", "point": True}},
                  {"transform": [{"filter": {"field": "Selected minimum", "equal": True}}],
                   "mark": {"type": "point", "filled": True, "color": "#e45756", "size": 150}}],
    }, width="stretch")
    with st.expander("Estimated conditional transition probabilities"):
        probabilities = [
            {"History": " ".join(map(str, history)) if history else "∅ (order 0)",
             **{f"P({symbol} | history)": probability for symbol, probability in row.items()}}
            for history, row in sorted(result["conditional_probabilities"].items())
        ]
        st.dataframe(probabilities, hide_index=True, width="stretch")
        st.caption("Unobserved histories are omitted because their probabilities are unestimated.")


def main():
    st.set_page_config(page_title="Markov Order Classifier", layout="wide")
    st.title("Markov Order Classifier")
    tabs = st.tabs(["Exact Classification", "Process Generation", "Representation Minimization", "Statistical Classification"])
    for tab, render in zip(tabs, [exact_tab, generation_tab, minimization_tab, statistical_tab]):
        with tab:
            render()


if __name__ == "__main__":
    main()
