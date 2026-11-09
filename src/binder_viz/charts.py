from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve

from .analysis import sequence_observations
from .experiments import sequence_labels
from .registry import direction_sign


COLORS = {"none": "#333838", "weak": "#ca7054", "medium": "#e7d04f", "strong": "#2ac565"}
PALETTE = ["#087f75", "#e68282", "#d674a2", "#557da1", "#8b668a"]
PNG_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": "binder_viz_chart", "scale": 2}}


def layout(figure: go.Figure, title: str, height: int = 400) -> go.Figure:
    figure.update_layout(title={"text": title, "font": {"size": 16}}, height=height,
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=PALETTE,
                         font={"family": "IBM Plex Sans, Helvetica Neue, sans-serif"},
                         margin={"l": 35, "r": 20, "t": 55, "b": 55},
                         legend={"title": None, "orientation": "h", "y": -0.2})
    figure.update_xaxes(showgrid=False, automargin=True, zeroline=False)
    figure.update_yaxes(gridcolor="rgba(128,128,128,0.2)", automargin=True, zeroline=False)
    return figure


def ranking_chart(ranking: pd.DataFrame) -> go.Figure:
    shown = ranking.dropna(subset=["value"]).head(20).iloc[::-1]
    figure = go.Figure(go.Bar(x=shown.value, y=shown.score, orientation="h", marker_color="#087f75",
                             error_x={"type": "data", "array": (shown.ci_high - shown.value).clip(lower=0),
                                      "arrayminus": (shown.value - shown.ci_low).clip(lower=0)},
                             customdata=np.column_stack([shown.n, shown.ci_low, shown.ci_high]),
                             hovertemplate="%{y}<br>%{x:.3f} [%{customdata[1]:.3f}, %{customdata[2]:.3f}]<br>N=%{customdata[0]}<extra></extra>"))
    return layout(figure, str(ranking.metric.iloc[0]), max(300, len(shown) * 28 + 100))


def curves(frame: pd.DataFrame, scores: list[str], directions: dict[str, str], kind="ROC") -> go.Figure:
    figure = go.Figure()
    for column in scores:
        observations = sequence_observations(frame, [column], "binds")
        if observations.binds.nunique() != 2 or observations.empty:
            continue
        values = observations[column] * direction_sign(directions[column])
        if kind == "ROC":
            horizontal, vertical, _ = roc_curve(observations.binds, values)
        else:
            vertical, horizontal, _ = precision_recall_curve(observations.binds, values)
        figure.add_trace(go.Scatter(x=horizontal, y=vertical, mode="lines", name=column))
    if kind == "ROC":
        figure.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line={"color": "#a5b1b1", "dash": "dot"})
    figure.update_xaxes(title="False positive rate" if kind == "ROC" else "Recall", range=[0, 1])
    figure.update_yaxes(title="True positive rate" if kind == "ROC" else "Precision", range=[0, 1.02])
    return layout(figure, f"{kind} curves")


def distribution(frame: pd.DataFrame, score: str) -> go.Figure:
    data = frame.dropna(subset=[score, "binding_class"]).drop_duplicates("sequence_id").copy()
    data["display_name"] = sequence_labels(data)
    figure = px.box(data, x="binding_class", y=score, color="binding_class", points="all",
                    color_discrete_map=COLORS, category_orders={"binding_class": list(COLORS)}, hover_data=["display_name"])
    return layout(figure, "Score by binding class")


def kinetics(frame: pd.DataFrame, score: str) -> go.Figure:
    data = frame[frame.source.eq("internal")].dropna(subset=[score, "log10_kd"]).drop_duplicates("sequence_id").copy()
    data["display_name"] = sequence_labels(data)
    data["measurement"] = np.where(data.kd_qualifier.isin(["<", ">", "<=", ">="]), "Censored bound", "Measured")
    figure = px.scatter(data, x=score, y="log10_kd", color="target", symbol="measurement",
                        symbol_map={"Measured": "circle", "Censored bound": "triangle-up"},
                        hover_data=["display_name", "kd_nM", "kd_qualifier"], color_discrete_sequence=PALETTE)
    return layout(figure, "Internal kinetics")


def empty_plot(title: str, message: str) -> go.Figure:
    figure = layout(go.Figure(), title)
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    return figure


def numeric_points(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not set(columns).issubset(frame):
        return frame.iloc[:0].copy()
    data = frame.copy()
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    data["display_name"] = sequence_labels(data)
    return data.dropna(subset=columns)


def affinity_marker_sizes(kd_nM: pd.Series) -> pd.Series:
    values = pd.to_numeric(kd_nM, errors="coerce")
    sizes = pd.Series(8.0, index=values.index)
    valid = values.gt(0) & np.isfinite(values)
    if valid.any():
        affinity = -np.log10(values[valid])
        spread = affinity.max() - affinity.min()
        sizes.loc[valid] = 13.0 if np.isclose(spread, 0) else 8 + 10 * (affinity - affinity.min()) / spread
    return sizes


BINDING_LEVELS = {"none": 0.0, "weak": 1.0, "medium": 2.0, "strong": 3.0, "nonbinding": 0.0, "binding": 3.0}
BINDING_SCALE = [[0.0, COLORS["none"]], [1 / 3, COLORS["weak"]], [2 / 3, COLORS["medium"]], [1.0, COLORS["strong"]]]
MARKER_KINDS = {"graded": ("", "circle", None), "ungraded": (" (binding, ungraded)", "square", "#557da1"),
                "unmeasured": (" (unmeasured)", "diamond", "#ccd2d2")}


def method_agreement(frame: pd.DataFrame, x: str, y: str, selected_ids=(), color_by: str = "binding_class") -> go.Figure:
    data = numeric_points(frame, [x, y]).drop_duplicates("sequence_id")
    if data.empty:
        return empty_plot("Cross-model agreement", "No paired numeric scores available")
    binary = data.get("binds", pd.Series(index=data.index, dtype="boolean"))
    if color_by == "binds":
        labels = binary.map({True: "binding", False: "nonbinding"}).fillna("unmeasured")
        ticks = {0.0: "nonbinding", 3.0: "binding"}
    else:
        labels = data.get("binding_class", pd.Series(index=data.index, dtype=str)).fillna("unmeasured")
        labels = labels.mask(labels.eq("unmeasured") & binary.eq(True).fillna(False), "binding (ungraded)")
        labels = labels.mask(labels.eq("unmeasured") & binary.eq(False).fillna(False), "none")
        ticks = {0.0: "none", 1.0: "weak", 2.0: "medium", 3.0: "strong"}
    levels = labels.map(BINDING_LEVELS)
    kinds = pd.Series("graded", index=data.index).mask(levels.isna(), "unmeasured").mask(labels.eq("binding (ungraded)"), "ungraded")
    kd = data.get("kd_nM", pd.Series(np.nan, index=data.index))
    if "source" in data:
        kd = kd.where(data.source.eq("internal"))
    if "kd_qualifier" in data:
        kd = kd.where(data.kd_qualifier.eq("="))
    sizes = affinity_marker_sizes(kd)
    targets = data.get("target", pd.Series("all", index=data.index)).astype(str)
    figure = go.Figure()
    scale_shown = False
    # One legend entry per target and marker kind, each individually toggleable.
    for target in sorted(targets.unique()):
        for kind, (suffix, symbol, color) in MARKER_KINDS.items():
            points = data[targets.eq(target) & kinds.eq(kind)]
            if points.empty:
                continue
            marker = {"size": sizes.loc[points.index], "opacity": 0.75, "symbol": symbol,
                      "line": {"width": 0.6, "color": "rgba(80,80,80,0.45)"}}
            if color:
                marker["color"] = color
            else:
                marker.update(color=levels.loc[points.index], colorscale=BINDING_SCALE, cmin=0, cmax=3,
                              showscale=not scale_shown,
                              colorbar={"title": "Binding", "tickvals": list(ticks), "ticktext": list(ticks.values()),
                                        "thickness": 12, "len": 0.75})
                scale_shown = True
            figure.add_trace(go.Scatter(x=points[x], y=points[y], mode="markers", legendgroup=target,
                                        name=f"{target}{suffix}", marker=marker, text=points.display_name,
                                        customdata=labels.loc[points.index],
                                        hovertemplate=f"%{{text}}<br>{x}=%{{x:.3g}}<br>{y}=%{{y:.3g}}"
                                                      f"<br>%{{customdata}}<extra>{target}</extra>"))
    selected = data[data.sequence_id.isin(selected_ids)]
    if not selected.empty:
        figure.add_trace(go.Scatter(x=selected[x], y=selected[y], mode="markers", name="Selected candidates",
                                    marker={"size": sizes.loc[selected.index] + 6, "symbol": "circle-open", "color": "#d2a12b", "line": {"width": 2}},
                                    text=selected.display_name, hovertemplate="%{text}<extra>Selected</extra>"))
    if x.rsplit("__", 1)[0] == y.rsplit("__", 1)[0]:
        lower, upper = data[[x, y]].min().min(), data[[x, y]].max().max()
        figure.add_shape(type="line", x0=lower, y0=lower, x1=upper, y1=upper, line={"dash": "dash", "color": "#899596"})
    figure.update_xaxes(title=x)
    figure.update_yaxes(title=y)
    figure = layout(figure, "Cross-model agreement", 470)
    figure.update_layout(legend={"title": "Click to toggle", "groupclick": "toggleitem"})
    return figure


def score_distribution(full: pd.DataFrame, active: pd.DataFrame, score: str, selected_ids=(), replicate_rows: bool = False) -> go.Figure:
    reference = numeric_points(full, [score])
    filtered = numeric_points(active, [score])
    if not replicate_rows:
        reference = reference.drop_duplicates("sequence_id")
        filtered = filtered.drop_duplicates("sequence_id")
    if reference.empty:
        return empty_plot("Score distribution", "No numeric scores available")
    figure = go.Figure()
    reference_label = "All loaded retained rows" if replicate_rows else "All loaded sequences"
    for data, name, color in [(reference, reference_label, "#899596"), (filtered, "Current filtered set", "#087f75")]:
        figure.add_trace(go.Histogram(x=data[score], name=name, histnorm="probability density", opacity=0.5,
                                      nbinsx=30, marker_color=color))
    selected = filtered[filtered.sequence_id.isin(selected_ids)]
    if not selected.empty:
        figure.add_trace(go.Scatter(x=selected[score], y=np.zeros(len(selected)), mode="markers", name="Selected candidates",
                                    marker={"symbol": "line-ns", "size": 18, "color": "#d2a12b"}, text=selected.display_name,
                                    hovertemplate="%{text}<br>%{x:.3g}<extra>Selected</extra>"))
    figure.update_layout(barmode="overlay")
    figure.update_xaxes(title=score)
    figure.update_yaxes(title="Density")
    return layout(figure, "Score distribution")


def stratified_distribution(frame: pd.DataFrame, score: str, group: str) -> go.Figure:
    data = numeric_points(frame, [score])
    if data.empty or group not in data:
        return empty_plot("Score by group", "No group measurements available")
    data[group] = data[group].fillna("unmeasured").astype(str)
    data = data.drop_duplicates(["sequence_id", group])
    figure = px.box(data, x=group, y=score, color=group, points="all", hover_data=["display_name"],
                    color_discrete_map=COLORS if group == "binding_class" else None, color_discrete_sequence=PALETTE)
    return layout(figure, f"Score by {group}")


def selected_method_blocks(table: pd.DataFrame, selected_ids, models: list[str]) -> go.Figure:
    selected = table[table.sequence_id.isin(selected_ids)].drop_duplicates("sequence_id").copy()
    selected["display_name"] = sequence_labels(selected)
    labels = {"composite": "Composite", "score_refold": "Refolding", "score_electrostatics": "Electrostatics", "score_rosetta": "Rosetta"}
    figure = go.Figure()
    for model in models:
        for row in selected.to_dict("records"):
            fields = [name for name in labels if pd.notna(row.get(f"{name}__{model}"))]
            if fields:
                figure.add_trace(go.Bar(x=[labels[name] for name in fields], y=[row[f"{name}__{model}"] for name in fields],
                                        name=f"{row['display_name'][:36]} / {model}", opacity=0.8,
                                        text=[row["display_name"]] * len(fields),
                                        hovertemplate=f"%{{text}}<br>{model}<br>%{{x}}=%{{y:.1f}}<extra></extra>"))
    if not figure.data:
        return empty_plot("Evidence blocks", "No selected composite evidence available")
    figure.update_layout(barmode="group")
    figure.update_yaxes(title="Percentile-derived score", range=[0, 100])
    return layout(figure, "Per-sequence evidence blocks", 480)


def kd_scatter(frame: pd.DataFrame, score: str) -> go.Figure:
    if "source" not in frame or "kd_nM" not in frame:
        return empty_plot("Observed affinity", "No internal Kd measurements available")
    data = numeric_points(frame[frame.source.eq("internal")], [score, "kd_nM"])
    data = data[data.kd_nM.gt(0)]
    if data.empty:
        return empty_plot("Observed affinity", "No positive internal Kd measurements available")
    data["measurement"] = np.where(data.kd_qualifier.isin(["<", ">", "<=", ">="]), "Censored bound", "Measured")
    figure = px.scatter(data, x=score, y="kd_nM", color="target", symbol="measurement",
                        symbol_map={"Measured": "circle", "Censored bound": "triangle-up"},
                        hover_data=["display_name", "kd_qualifier"], color_discrete_sequence=PALETTE)
    figure.update_yaxes(title="Observed Kd (nM; lower is tighter)", type="log")
    return layout(figure, "Observed internal affinity", 450)


def candidate_nomination_budget(table: pd.DataFrame, score: str, direction: str, selected_ids=(), count: int = 20) -> go.Figure:
    data = numeric_points(table, [score]).drop_duplicates("sequence_id")
    data = data.sort_values(score, ascending=direction == "lower_better", kind="stable").head(count)
    if data.empty:
        return empty_plot("Candidate ranking", "No eligible candidate scores available")
    colors = np.where(data.sequence_id.isin(selected_ids), "#d2a12b", "#087f75")
    figure = go.Figure(go.Bar(x=data[score], y=data.sequence_id, orientation="h", marker_color=colors,
                             text=data.display_name, textposition="none",
                             customdata=np.column_stack([data.target, np.arange(1, len(data) + 1)]),
                             hovertemplate="%{text}<br>target=%{customdata[0]}<br>rank=%{customdata[1]}<br>score=%{x:.3g}<extra></extra>"))
    figure.update_yaxes(autorange="reversed", title="Sequence", tickmode="array", tickvals=data.sequence_id,
                        ticktext=data.display_name.str.slice(stop=36))
    figure.update_xaxes(title=score)
    figure = layout(figure, "Candidate ranking", max(400, 26 * len(data) + 100))
    figure.update_layout(margin_l=155)
    return figure


def cluster_distribution(selected: pd.DataFrame) -> go.Figure:
    if selected.empty or "cluster_id" not in selected:
        return empty_plot("Sequence diversity", "No clustered selection available")
    counts = selected.dropna(subset=["cluster_id"]).groupby(["target", "cluster_id"]).size().rename("sequences").reset_index()
    if counts.empty:
        return empty_plot("Sequence diversity", "No assigned sequence clusters")
    counts["cluster_id"] = counts.cluster_id.astype(str)
    figure = px.bar(counts, x="cluster_id", y="sequences", color="target", barmode="group", color_discrete_sequence=PALETTE)
    figure.update_yaxes(title="Selected sequences", dtick=1)
    return layout(figure, "Sequence diversity by cluster")