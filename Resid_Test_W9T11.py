# =========================
# Residual Testing V.1
# Vin dela Serna Lopez
# Dec 25 2025
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = (
    "/Users/alvindclopez/Library/CloudStorage/"
    "GoogleDrive-alvindelasernalopez@gmail.com/My Drive/Research/"
    "Characterization of Jupiter-like Exoplanet in a Main Sequence Star/"
    "Sources/PSCompPars_Xmas25.csv"
)

# Robustness proxy threshold for Fig 11′ (dex spread in log10|Pdot| across Q models)
DELTA_MODEL_STABLE_THRESH_DEX = 1.0


# =========================
# Helpers
# =========================
def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def new_fig(window_title: str, figsize=(8.5, 5.5)):
    """
    Title in the window toolbar/titlebar (NOT inside the axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    try:
        fig.canvas.manager.set_window_title(window_title)
    except Exception:
        pass
    return fig, ax


def clean_required(df: pd.DataFrame) -> pd.DataFrame:
    # Required columns
    for c in ["host_bin", "t_a_wein_yr"]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    df = df.copy()

    # Clean host_bin & tau
    df["host_bin"] = df["host_bin"].astype(str).str.strip()
    df["t_a_wein_yr"] = to_num(df["t_a_wein_yr"])

    bad_host = df["host_bin"].isin(["", "nan", "None", "NA", "N/A"])
    bad_tau = df["t_a_wein_yr"].isna() | (df["t_a_wein_yr"] <= 0)

    df = df.loc[~(bad_host | bad_tau)].copy()

    # Orbital period days (prefer pl_orbper)
    if "pl_orbper" in df.columns:
        df["pl_orbper"] = to_num(df["pl_orbper"])
        df["Porb_day"] = df["pl_orbper"]
    elif "secs_pl_orbper" in df.columns:
        df["secs_pl_orbper"] = to_num(df["secs_pl_orbper"])
        df["Porb_day"] = df["secs_pl_orbper"] / 86400.0
    else:
        raise KeyError("Need 'pl_orbper' or 'secs_pl_orbper'.")

    df = df.loc[df["Porb_day"].notna() & (df["Porb_day"] > 0)].copy()

    # Logs
    df["log10_Porb_day"] = np.log10(df["Porb_day"])
    df["log10_tauW_yr"] = np.log10(df["t_a_wein_yr"])

    # Planet mass
    if "pl_bmassj" in df.columns:
        df["pl_bmassj"] = to_num(df["pl_bmassj"])
        df.loc[df["pl_bmassj"] <= 0, "pl_bmassj"] = np.nan
        df["log10_Mp_MJ"] = np.log10(df["pl_bmassj"])
    else:
        df["pl_bmassj"] = np.nan
        df["log10_Mp_MJ"] = np.nan

    return df


def scatter_by_host(ax, df, xcol, ycol, host_bins, s=28, alpha=0.75):
    for hb in host_bins:
        d = df[df["host_bin"] == hb]
        ax.scatter(
            d[xcol],
            d[ycol],
            s=s,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.3,
            label=str(hb),
        )


def tau_from_pdot_msyr(P_day, Pdot_msyr):
    """
    Physics-consistent conversion using stated relation:
        Pdot = -(3/2) P / tau  ->  tau = (3/2) P / |Pdot|
    P in ms, Pdot in ms/yr -> tau in yr.
    """
    P_ms = P_day * 86400.0 * 1000.0
    return 1.5 * P_ms / np.abs(Pdot_msyr)

df0 = pd.read_csv(CSV_PATH)
df = clean_required(df0)

host_bins = sorted(df["host_bin"].unique())

# =========================
# Fig 1 — Porb–Mp sample map
# =========================
if df["pl_bmassj"].notna().any():
    fig1, ax1 = new_fig("Fig 1 — Porb–Mp sample map")
    for hb in host_bins:
        d = df[(df["host_bin"] == hb) & df["pl_bmassj"].notna()]
        ax1.scatter(
            d["Porb_day"],
            d["pl_bmassj"],
            s=30,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.3,
            label=hb,
        )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$P_{\rm orb}$ (day)")
    ax1.set_ylabel(r"$M_p$ ($M_J$)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8, frameon=True)

# =========================
# Fig 2 — Host-bin completeness
# =========================
fig2, ax2 = new_fig("Fig 2 — Host-bin completeness", figsize=(9.0, 5.5))

if ("mass_class" in df.columns) and ("evol_stage" in df.columns):
    df["mass_class"] = df["mass_class"].astype(str).str.strip()
    df["evol_stage"] = df["evol_stage"].astype(str).str.strip()
    df["bin3"] = df["mass_class"] + " × " + df["evol_stage"]

    counts = (
        df.groupby(["host_bin", "bin3"]).size().unstack(fill_value=0).reindex(host_bins)
    )

    x = np.arange(len(counts.index))
    bottom = np.zeros(len(counts.index))
    for col in counts.columns:
        ax2.bar(x, counts[col].to_numpy(), bottom=bottom, label=col)
        bottom += counts[col].to_numpy()

    ax2.legend(fontsize=8, frameon=True, ncol=2)
else:
    counts = df["host_bin"].value_counts().reindex(host_bins).fillna(0).astype(int)
    ax2.bar(np.arange(len(counts)), counts.to_numpy())

ax2.set_xticks(np.arange(len(host_bins)))
ax2.set_xticklabels(host_bins, rotation=30, ha="right")
ax2.set_xlabel("host_bin")
ax2.set_ylabel("count")
ax2.grid(True, axis="y", alpha=0.25)

# =========================
# Fig 3 — Baseline ⟨τ⟩ vs Porb (log–log)
# =========================
fig3, ax3 = new_fig("Fig 3 — Baseline ⟨τ⟩ vs Porb (WN)", figsize=(8.5, 5.5))
scatter_by_host(ax3, df, "log10_Porb_day", "log10_tauW_yr", host_bins)
ax3.set_xlabel(r"$\log_{10}(P_{\rm orb}/{\rm day})$")
ax3.set_ylabel(r"$\log_{10}(\langle\tau\rangle/{\rm yr})$")
ax3.grid(True, alpha=0.25)
ax3.legend(fontsize=8, frameon=True)

# =========================
# Fig 4 — Baseline power-law diagnostic (per host_bin fit)
# =========================
fig4, ax4 = new_fig("Fig 4 — Baseline power-law diagnostic", figsize=(8.5, 5.5))

fit_text = []
for hb in host_bins:
    d = df[df["host_bin"] == hb].dropna(subset=["log10_Porb_day", "log10_tauW_yr"])
    ax4.scatter(
        d["log10_Porb_day"],
        d["log10_tauW_yr"],
        s=26,
        alpha=0.6,
        edgecolor="white",
        linewidth=0.3,
        label=hb,
    )
    if len(d) >= 3:
        slope, intercept = np.polyfit(d["log10_Porb_day"], d["log10_tauW_yr"], 1)
        xx = np.linspace(df["log10_Porb_day"].min(), df["log10_Porb_day"].max(), 200)
        ax4.plot(xx, intercept + slope * xx, linewidth=2)
        fit_text.append(f"{hb}: slope={slope:.2f}")

ax4.set_xlabel(r"$\log_{10}(P_{\rm orb}/{\rm day})$")
ax4.set_ylabel(r"$\log_{10}(\langle\tau\rangle/{\rm yr})$")
ax4.grid(True, alpha=0.25)


ax4.legend(fontsize=8, frameon=True)


"""
if fit_text:
    ax4.text(
        0.02,
        0.02,
        "\n".join(fit_text),
        transform=ax4.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.2),
    )
"""

def place_text_in_empty_corner(ax, x, y, txt):
    """
    x, y are 1D arrays of plotted data in DATA coordinates.
    Tries 4 corners in axes coords and picks the one with the fewest points.
    """
    corners = [
        (0.02, 0.98, "left",  "top"),     # top-left
        (0.98, 0.98, "right", "top"),     # top-right
        (0.02, 0.02, "left",  "bottom"),  # bottom-left
        (0.98, 0.02, "right", "bottom"),  # bottom-right
    ]

    # convert all data points to axes-fraction coordinates
    pts_disp = ax.transData.transform(np.column_stack([x, y]))
    pts_axes = ax.transAxes.inverted().transform(pts_disp)

    # count points near each corner (within a small square region)
    r = 0.20  # "corner box" size in axes fraction (tune 0.15–0.25)
    scores = []
    for cx, cy, ha, va in corners:
        mask = (np.abs(pts_axes[:, 0] - cx) < r) & (np.abs(pts_axes[:, 1] - cy) < r)
        scores.append(mask.sum())

    best = int(np.argmin(scores))
    cx, cy, ha, va = corners[best]

    ax.text(
        cx, cy, txt,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.2),
        zorder=10
    )

# usage (after plotting):
x_all = df["log10_Porb_day"].to_numpy()
y_all = df["log10_tauW_yr"].to_numpy()
place_text_in_empty_corner(ax4, x_all, y_all, "\n".join(fit_text))


# =========================================================
# Fig 5′ — Model overlay: WN τ vs τ implied by equilibrium-tide Q′ models
# =========================================================
QCOLS = [c for c in ["Pdot_Q6_ms_yr", "Pdot_Q7_ms_yr", "Pdot_Q8_ms_yr"] if c in df.columns]
have_any_Q = len(QCOLS) > 0

if have_any_Q:
    for c in QCOLS:
        df[c] = to_num(df[c])
        df.loc[df[c] == 0, c] = np.nan

    if "Pdot_Q6_ms_yr" in df.columns:
        df["tau_Q6_yr"] = tau_from_pdot_msyr(df["Porb_day"], df["Pdot_Q6_ms_yr"])
    if "Pdot_Q7_ms_yr" in df.columns:
        df["tau_Q7_yr"] = tau_from_pdot_msyr(df["Porb_day"], df["Pdot_Q7_ms_yr"])
    if "Pdot_Q8_ms_yr" in df.columns:
        df["tau_Q8_yr"] = tau_from_pdot_msyr(df["Porb_day"], df["Pdot_Q8_ms_yr"])

    fig5, ax5 = new_fig("Fig 5′ — Mapping overlay: WN ⟨τ⟩ vs equilibrium-tide τ", figsize=(8.5, 5.5))

    # Baseline WN τ
    ax5.scatter(
        df["log10_Porb_day"],
        np.log10(df["t_a_wein_yr"]),
        s=22,
        alpha=0.35,
        edgecolor="white",
        linewidth=0.3,
        label="Baseline ⟨τ⟩",
    )

    # Overlays in τ-space (log)
    if "tau_Q6_yr" in df.columns:
        ax5.scatter(df["log10_Porb_day"], np.log10(df["tau_Q6_yr"]), s=18, alpha=0.45, marker="^", label="τ(Q′=10⁶)")
    if "tau_Q7_yr" in df.columns:
        ax5.scatter(df["log10_Porb_day"], np.log10(df["tau_Q7_yr"]), s=18, alpha=0.45, marker="o", label="τ(Q′=10⁷)")
    if "tau_Q8_yr" in df.columns:
        ax5.scatter(df["log10_Porb_day"], np.log10(df["tau_Q8_yr"]), s=18, alpha=0.45, marker="v", label="τ(Q′=10⁸)")

    ax5.set_xlabel(r"$\log_{10}(P_{\rm orb}/{\rm day})$")
    ax5.set_ylabel(r"$\log_{10}(\tau/{\rm yr})$")
    ax5.grid(True, alpha=0.25)
    ax5.legend(fontsize=8, frameon=True)

# =========================
# Residuals (internal diagnostic): log τ - per-bin fit(log τ | log P)
# =========================
df["resid_logtau_internal"] = np.nan
for hb in host_bins:
    d = df[df["host_bin"] == hb].dropna(subset=["log10_Porb_day", "log10_tauW_yr"])
    if len(d) < 3:
        continue
    slope, intercept = np.polyfit(d["log10_Porb_day"], d["log10_tauW_yr"], 1)
    m = df["host_bin"] == hb
    df.loc[m, "resid_logtau_internal"] = df.loc[m, "log10_tauW_yr"] - (intercept + slope * df.loc[m, "log10_Porb_day"])

# =========================
# Fig 6 — Residual vs Porb
# =========================
fig6, ax6 = new_fig("Fig 6 — Residual vs Porb", figsize=(8.5, 5.5))
for hb in host_bins:
    d = df[(df["host_bin"] == hb) & df["resid_logtau_internal"].notna()]
    ax6.scatter(
        d["log10_Porb_day"],
        d["resid_logtau_internal"],
        s=28,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.3,
        label=hb,
    )
ax6.axhline(0.0, linewidth=1.5)
ax6.set_xlabel(r"$\log_{10}(P_{\rm orb}/{\rm day})$")
ax6.set_ylabel(r"$\Delta\log_{10}\tau$")
ax6.grid(True, alpha=0.25)
ax6.legend(fontsize=8, frameon=True)

# =========================
# Fig 7 — Residual distribution (χ proxy)
# =========================
fig7, ax7 = new_fig("Fig 7 — Residual distribution (χ proxy)", figsize=(8.5, 5.5))
for hb in host_bins:
    arr = df.loc[df["host_bin"] == hb, "resid_logtau_internal"].dropna().to_numpy()
    if len(arr) < 2:
        continue
    ax7.hist(arr, bins=25, alpha=0.35, density=True, label=hb)
ax7.axvline(0.0, linewidth=1.5)
ax7.set_xlabel(r"$\Delta\log_{10}\tau$ (dex)")
ax7.set_ylabel("density")
ax7.grid(True, alpha=0.25)
ax7.legend(fontsize=8, frameon=True)

# =========================
# Fig 8 — Residual vs Mp
# =========================
if df["pl_bmassj"].notna().any():
    fig8, ax8 = new_fig("Fig 8 — Residual vs planet mass", figsize=(8.5, 5.5))
    d8 = df[df["log10_Mp_MJ"].notna() & df["resid_logtau_internal"].notna()]
    for hb in host_bins:
        dd = d8[d8["host_bin"] == hb]
        ax8.scatter(
            dd["log10_Mp_MJ"],
            dd["resid_logtau_internal"],
            s=28,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.3,
            label=hb,
        )
    ax8.axhline(0.0, linewidth=1.5)
    ax8.set_xlabel(r"$\log_{10}(M_p/M_J)$")
    ax8.set_ylabel(r"$\Delta\log_{10}\tau$ (dex)")
    ax8.grid(True, alpha=0.25)
    ax8.legend(fontsize=8, frameon=True)

# =========================================================
# Figs 9′–11′ — Fixed warnings: robust Delta_model computation
# =========================================================
if have_any_Q:
    # Coerce to numeric, take abs, and NaN-out invalid/<=0 values BEFORE log10
    for c in QCOLS:
        df[c] = to_num(df[c]).abs()
        df.loc[~np.isfinite(df[c]) | (df[c] <= 0), c] = np.nan

    Q = df[QCOLS].to_numpy(dtype=float)  # N x k

    with np.errstate(divide="ignore", invalid="ignore"):
        logQ = np.log10(Q)  # safe: non-positive already NaN

    finite_ct = np.sum(np.isfinite(logQ), axis=1)

    Delta = np.full(len(df), np.nan)
    ok = finite_ct >= 2  # require at least 2 models per row
    if np.any(ok):
        Delta[ok] = np.nanmax(logQ[ok, :], axis=1) - np.nanmin(logQ[ok, :], axis=1)

    df["Delta_model"] = Delta

    # ---- Fig 9′: Delta_model vs Porb
    fig9, ax9 = new_fig("Fig 9′ — Δ_model vs Porb (Q′ spread)", figsize=(8.5, 5.5))
    for hb in host_bins:
        d = df[(df["host_bin"] == hb) & df["Delta_model"].notna()]
        ax9.scatter(
            d["log10_Porb_day"],
            d["Delta_model"],
            s=28,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.3,
            label=hb,
        )
    ax9.set_xlabel(r"$\log_{10}(P_{\rm orb}/{\rm day})$")
    ax9.set_ylabel(r"$\Delta_{\rm model}$ in $\log_{10}|\dot P|$ across Q′ models (dex)")
    ax9.grid(True, alpha=0.25)
    ax9.legend(fontsize=8, frameon=True)

    # ---- Fig 10′: Delta_model distribution grouped by WN vs Q7 dominance (if ratio exists)
    if "ratio_WN_to_Q7" in df.columns:
        df["ratio_WN_to_Q7"] = to_num(df["ratio_WN_to_Q7"])
        df.loc[df["ratio_WN_to_Q7"] <= 0, "ratio_WN_to_Q7"] = np.nan
        df["log10_ratio_WN_Q7"] = np.log10(df["ratio_WN_to_Q7"])

        df["regime_WNvsQ7"] = np.where(df["log10_ratio_WN_Q7"].notna() & (df["log10_ratio_WN_Q7"] >= 0),
                                       "WN stronger",
                                       "Q7 stronger")

        fig10, ax10 = new_fig("Fig 10′ — Δ_model distribution by WN vs Q7 dominance", figsize=(8.5, 5.5))
        for reg in ["WN stronger", "Q7 stronger"]:
            arr = df.loc[df["regime_WNvsQ7"] == reg, "Delta_model"].dropna().to_numpy()
            if len(arr) < 2:
                continue
            ax10.hist(arr, bins=25, alpha=0.45, density=True, label=reg)
        ax10.set_xlabel(r"$\Delta_{\rm model}$ (dex)")
        ax10.set_ylabel("density")
        ax10.grid(True, alpha=0.25)
        #ax10.legend(fontsize=9, frameon=True)

        # ---- Fig 11′: Stability proxy by host_bin using a Δ_model threshold
        df["stable_proxy"] = df["Delta_model"] < DELTA_MODEL_STABLE_THRESH_DEX
        stab = df.groupby("host_bin")["stable_proxy"].mean().reindex(host_bins)

        fig11, ax11 = new_fig("Fig 11′ — Robustness proxy by host_bin", figsize=(8.8, 5.5))
        ax11.bar(np.arange(len(stab.index)), stab.to_numpy())
        ax11.set_xticks(np.arange(len(stab.index)))
        ax11.set_xticklabels(stab.index, rotation=30, ha="right")
        ax11.set_ylim(0, 1)
        ax11.set_xlabel("host_bin")
        ax11.set_ylabel(f"fraction with Δ_model < {DELTA_MODEL_STABLE_THRESH_DEX:.1f} dex")
        ax11.grid(True, axis="y", alpha=0.25)

plt.show()