#!/usr/bin/env python3
"""
Generate an interactive HTML background budget plot with an IV radius slider.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd


# ── Hardcoded inputs ─────────────────────────────────────────────────────
DEFAULT_INPUT_FILE = Path(__file__).resolve().parent / (
    "Summary_D-047_v86_250113-233135_2025-09-11.xlsx"
)
FALLBACK_INPUT_FILE = Path(
    "/Users/antoine/My Drive/Documents/Th\u00e8se/Nickel Cryostats/"
    "nickel-cryostats-analysis/budget/"
    "Summary_D-047_v86_250113-233135_2025-09-11.xlsx"
)

MIN_COUNT = 1e-4
DEFAULT_IV_RADIUS_MM = 1100.0
BASELINE_RADIUS_MM = 1691.0
MU = 0.00674403  # HFE-7200 at 165 K (approx. LXe temperature), 1/mm

DEFAULT_RADIUS_MIN_MM = 800.0
DEFAULT_RADIUS_MAX_MM = 2000.0
DEFAULT_RADIUS_STEP_MM = 10.0

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"

OUTSIDE_HFE = {
    "Cryopit concrete and shotcrete",
    "Outer Vessel Support",
    "Outer Cryostat",
    "Inner Vessel Support",
    "Inner Cryostat MLI",
    "Inner Cryostat",
    "CRE Transition Enclosures",
    "CRE Transition Boards",
    "PRE Transition Enclosures",
    "PRE Transition Boards",
    "OD: PMTs, PMT cable, and PMT mounts",
    "OD: Tank",
}

COMPONENT_PREFIX_RENAMES = [
    ("Outer Cryostat Support", "Outer Vessel Support"),
    ("Inner Cryostat Support", "Inner Vessel Support"),
    ("Outer Cryostat (", "Outer Cryostat"),
    ("Inner Cryostat (", "Inner Cryostat"),
    ("Outer Cryostat Liner", "Outer Cryostat Liner"),
    ("Inner Cryostat Liner", "Inner Cryostat Liner"),
    ("Outer Cryostat Feedthrough", "Outer Cryostat Feedthrough"),
    ("Inner Cryostat Feedthrough", "Inner Cryostat Feedthrough"),
    ("Inactive LXe", "Skin LXe"),
    ("Active LXe", "TPC LXe"),
]


def sqrt_sum_sq(series: pd.Series) -> float:
    """Return sqrt(sum(x^2)) for a numeric Series."""
    arr = series.to_numpy(dtype=float)
    return float(np.sqrt(np.sum(arr ** 2)))


def load_grouped(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(input_file)

    df = pd.read_excel(
        input_file,
        sheet_name="Summary",
        usecols="A:I",
        engine="openpyxl",
        skipfooter=1,
    ).rename(columns={"Background [counts/y/2t/FWHM]": "TG Mean", "Error": "TG Spread"})

    df["TG Mean"] = pd.to_numeric(df["TG Mean"], errors="coerce")
    df["TG Spread"] = pd.to_numeric(df["TG Spread"], errors="coerce")

    # Normalize component names
    c = df["Component"].astype(str).str.strip()
    for prefix, repl in COMPONENT_PREFIX_RENAMES:
        c = c.where(~c.str.startswith(prefix, na=False), repl)
    df["Component"] = c

    # Intrinsic tagging/cleanup
    df.loc[
        df["Isotope"].astype(str).str.startswith("bb2n", na=False),
        "Category",
    ] = "Intrinsic Radioactivity"

    intrinsic = df["Category"].astype(str).str.startswith("Intrinsic", na=False)
    df.loc[
        intrinsic & df["Component"].astype(str).str.contains("LXe", na=False),
        "Component",
    ] = "LXe"

    df = df.loc[~df["Isotope"].isin(["bb0n", "Cs-137"]) & intrinsic].copy()

    grouped = (
        df.groupby("Component", dropna=False)
        .agg(
            TG_Mean=("TG Mean", "sum"),
            TG_Spread=("TG Spread", sqrt_sum_sq),
        )
        .sort_values("TG_Mean")
    )
    grouped = grouped[grouped["TG_Mean"] >= MIN_COUNT]
    return grouped


def build_html(
    grouped: pd.DataFrame,
    output_path: Path,
    radius_mm: float,
    radius_min_mm: float,
    radius_max_mm: float,
    radius_step_mm: float,
    embed_js: bool,
) -> None:
    labels = grouped.index.astype(str).tolist()
    means = grouped["TG_Mean"].to_numpy(dtype=float)
    spreads = grouped["TG_Spread"].to_numpy(dtype=float)
    is_external = np.array([label in OUTSIDE_HFE for label in labels], dtype=bool)

    y_vals = np.arange(len(labels), dtype=float)

    def split_values(mask: np.ndarray) -> dict[str, list[float]]:
        return {
            "x": means[mask].tolist(),
            "y": y_vals[mask].tolist(),
            "err": spreads[mask].tolist(),
            "text": [labels[i] for i, keep in enumerate(mask) if keep],
        }

    internal = split_values(~is_external)
    external = split_values(is_external)

    data_payload = {
        "internal": internal,
        "external": external,
        "labels": labels,
        "yVals": y_vals.tolist(),
        "minCount": MIN_COUNT,
        "baselineRadius": BASELINE_RADIUS_MM,
        "mu": MU,
    }

    if embed_js:
        try:
            plotly_js = urllib.request.urlopen(PLOTLY_CDN, timeout=30).read().decode(
                "utf-8"
            )
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(
                "Failed to download plotly.js for embedding."
            ) from exc
        plotly_tag = f"<script>{plotly_js}</script>"
    else:
        plotly_tag = f'<script src="{PLOTLY_CDN}"></script>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Background Budget (Interactive)</title>
  {plotly_tag}
  <style>
    :root {{
      --green: #006400;
      --blue: #00008b;
      --grid: #bdbdbd;
      --bg: #f7f6f2;
      --panel: #ffffff;
    }}
    body {{
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      margin: 0;
      background: var(--bg);
      color: #111;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px 20px 40px 20px;
    }}
    .panel {{
      background: var(--panel);
      border-radius: 12px;
      padding: 16px 18px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
      margin-bottom: 16px;
    }}
    .controls {{
      display: grid;
      gap: 8px;
    }}
    .controls label {{
      font-weight: 600;
    }}
    .controls input[type="range"] {{
      width: 100%;
    }}
    .meta {{
      font-size: 0.95rem;
      color: #333;
    }}
    #plot {{
      width: 100%;
      height: 720px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel controls">
      <label for="radius">IV radius (mm): <span id="radius-value">{radius_mm:.0f}</span></label>
      <input id="radius" type="range" min="{radius_min_mm}" max="{radius_max_mm}" step="{radius_step_mm}" value="{radius_mm}">
      <div class="meta" id="scale-value"></div>
    </div>
    <div class="panel">
      <div id="plot"></div>
    </div>
  </div>

  <script>
    const payload = {json.dumps(data_payload, ensure_ascii=True, separators=(",", ":"))};
    const minCount = payload.minCount;
    const baselineRadius = payload.baselineRadius;
    const mu = payload.mu;

    const internalTrace = {{
      type: "scatter",
      mode: "markers",
      name: "Internal components",
      x: payload.internal.x,
      y: payload.internal.y,
      text: payload.internal.text,
      marker: {{ color: "darkgreen", size: 10 }},
      error_x: {{ type: "data", array: payload.internal.err, thickness: 2 }},
      hovertemplate: "%{{text}}<br>%{{x:.3g}}<extra></extra>",
    }};

    const externalTrace = {{
      type: "scatter",
      mode: "markers",
      name: "External components",
      x: payload.external.x,
      y: payload.external.y,
      text: payload.external.text,
      marker: {{ color: "darkblue", size: 10 }},
      error_x: {{ type: "data", array: payload.external.err, thickness: 2 }},
      hovertemplate: "%{{text}}<br>%{{x:.3g}}<extra></extra>",
    }};

    const layout = {{
      title: {{
        text: "Background Budget (Inner Vessel r = {radius_mm:.0f} mm)",
        pad: {{ t: 10, b: 10 }},
      }},
      xaxis: {{
        title: "Background counts/(y/2t/FWHM)",
        type: "log",
        range: [Math.log10(minCount), 1],
        gridcolor: "#bdbdbd",
        zeroline: false,
      }},
      yaxis: {{
        tickvals: payload.yVals,
        ticktext: payload.labels,
        automargin: true,
      }},
      margin: {{ l: 260, r: 40, t: 70, b: 60 }},
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      showlegend: true,
    }};

    const config = {{ responsive: true, displayModeBar: true }};

    const plotTarget = document.getElementById("plot");
    const radiusInput = document.getElementById("radius");
    const radiusValue = document.getElementById("radius-value");
    const scaleValue = document.getElementById("scale-value");

    function updateExternal(radiusMm) {{
      const scale = Math.exp(-mu * (radiusMm - baselineRadius));
      const scaledX = payload.external.x.map((value) => value * scale);
      const scaledErr = payload.external.err.map((value) => value * scale);

      Plotly.restyle(
        plotTarget,
        {{
          x: [scaledX],
          "error_x.array": [scaledErr],
          name: [`External components (${{scale.toFixed(3)}}x)`],
        }},
        [1]
      );

      Plotly.relayout(plotTarget, {{
        "title.text": `Background Budget (Inner Vessel r = ${{radiusMm.toFixed(0)}} mm)`,
      }});

      radiusValue.textContent = radiusMm.toFixed(0);
      scaleValue.textContent = `External scale: ${{scale.toFixed(3)}}x`;
    }}

    Plotly.newPlot(plotTarget, [internalTrace, externalTrace], layout, config).then(() => {{
      updateExternal(parseFloat(radiusInput.value));
    }});

    radiusInput.addEventListener("input", (event) => {{
      updateExternal(parseFloat(event.target.value));
    }});
  </script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate interactive background budget HTML."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE if DEFAULT_INPUT_FILE.exists() else FALLBACK_INPUT_FILE,
        help="Path to the Summary Excel file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "background_budget_iv_adapted_interactive.html",
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_IV_RADIUS_MM,
        help="Default IV radius in mm.",
    )
    parser.add_argument(
        "--radius-min",
        type=float,
        default=DEFAULT_RADIUS_MIN_MM,
        help="Slider min radius in mm.",
    )
    parser.add_argument(
        "--radius-max",
        type=float,
        default=DEFAULT_RADIUS_MAX_MM,
        help="Slider max radius in mm.",
    )
    parser.add_argument(
        "--radius-step",
        type=float,
        default=DEFAULT_RADIUS_STEP_MM,
        help="Slider step size in mm.",
    )
    parser.add_argument(
        "--embed-js",
        action="store_true",
        help="Embed plotly.js in the HTML instead of using the CDN.",
    )
    args = parser.parse_args()

    grouped = load_grouped(args.input)
    build_html(
        grouped=grouped,
        output_path=args.output,
        radius_mm=args.radius,
        radius_min_mm=args.radius_min,
        radius_max_mm=args.radius_max,
        radius_step_mm=args.radius_step,
        embed_js=args.embed_js,
    )

    if args.embed_js:
        print(f"Wrote self-contained HTML to {args.output}")
    else:
        print(f"Wrote HTML (Plotly CDN) to {args.output}")


if __name__ == "__main__":
    main()
