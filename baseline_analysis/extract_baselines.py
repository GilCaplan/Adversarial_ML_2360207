"""
extract_baselines.py
--------------------
Reads all existing hallucination and transfer JSON result files and extracts
the UNSTEERED baseline responses that are embedded in every entry's "baseline"
field.  Produces:

  baselines_table.json   — machine-readable table
  (stdout)               — formatted tables for copy-paste into the paper

No model inference is performed.  Run with:
    python extract_baselines.py
"""

import os, json, glob
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULT_DIRS = {
    "weather":  "json_results",
    "emotions": "json_results_emotions",
}

# ---------------------------------------------------------------------------
# Keyword maps (same as visualize.py — we score both steered & baseline
# with identical logic so numbers are directly comparable)
# ---------------------------------------------------------------------------
EMOTION_KEYWORD_MAP = {
    "happy":   ["happy","smile","smiling","cheerful","vibrant","sunny","bright",
                "joy","lively","energetic","warm","bloom","sunflower","pleasant",
                "tranquility","serene","celebration","enjoying","vivid","laughing"],
    "sad":     ["sad","crying","sorrow","muted","obscured","distorted","blurred",
                "melancholy","somber","gloomy","dark","gray","arid","dry",
                "sparse","lonely","desolate","obscure","pixelated","neutral","tear"],
    "angry":   ["angry","mad","rage","fiery","intense","sharp","jagged","red",
                "orange","vivid","dynamic","movement","energy","stormy",
                "aggressive","bold","clashing","chaos","flaming","frown","shouting"],
    "neutral": ["neutral","expressionless","calm","flat","blank","plain",
                "standard","simple","minimalist","monochrome","beige","gray",
                "balanced","still","undisturbed","serene"],
}
WEATHER_KEYWORD_MAP = {
    "rain":    ["rain","rainy","storm","wet","drizzle","pouring","thunderstorm",
                "shower","droplets","puddle","soaked","gray","dark","water","splash"],
    "shine":   ["sun","sunny","shine","bright","clear","blue sky","light",
                "rays","beam","glare","radiant","warm","summer","illuminated"],
    "cloudy":  ["cloud","cloudy","overcast","gray","gloom","fog","mist",
                "haze","white","fluffy","cumulus","sky","covered","dim"],
    "sunrise": ["sunrise","dawn","morning","breaking","sunup","orange","glow",
                "horizon","early","start","rising","pink","golden","dusk"],
}
FACE_KEYWORDS  = ["face","person","human","expression","eyes","mouth","facial",
                  "features","head","portrait","individual","skin","nose",
                  "appearance","man","woman","child","girl","boy"]
SCENE_KEYWORDS = ["outdoor","sky","landscape","nature","view","scene",
                  "environment","outside","horizon","mountain","grass","tree",
                  "world","ground","field","water","sea","ocean","weather",
                  "atmosphere","day","night"]

def _kw_map_for(concept, domain):
    if domain == "weather":
        return WEATHER_KEYWORD_MAP, SCENE_KEYWORDS
    return EMOTION_KEYWORD_MAP, FACE_KEYWORDS

def _match(text, keywords):
    t = text.lower()
    return any(kw in t for kw in keywords)

# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------
def extract_from_file(fpath, domain):
    """
    Returns a list of dicts, one per unique (concept, model_name) combination,
    with per-entry baseline scores aggregated.
    """
    model_name = os.path.basename(fpath).replace(".json","")
    # strip known prefixes
    for pfx in ("exp_blank_hallucination_", "exp_emotion_transfer_"):
        model_name = model_name.replace(pfx, "")

    with open(fpath) as f:
        entries = json.load(f)

    # Determine experiment type from first entry
    if not entries:
        return []
    first = entries[0]
    is_hallucination = "target_emotion" in first

    kw_map, gen_kw = _kw_map_for(first.get("target_emotion") or first.get("target",""), domain)

    records = []
    for entry in entries:
        if is_hallucination:
            concept = entry.get("target_emotion","")
            pair    = None
        else:
            concept = entry.get("target","")
            pair    = f"{entry.get('source','')} -> {entry.get('target','')}"

        kw_map, gen_kw = _kw_map_for(concept, domain)
        keywords = kw_map.get(concept, [concept])

        results = entry.get("results", [])
        if not results:
            continue

        baseline_spec  = [_match(r["baseline"], keywords) for r in results]
        baseline_gen   = [_match(r["baseline"], gen_kw)   for r in results]
        steered_spec   = [_match(r["steered"],  keywords) for r in results]
        steered_gen    = [_match(r["steered"],  gen_kw)   for r in results]

        n = len(results)
        records.append({
            "domain":           domain,
            "model":            model_name,
            "experiment":       "hallucination" if is_hallucination else "transfer",
            "concept":          concept,
            "pair":             pair,
            "layer":            entry.get("layer"),
            "component":        entry.get("component"),
            "alpha":            entry.get("alpha"),
            "vector_norm":      entry.get("vector_norm"),
            "n_responses":      n,
            "baseline_specific_pct": round(sum(baseline_spec)/n*100, 1),
            "baseline_general_pct":  round(sum(baseline_gen) /n*100, 1),
            "steered_specific_pct":  round(sum(steered_spec) /n*100, 1),
            "steered_general_pct":   round(sum(steered_gen)  /n*100, 1),
            "lift_specific_pct":     round((sum(steered_spec)-sum(baseline_spec))/n*100, 1),
        })
    return records

def load_all():
    all_records = []
    for domain, rdir in RESULT_DIRS.items():
        for fpath in sorted(glob.glob(os.path.join(rdir, "*.json"))):
            recs = extract_from_file(fpath, domain)
            all_records.extend(recs)
            print(f"  Loaded {len(recs):>4} entries from {fpath}")
    return all_records

# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _agg(records, keys, value_keys):
    """Group records by `keys`, compute mean of `value_keys`."""
    groups = defaultdict(list)
    for r in records:
        gk = tuple(r[k] for k in keys)
        groups[gk].append(r)

    rows = []
    for gk, grp in sorted(groups.items()):
        row = dict(zip(keys, gk))
        for vk in value_keys:
            vals = [r[vk] for r in grp]
            row[vk] = round(sum(vals)/len(vals), 1)
        row["n_entries"] = len(grp)
        rows.append(row)
    return rows

# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------
def _fmt(v):
    return f"{v:6.1f}%" if isinstance(v, float) else str(v)

def print_table(title, rows, cols):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    header = "  ".join(f"{c:<22}" for c in cols)
    print(header)
    print("-"*len(header))
    for r in rows:
        print("  ".join(f"{_fmt(r.get(c,'')):<22}" for c in cols))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading result files …")
    all_recs = load_all()
    print(f"Total entries loaded: {len(all_recs)}\n")

    VKEYS = ["baseline_specific_pct","steered_specific_pct","lift_specific_pct",
             "baseline_general_pct","steered_general_pct"]

    # ------------------------------------------------------------------ #
    # TABLE 1 — Hallucination: mean baseline vs steered (all layers/alpha)
    # ------------------------------------------------------------------ #
    hall = [r for r in all_recs if r["experiment"] == "hallucination"]
    agg1 = _agg(hall, ["domain","concept","model"], VKEYS)
    print_table(
        "TABLE 1 — Hallucination: mean across all layers & alphas",
        agg1,
        ["domain","concept","model",
         "baseline_specific_pct","steered_specific_pct","lift_specific_pct","n_entries"]
    )

    # ------------------------------------------------------------------ #
    # TABLE 2 — Hallucination: best steered entry per concept/model
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print("  TABLE 2 — Hallucination: best steered specific% per concept/model")
    print(f"{'='*70}")
    groups2 = defaultdict(list)
    for r in hall:
        groups2[(r["domain"], r["concept"], r["model"])].append(r)

    header2 = f"{'domain':<10}{'concept':<10}{'model':<20}{'best_steered%':>14}  {'at layer':>8}  {'comp':<8}  {'alpha':>6}  {'baseline%':>10}  {'lift%':>8}"
    print(header2)
    print("-"*len(header2))
    for (dom, conc, mdl), grp in sorted(groups2.items()):
        best = max(grp, key=lambda x: x["steered_specific_pct"])
        print(f"{dom:<10}{conc:<10}{mdl:<20}{best['steered_specific_pct']:>13.1f}%  "
              f"{best['layer']:>8}  {best['component']:<8}  {best['alpha']:>6}  "
              f"{best['baseline_specific_pct']:>9.1f}%  {best['lift_specific_pct']:>7.1f}%")

    # ------------------------------------------------------------------ #
    # TABLE 3 — Hallucination: alpha sweep (mean specific match)
    # ------------------------------------------------------------------ #
    agg3 = _agg(hall, ["domain","concept","model","alpha"], VKEYS)
    print_table(
        "TABLE 3 — Hallucination alpha sweep (mean specific match)",
        agg3,
        ["domain","concept","model","alpha",
         "baseline_specific_pct","steered_specific_pct","lift_specific_pct"]
    )

    # ------------------------------------------------------------------ #
    # TABLE 4 — Hallucination: best layer per concept/model
    #           (layer with highest mean steered_specific across all alphas & components)
    # ------------------------------------------------------------------ #
    agg4 = _agg(hall, ["domain","concept","model","layer"], VKEYS)
    print(f"\n{'='*70}")
    print("  TABLE 4 — Best layer per concept/model (hallucination, avg across alphas & components)")
    print(f"{'='*70}")
    by_key4 = defaultdict(list)
    for r in agg4:
        by_key4[(r["domain"], r["concept"], r["model"])].append(r)
    header4 = f"{'domain':<10}{'concept':<10}{'model':<20}{'best_layer':>10}  {'steered%':>9}  {'baseline%':>10}  {'lift%':>8}"
    print(header4)
    print("-"*len(header4))
    for (dom, conc, mdl), rows in sorted(by_key4.items()):
        best = max(rows, key=lambda x: x["steered_specific_pct"])
        print(f"{dom:<10}{conc:<10}{mdl:<20}{best['layer']:>10}  "
              f"{best['steered_specific_pct']:>8.1f}%  "
              f"{best['baseline_specific_pct']:>9.1f}%  "
              f"{best['lift_specific_pct']:>7.1f}%")

    # ------------------------------------------------------------------ #
    # TABLE 5 — Transfer: best entry per pair/model
    # ------------------------------------------------------------------ #
    trans = [r for r in all_recs if r["experiment"] == "transfer"]
    print(f"\n{'='*70}")
    print("  TABLE 5 — Transfer: best steered specific% per pair/model")
    print(f"{'='*70}")
    groups5 = defaultdict(list)
    for r in trans:
        groups5[(r["domain"], r["pair"], r["model"])].append(r)
    header5 = f"{'domain':<10}{'pair':<22}{'model':<20}{'best_steered%':>14}  {'at layer':>8}  {'comp':<8}  {'baseline%':>10}  {'lift%':>8}"
    print(header5)
    print("-"*len(header5))
    for (dom, pair, mdl), grp in sorted(groups5.items()):
        best = max(grp, key=lambda x: x["steered_specific_pct"])
        print(f"{dom:<10}{str(pair):<22}{mdl:<20}{best['steered_specific_pct']:>13.1f}%  "
              f"{best['layer']:>8}  {best['component']:<8}  "
              f"{best['baseline_specific_pct']:>9.1f}%  {best['lift_specific_pct']:>7.1f}%")

    # ------------------------------------------------------------------ #
    # Save machine-readable output
    # ------------------------------------------------------------------ #
    output = {
        "meta": {
            "description": "Extracted baseline and steered match rates from all result JSONs",
            "keyword_source": "visualize.py (original, unmodified)",
            "metric_definitions": {
                "baseline_specific_pct": "% of baseline (unsteered) responses matching concept keywords",
                "steered_specific_pct":  "% of steered responses matching concept keywords",
                "lift_specific_pct":     "steered_specific_pct - baseline_specific_pct",
                "baseline_general_pct":  "% of baseline responses matching general domain keywords (face/scene)",
                "steered_general_pct":   "% of steered responses matching general domain keywords",
            },
        },
        "all_entries": all_recs,
        "table1_hallucination_mean":       _agg(hall,  ["domain","concept","model"], VKEYS),
        "table3_hallucination_alpha_sweep": _agg(hall, ["domain","concept","model","alpha"], VKEYS),
        "table4_best_layer":                _agg(hall, ["domain","concept","model","layer"], VKEYS),
        "table5_transfer_mean":             _agg(trans, ["domain","pair","model"], VKEYS),
    }

    out_path = "baselines_table.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Saved] {out_path}  ({len(all_recs)} records)")

if __name__ == "__main__":
    main()
