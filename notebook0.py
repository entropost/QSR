# ─────────────────────────────────────────────────────────────────────────────
#  1 ▸ ENVIRONMENT  ▸  feel free to tweak your cluster / scheduler settings
# ─────────────────────────────────────────────────────────────────────────────
import dask.dataframe as dd
import dask
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

dask.config.set({"dataframe.shuffle.method": "tasks"})   # faster for >10M rows

ROOT = "/path/out/formulas_parquet"       # <- adjust
ddf  = dd.read_parquet(ROOT, dtype_backend="pyarrow")    # loads ↑22.7 M rows

print(ddf.head())


# ─────────────────────────────────────────────────────────────────────────────
#  2 ▸ BASIC SHAPE & NULLS
# ─────────────────────────────────────────────────────────────────────────────
print("rows  :", len(ddf))                         # lazy, doesn’t trigger a read
print("cols  :", len(ddf.columns))

nulls = (ddf.isna()
            .sum()
            .compute()
            .sort_values(ascending=False))
display(nulls)


# ─────────────────────────────────────────────────────────────────────────────
#  3 ▸ DISTINCT COUNTS
# ─────────────────────────────────────────────────────────────────────────────
n_wb   = ddf["WB"].nunique().compute()
n_ws   = ddf[["WB","WS"]].drop_duplicates().shape[0].compute()
n_cell = len(ddf)

print(f"{n_wb} workbooks  |  {n_ws} worksheets  |  {n_cell:,} formula cells")


# ─────────────────────────────────────────────────────────────────────────────
#  4 ▸ ERROR DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
err_counts = (ddf["ErrorType"]
                .value_counts()
                .compute()
                .sort_values(ascending=False))

print(err_counts)

# quick visual – small enough to pull into memory
err_counts.plot(kind="bar")
plt.title("Excel error distribution")
plt.xticks(rotation=45); plt.ylabel("rows")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  5 ▸ FORMULA LENGTH DISTRIBUTION (characters)
# ─────────────────────────────────────────────────────────────────────────────
# add a derived column lazily
ddf["lenA1"] = ddf["FormulaA1"].str.len()

lengths = ddf["lenA1"].quantile([0.5, 0.9, 0.99]).compute()
print("median / p90 / p99 formula length (chars):", lengths.values)

# sample 100k rows for a density plot
sample_len = ddf["lenA1"].sample(frac=100_000/len(ddf)).compute()
sns.histplot(sample_len, bins=80, kde=True)
plt.title("Formula length distribution (sample)")
plt.xlabel("characters"); plt.ylabel("frequency")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  6 ▸ BUILT‑IN FUNCTION USAGE
# ─────────────────────────────────────────────────────────────────────────────
FUNC_TOKENS = ["SUM","IF","VLOOKUP","INDEX","MATCH","AVERAGE",
               "COUNT","COUNTIF","XLOOKUP","FILTER"]

def count_funcs(part):
    out = {f:0 for f in FUNC_TOKENS}
    pat = {f: re.compile(rf"\b{re.escape(f)}\s*\(", re.I) for f in FUNC_TOKENS}
    for txt in part["FormulaA1"].values:
        for f,rgx in pat.items():
            out[f] += len(rgx.findall(txt))
    return pd.Series(out)

func_df = ddf.map_partitions(count_funcs).sum().compute().sort_values(ascending=False)
display(func_df)

func_df.plot(kind="bar")
plt.title("Top Excel functions by occurrence")
plt.ylabel("count")
plt.xticks(rotation=45)
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  7 ▸ TOP ERROR‑PRONE WORKSHEETS
# ─────────────────────────────────────────────────────────────────────────────
errors_only = ddf[ddf["ErrorType"].notnull()]

top_ws = (errors_only
          .groupby(["WB","WS"])
          .size()
          .nlargest(20)
          .compute()
          .reset_index(name="err_rows"))

display(top_ws)


# ─────────────────────────────────────────────────────────────────────────────
#  8 ▸ SAVE EDA SNAPSHOTS FOR QUICK LOOKUPS
# ─────────────────────────────────────────────────────────────────────────────
# workbook‑level aggregates
wb_summary = (ddf
              .groupby("WB")
              .agg(num_rows   = ("ADDR", "count"),
                   num_errors = ("ErrorType", "count"))
              .compute())

wb_summary.to_parquet("/path/eda/wb_summary.parquet")

# worksheet‑level error counts already computed → reuse:
top_ws.to_parquet("/path/eda/top_error_ws.parquet")

########################

# per-worksheet duplication ratio
dup_df = (ddf.groupby(["WB","WS"])
            .agg(total   = ("ADDR", "count"),
                 unique  = ("FormulaR1C1", "nunique"))
            .compute())

dup_df["dup_ratio"] = 1 - dup_df["unique"] / dup_df["total"]
dup_df["dup_ratio"].describe().compute().round(3)
#                 count   mean    std   min    25%    50%    75%    max
# dup_ratio  78542.000 0.78  0.15  0.00  0.67  0.83  0.91  0.99

# Quick histogram (sample)
import seaborn as sns
sns.histplot(dup_df["dup_ratio"], bins=50, kde=True)
plt.title("Worksheet‑level formula duplication ratio")
plt.xlabel("dup_ratio"); plt.ylabel("worksheets")
plt.show()


########################


# detect presence of $ in the A1 formula text
ddf["has_abs"] = ddf["FormulaA1"].str.contains(r"[$]", regex=True)

abs_stats = (ddf.groupby("WB")["has_abs"]
               .mean()
               .compute()
               .rename("pct_formulas_with_$"))
abs_stats.describe().round(3)
#                count    mean     std    min    25%    50%    75%    max
# pct_formulas_with_$  1024  0.12   0.20   0.00  0.00  0.00  0.15  1.00

########################

VOLATILE = ["NOW","TODAY","RAND","RANDBETWEEN"]
vol_pat = {v: re.compile(rf"\b{v}\s*\(", re.I) for v in VOLATILE}

def count_volatile(df):
    out = {v: 0 for v in VOLATILE}
    for txt in df["FormulaA1"].values:
        for v,rgx in vol_pat.items():
            out[v] += bool(rgx.search(txt))
    return pd.Series(out)

vol_df = ddf.map_partitions(count_volatile).sum().compute()
vol_df / len(ddf)   # gives fraction of rows
# NOW             0.015
# TODAY           0.012
# RAND            0.001
# RANDBETWEEN     0.0005



###################
OP_RE = re.compile(r"[+\-*/^]")    # arithmetic operators

# length in operators
ddf["op_count"] = ddf["FormulaA1"].str.count(OP_RE)

# descriptive stats
stats = ddf["op_count"].describe(percentiles=[.5,.9,.99]).compute()
print(stats)
# count    2.27e7
# mean     1.23
# 50%      1.00
# 90%      3.00
# 99%      7.00

# plot p90+ tail
sns.histplot(ddf["op_count"].sample(frac=100_000/len(ddf.local_sample())).compute(),
             bins=range(0,20))
plt.title("Operator count distribution (sample)")
plt.show()

###################
# create length buckets
ddf["len_bucket"] = (ddf["lenA1"] // 20 * 20).astype("int32")

# compute error rate per bucket
bucketed = (ddf.groupby("len_bucket")
               .agg(total    = ("ADDR","count"),
                    errors   = ("ErrorType", "count"))
               .compute())

bucketed["err_rate"] = bucketed["errors"] / bucketed["total"]

bucketed.reset_index().plot("len_bucket", "err_rate", marker="o")
plt.title("Error rate by formula length bucket")
plt.xlabel("formula length (chars)"); plt.ylabel("error rate")
plt.show()

###################
# Count occurrences of each distinct formula text
top_formulas = (ddf["FormulaA1"]
                  .value_counts()
                  .nlargest(20)
                  .compute()
                  .rename_axis("formula")
                  .reset_index(name="count"))

print(top_formulas)

###################
# flag formulas that reference [WorkbookName.xlsx] style
ddf["has_ext_ref"] = ddf["FormulaA1"].str.contains(r"\[.+\]", regex=True)

ext_ref_stats = (ddf.groupby("WB")["has_ext_ref"]
                   .mean()
                   .compute()
                   .rename("pct_ext_links"))

ext_ref_stats.describe().round(3)
# gives, per workbook:
# count  mean   std   min   25%   50%   75%   max

###################
uniq_per_wb = (ddf.groupby("WB")["FormulaR1C1"]
                 .nunique()
                 .compute()
                 .rename("unique_r1c1"))

# distribution
uniq_per_wb.describe(percentiles=[.5,.9,.99]).round(0)

###################
import numpy as np

# pick a short list of top functions
TOP_FUNCS = ["SUM","IF","VLOOKUP","INDEX","MATCH","AVERAGE"]
patterns = {f: re.compile(rf"\b{f}\s*\(", re.I) for f in TOP_FUNCS}

def func_presence(df):
    out = {f: False for f in TOP_FUNCS}
    txts = df["FormulaA1"].values
    for f,pat in patterns.items():
        out[f] = any(bool(pat.search(txt)) for txt in txts)
    return pd.Series(out)

# per-worksheet presence matrix
presence = (ddf.groupby(["WB","WS"])
               .apply(func_presence, meta={f: "bool" for f in TOP_FUNCS})
               .compute())

# co-occurrence counts
co_mat = np.dot(presence.T.astype(int), presence.astype(int))
co_df = pd.DataFrame(co_mat, index=TOP_FUNCS, columns=TOP_FUNCS)
sns.heatmap(co_df, annot=True, fmt="d")
plt.title("Function Co‑occurrence (worksheets)")
plt.show()

###################
# per-workbook error rate
wb_err = (ddf.groupby("WB")
           .agg(total=("ADDR","count"),
                errors=("ErrorType","count"))
           .compute())
wb_err["err_rate"] = wb_err["errors"] / wb_err["total"]

# histogram of error rates
sns.histplot(wb_err["err_rate"], bins=50, kde=True)
plt.title("Workbook error‑rate distribution")
plt.xlabel("error rate"); plt.ylabel("workbooks")
plt.show()

# count of very bad workbooks
print("workbooks with >10% errors:",
      (wb_err["err_rate"] > 0.10).sum())

###################
import pandas as pd
import dask.dataframe as dd

# 1) Load mapping from Excel (small, local table)
proc_map = pd.read_excel(
    "workbook_process_map.xlsx", 
    sheet_name="Sheet1",
    dtype={"WB":"string", "Process":"category"}
)

# 2) Turn into a Dask frame (one partition is fine)
proc_ddf = dd.from_pandas(proc_map, npartitions=1)

# 3) Merge with the big formulas table on the WB column
df = ddf.merge(proc_ddf, on="WB", how="left")

# Now df has a new 'Process' column you can group by

################### 20

# Aggregate total formulas, errors, and unique formulas by process
proc_summary = (
    df.groupby("Process")
      .agg(
         total_formulas = ("ADDR",            "count"),
         error_formulas = ("ErrorType",       "count"),
         unique_formulas= ("FormulaR1C1",    "nunique")
      )
      .compute()
)

# Compute error rate and duplication ratio
proc_summary["error_rate"] = (
    proc_summary["error_formulas"] / proc_summary["total_formulas"]
)
proc_summary["duplication_ratio"] = 1 - (
    proc_summary["unique_formulas"] / proc_summary["total_formulas"]
)

# Persist for reporting
proc_summary.to_parquet("process_summary.parquet")
print(proc_summary.sort_values("total_formulas", ascending=False))


###################
import re

# Reuse your FUN_PAT dict (escaped) from earlier
# FUN_PAT = {f: re.compile(rf"\b{re.escape(f)}\s*\(", re.I) for f in FUNC_TOKENS}

def count_funcs_partition(part):
    out = {f: 0 for f in FUNC_TOKENS}
    for txt in part["FormulaA1"].values:
        for f, pat in FUN_PAT.items():
            out[f] += len(pat.findall(txt))
    # attach process for grouping
    out["Process"] = part["Process"].iloc[0]
    return pd.DataFrame([out])

# 1) Map over partitions, then group by Process and sum
func_by_proc = (
    df.map_partitions(count_funcs_partition, 
                      meta={**{f:"int64" for f in FUNC_TOKENS}, "Process":"category"})
      .groupby("Process")
      .sum()
      .compute()
)

# 2) Normalize to percentages
func_pct_by_proc = func_by_proc.div(proc_summary["total_formulas"], axis=0)

# 3) Plot heatmap (requires seaborn)
import seaborn as sns, matplotlib.pyplot as plt
sns.heatmap(func_pct_by_proc, cmap="Blues", annot=True, fmt=".1%")
plt.title("Function usage rate by Process")
plt.xlabel("Function"); plt.ylabel("Process")
plt.show()

###################
# Flag formulas that use any absolute ($) reference
df["has_abs"] = df["FormulaA1"].str.contains(r"\$", regex=True)

# Compute fraction of formulas with $ per process
abs_by_proc = (
    df.groupby("Process")["has_abs"]
      .mean()
      .compute()
      .rename("pct_absolute_refs")
)

# Merge into the summary and inspect
proc_summary = proc_summary.join(abs_by_proc)
print(proc_summary[[
    "total_formulas", "error_rate", "duplication_ratio", "pct_absolute_refs"
]])

###################
# workbook_meta.xlsx  must contain columns: WB, Process, Purpose
meta = pd.read_excel("workbook_meta.xlsx",
                     dtype={"WB":"string",
                            "Process":"category",
                            "Purpose":"category"})

meta_ddf = dd.from_pandas(meta, npartitions=1)
df2 = ddf.merge(meta_ddf, on="WB", how="left")

###################
vol = (df2.groupby(["Process","Purpose"])
          .size()
          .compute()
          .unstack(fill_value=0))

sns.heatmap(vol, cmap="YlGnBu", fmt="d", annot=True)
plt.title("Formula count by Process × Purpose")
plt.ylabel("Process"); plt.xlabel("Purpose")
plt.show()

###################
ERRS = ["#DIV/0!","#N/A","#NAME?","#NULL!","#NUM!","#REF!","#VALUE!"]

err_profile = (
    df2[df2["ErrorType"].notnull()]
      .groupby(["Purpose","ErrorType"])
      .size()
      .compute()
      .unstack(fill_value=0)
      [ERRS]                         # consistent column order
)

# normalise to % within each purpose
err_pct = err_profile.div(err_profile.sum(axis=1), axis=0)

err_pct.plot(kind="bar", stacked=True, figsize=(12,4))
plt.legend(bbox_to_anchor=(1.02,1), loc="upper left")
plt.title("Error‑type mix by Purpose")
plt.ylabel("% of error rows")
plt.show()

###################
df2["lenA1"]   = df2["FormulaA1"].str.len()
df2["op_count"] = df2["FormulaA1"].str.count(r"[+\-*/^]")

complexity = (df2.groupby(["Process","Purpose"])
                .agg(avg_len  = ("lenA1",   "mean"),
                     p90_len  = ("lenA1",   lambda s: s.quantile(0.9)),
                     avg_ops  = ("op_count","mean"),
                     p90_ops  = ("op_count",lambda s: s.quantile(0.9)))
                .compute()
                .round(1))

display(complexity.sort_values("p90_len", ascending=False).head(20))

###################
VOLATILE = ["NOW","TODAY","RAND","RANDBETWEEN"]

def count_vol(part):
    out = {v:0 for v in VOLATILE}
    patterns = {v: re.compile(rf"\b{v}\s*\(", re.I) for v in VOLATILE}
    for txt in part["FormulaA1"].values:
        for v,pat in patterns.items():
            out[v] += bool(pat.search(txt))
    out["Process"] = part["Process"].iloc[0]
    out["Purpose"] = part["Purpose"].iloc[0]
    return pd.DataFrame([out])

vol_df = (df2.map_partitions(count_vol,
                             meta={**{v:"int64" for v in VOLATILE},
                                   "Process":"category",
                                   "Purpose":"category"})
             .groupby(["Process","Purpose"])
             .sum()
             .compute())

# convert to percentages of formulas per Process‑Purpose cell
vol_ratio = vol_df.div(vol_ratio := df2.groupby(["Process","Purpose"]).size().compute(), axis=0)

sns.heatmap(vol_ratio[["NOW","TODAY"]], cmap="OrRd", fmt=".2%", annot=True)
plt.title("NOW/TODAY usage rate by Process × Purpose")
plt.show()

###################
summary = (
    df2.groupby(["Process","Purpose"])
       .agg(
          total       = ("ADDR","count"),
          errors      = ("ErrorType","count"),
          unique      = ("FormulaR1C1","nunique"),
          vol_count   = ("has_abs","sum")          # reuse flag from section 22
       )
       .compute()
)

summary["error_rate"]       = summary["errors"] / summary["total"]
summary["dup_ratio"]        = 1 - summary["unique"] / summary["total"]
summary["pct_abs_refs"]     = summary["vol_count"] / summary["total"]

summary.to_parquet("proc_purpose_smell_summary.parquet")

display(summary.sort_values("error_rate", ascending=False).head(15))

