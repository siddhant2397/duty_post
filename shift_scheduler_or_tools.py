import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model
from docx import Document
from collections import defaultdict
import io
import random

st.title("Smart Shift Scheduler (Fair & Constraint-Based with OR-Tools)")

# --- 1. INPUT SECTION ---
st.subheader("Upload Excel File with Names")
excel_file = st.file_uploader("Upload Excel with Names", type=[".xlsx"])
names = []
if excel_file:
    df = pd.read_excel(excel_file)
    names = df.iloc[:, 0].dropna().tolist()

common_posts = [p.strip() for p in st.text_area("Enter COMMON duty posts (priority order, one per line)").split("\n") if p.strip()]
c_only_posts = [p.strip() for p in st.text_area("Enter C-SHIFT-ONLY duty posts (must be filled)").split("\n") if p.strip()]
merge_input = st.text_area("Enter MERGE groups (comma-separated per line)").split("\n")

prev_shift_input = st.text_area("Enter previous shift info (Name: Shift per line)")
prev_shift_map = {}
if prev_shift_input.strip():
    for line in prev_shift_input.strip().split("\n"):
        if ":" in line:
            name, shift = line.split(":")
            prev_shift_map[name.strip()] = shift.strip()
else:
    st.warning("No previous shift data provided. Some constraints will be relaxed.")

st.subheader("Upload Weekly Shift History (Excel)")
hist_file = st.file_uploader("Upload Weekly History Excel", type=[".xlsx"], key="weekly_hist")
weekly_counts = defaultdict(lambda: {"C": 0, "Night12": 0, "Day12": 0})
if hist_file:
    hist_df = pd.read_excel(hist_file)
    for _, row in hist_df.iterrows():
        name = row[0]
        weekly_counts[name] = {
            "C": int(row.get("C", 0)),
            "Night12": int(row.get("Night12", 0)),
            "Day12": int(row.get("Day12", 0))
        }

# --- PARSE MERGE LIST ---
merged_groups = []
for line in merge_input:
    parts = [p.strip() for p in line.split(",") if p.strip()]
    if len(parts) > 1:
        merged_groups.append(parts)

# --- SCHEDULING ---
if st.button("Generate Schedule") and names:
    model = cp_model.CpModel()
    shifts = ["A", "B", "C", "Day12", "Night12", "Off"]
    shift_vars = {(p, s): model.NewBoolVar(f"{p}_{s}") for p in names for s in shifts}
    assign_vars = {}

    all_posts = c_only_posts + common_posts.copy()
    for group in merged_groups:
        top = next((p for p in common_posts if p in group), None)
        if top:
            for sub in group:
                if sub != top and sub in all_posts:
                    all_posts.remove(sub)

    for p in names:
        for post in all_posts:
            for s in ["A", "B", "C", "Day12", "Night12"]:
                assign_vars[p, post, s] = model.NewBoolVar(f"assign_{p}_{post}_{s}")

    for p in names:
        model.AddExactlyOne(shift_vars[p, s] for s in shifts)

    for (p, post, s), var in assign_vars.items():
        model.Add(var <= shift_vars[p, s])

    for post in c_only_posts:
        model.AddExactlyOne(assign_vars[p, post, "C"] for p in names if (p, post, "C") in assign_vars)

    for post in common_posts:
        for s in ["A", "B", "C"]:
            post_key = [assign_vars[p, post, s] for p in names if (p, post, s) in assign_vars]
            if post_key:
                model.Add(sum(post_key) == 1)

    if weekly_counts:
        max_4_allowed = max(1, int(len(names) * 0.10))
        flags = []
        for p in names:
            wc = weekly_counts[p]
            total_night = model.NewIntVar(0, 10, f"tn_{p}")
            total_12hr = model.NewIntVar(0, 10, f"t12_{p}")
            model.Add(total_night == wc["C"] + wc["Night12"] + shift_vars[p,"C"] + shift_vars[p,"Night12"])
            model.Add(total_12hr == wc["Day12"] + wc["Night12"] + shift_vars[p,"Day12"] + shift_vars[p,"Night12"])
            model.Add(total_12hr <= 3)
            flag = model.NewBoolVar(f"flag_{p}")
            gt3 = model.NewBoolVar(f"gt3_{p}")
            model.Add(total_night > 3).OnlyEnforceIf(gt3)
            model.Add(total_night <= 3).OnlyEnforceIf(gt3.Not())
            model.Add(flag == 1).OnlyEnforceIf(gt3)
            model.Add(flag == 0).OnlyEnforceIf(gt3.Not())
            flags.append(flag)
        model.Add(sum(flags) <= max_4_allowed)

    for p in names:
        prev = prev_shift_map.get(p, "")
        if prev in ["C", "Night12"]:
            model.Add(shift_vars[p, "A"] == 0)
            model.Add(shift_vars[p, "Day12"] == 0)

    for p in names:
        assigned_posts = [var for (pp, post, s), var in assign_vars.items() if pp == p]
        model.Add(sum(assigned_posts) <= 1)

    for p in names:
        model.Add(shift_vars[p, "Off"] == 1).OnlyEnforceIf([
            shift_vars[p, "A"].Not(),
            shift_vars[p, "B"].Not(),
            shift_vars[p, "C"].Not(),
            shift_vars[p, "Day12"].Not(),
            shift_vars[p, "Night12"].Not()
        ])

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = random.randint(1, 10000)
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        st.success("Schedule generated successfully!")
        doc = Document()
        doc.add_heading("Shift Schedule by Duty Post", 0)

        shift_order = ["A", "B", "C", "Day12", "Night12"]
        post_shift_map = defaultdict(lambda: {s: "" for s in shift_order})

        for (p, post, s) in assign_vars:
            if solver.Value(assign_vars[p, post, s]) == 1:
                post_shift_map[post][s] = p

        for post in all_posts:
            doc.add_paragraph(post, style='Heading 2')
            for s in shift_order:
                if post_shift_map[post][s]:
                    doc.add_paragraph(f"Shift {s}: {post_shift_map[post][s]}", style='List Bullet')

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        st.download_button("Download Schedule", buffer, file_name="shift_schedule_by_post.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    else:
        st.error("No feasible schedule found. Try adjusting inputs or constraints.")
