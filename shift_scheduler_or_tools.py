



# --- Shift Scheduler with OR-Tools, Word Export, Excel Input, Weekly Constraints, Merging, and 12-Hour Fallback ---


import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model
from docx import Document
from collections import defaultdict
import datetime
import io

st.title("Smart Shift Scheduler with Constraints and Fallback Logic")

# --- INPUT SECTION ---
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
    st.warning("No previous shift data provided. Some shift constraints will be skipped.")

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
else:
    st.warning("Weekly shift history not provided. Weekly limits disabled.")

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
    vars = {(p, s): model.NewBoolVar(f"{p}_{s}") for p in names for s in shifts}

    for p in names:
        model.AddExactlyOne(vars[p, s] for s in shifts)
        prev = prev_shift_map.get(p, "")
        if prev in ["C", "Night12"]:
            model.Add(vars[p, "A"] == 0)
            model.Add(vars[p, "Day12"] == 0)

    if weekly_counts:
        n = len(names)
        max_4_allowed = max(1, int(n * 0.10))
        exceeds_night = []
        for p in names:
            c_prev, n12_prev, d12_prev = weekly_counts[p].values()
            total_night = model.NewIntVar(0, 10, f"total_night_{p}")
            total_12hr = model.NewIntVar(0, 10, f"total_12hr_{p}")
            model.Add(total_night == c_prev + n12_prev + vars[p,"C"] + vars[p,"Night12"])
            model.Add(total_12hr == d12_prev + n12_prev + vars[p,"Day12"] + vars[p,"Night12"])
            model.Add(total_12hr <= 3)
            flag = model.NewBoolVar(f"exceeds_night_{p}")
            cond = model.NewBoolVar(f"night_gt3_{p}")
            model.Add(total_night > 3).OnlyEnforceIf(cond)
            model.Add(total_night <= 3).OnlyEnforceIf(cond.Not())
            model.Add(flag == 1).OnlyEnforceIf(cond)
            model.Add(flag == 0).OnlyEnforceIf(cond.Not())
            exceeds_night.append(flag)
        model.Add(sum(exceeds_night) <= max_4_allowed)

    used = set()
    assigned_vars = []
    post_needs = defaultdict(list)

    # --- STEP 1: Fill all C-only posts with C shift ---
    for post in c_only_posts:
        for p in names:
            if p not in used:
                model.Add(vars[p,"C"] == 1)
                assigned_vars.append((p,"C",post))
                used.add(p)
                break

    # --- STEP 2: Attempt 8-hour shifts for common posts ---
    for post in common_posts:
        post_needs[post] = ["A", "B", "C"]

    total_need = sum(len(v) for v in post_needs.values())
    people_left = len(names) - len(used)

    # --- STEP 3: Try merging ---
    if people_left < total_need:
        for group in merged_groups:
            top = next((p for p in common_posts if p in group), None)
            if top:
                for sub in group:
                    if sub != top and sub in post_needs:
                        del post_needs[sub]
                post_needs[top] = ["A", "B", "C"]

    # --- STEP 4: Convert bottom posts to 12-hr if still short ---
    flat = list(post_needs.items())[::-1]
    for post, shifts_list in flat:
        people_left = len(names) - len(used)
        if people_left >= len(shifts_list):
            continue
        elif people_left >= 2:
            post_needs[post] = ["Day12", "Night12"]
        else:
            del post_needs[post]

    # --- STEP 5: Assign shifts to people ---
    for post, shifts in post_needs.items():
        for s in shifts:
            for p in names:
                if p not in used:
                    model.Add(vars[p, s] == 1)
                    assigned_vars.append((p, s, post))
                    used.add(p)
                    break

    # --- STEP 6: Weekly off ---
    for p in names:
        if p not in used:
            model.Add(vars[p, "Off"] == 1)

    # --- STEP 7: Solve ---
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
        st.success("Schedule generated successfully!")
        doc = Document()
        doc.add_heading("Shift Schedule", 0)
        table = doc.add_table(rows=1, cols=3)
        table.style = 'Light Grid'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Name'
        hdr_cells[1].text = 'Shift'
        hdr_cells[2].text = 'Duty Post'

        for p in names:
            for s in shifts:
                if solver.Value(vars[p,s]) == 1:
                    duty = next((x[2] for x in assigned_vars if x[0]==p), "N/A")
                    row = table.add_row().cells
                    row[0].text = p
                    row[1].text = s
                    row[2].text = duty

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        st.download_button(
            label="Download Schedule as Word",
            data=buffer,
            file_name="shift_schedule.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    else:
        st.error("Could not generate feasible schedule with given inputs.")


