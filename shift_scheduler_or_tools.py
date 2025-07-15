# --- Shift Scheduler with OR-Tools, Word Export, Excel Input, Weekly Constraints, Merging, and 12-Hour Fallback ---

import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model
from docx import Document
from collections import defaultdict
import datetime
import io

st.title("Smart Shift Scheduler with Constraints and Fallback Logic")

# --- 1. INPUT FROM USER ---

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

# --- NEW: Upload Excel for Weekly History ---
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

# --- 2. BUILD MODEL ---

if st.button("Generate Schedule") and names:
    model = cp_model.CpModel()
    shifts = ["A", "B", "C", "Day12", "Night12", "Off"]
    vars = {(p, s): model.NewBoolVar(f"{p}_{s}") for p in names for s in shifts}

    for p in names:
        model.AddExactlyOne(vars[p, s] for s in shifts)

    for p in names:
        prev = prev_shift_map.get(p, "")
        if prev in ["C","Night12"]:
            model.Add(vars[p, "A"] == 0)
        if prev in ["C", "Night12"]:
            model.Add(vars[p, "Day12"] == 0)

    if weekly_counts:
        n = len(names)
        max_4_allowed = max(1, int(n * 0.10))
        exceeds_night = []

        for p in names:
            prior_c = weekly_counts[p]["C"]
            prior_n12 = weekly_counts[p]["Night12"]
            prior_d12 = weekly_counts[p]["Day12"]

            total_night = prior_c + prior_n12
            total_12hr = prior_d12 + prior_n12

            flag = model.NewBoolVar(f"exceeds_night_{p}")
            model.Add(total_night + vars[p, "C"] + vars[p, "Night12"] <= 4).OnlyEnforceIf(flag.Not())
            model.Add(flag == 1).OnlyEnforceIf(
                total_night + vars[p, "C"] + vars[p, "Night12"] > 3
            )
            exceeds_night.append(flag)

            model.Add(total_12hr + vars[p, "Day12"] + vars[p, "Night12"] <= 3)

        model.Add(sum(exceeds_night) <= max_4_allowed)

    assigned_vars = []
    used = set()

    # --- 1. Fill C-only duty posts with C shift ---
    for post in c_only_posts:
        for p in names:
            if p not in used:
                model.Add(vars[p, "C"] == 1)
                assigned_vars.append((p, "C", post))
                used.add(p)
                break

    # --- 2. Try to assign 8-hour shifts for common posts ---
    post_needs = defaultdict(list)
    for post in common_posts:
        for shift in ["A", "B", "C"]:
            post_needs[post].append(shift)

    # --- 3. If not enough people, try merging ---
    total_needed = sum(len(v) for v in post_needs.values())
    people_left = len(names) - len(used)

    merged_posts = []
    merged_set = set()
    if people_left < total_needed:
        for group in merged_groups:
            # Replace original posts with merged post (keep highest priority)
            top_post = None
            for post in common_posts:
                if post in group:
                    top_post = post
                    break
            if top_post:
                merged_posts.append(top_post)
                for post in group:
                    if post != top_post and post in post_needs:
                        del post_needs[post]
                        merged_set.add(post)
                post_needs[top_post] = ["A", "B", "C"]

    # --- 4. If still not enough, convert lowest priority posts into 12-hour shifts (pair: Day12 + Night12) ---
    flat_post_list = list(post_needs.items())[::-1]  # bottom priority first
    for post, shifts_needed in flat_post_list:
        if people_left >= len(shifts_needed):
            continue  # no need to convert
        if people_left >= 2:
            post_needs[post] = ["Day12", "Night12"]
            people_left -= 2
        else:
            del post_needs[post]  # mark as unfilled

    # --- 5. Assign remaining posts ---
    for post, shift_list in post_needs.items():
        for shift in shift_list:
            for p in names:
                if p not in used:
                    model.Add(vars[p, shift] == 1)
                    assigned_vars.append((p, shift, post))
                    used.add(p)
                    break

    # --- 6. Assign Off to unused people ---
    for p in names:
        if p not in used:
            model.Add(vars[p, "Off"] == 1)

    # --- 7. Solve ---
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

        vacant_posts = []

        for p in names:
            for s in shifts:
                if solver.Value(vars[p, s]) == 1:
                    post = next((x[2] for x in assigned_vars if x[0] == p), "N/A")
                    row_cells = table.add_row().cells
                    row_cells[0].text = p
                    row_cells[1].text = s
                    row_cells[2].text = post

        for post in common_posts:
            if post not in post_needs:
                vacant_posts.append(post)

        if vacant_posts:
            st.warning(f"The following posts could not be filled due to insufficient people: {', '.join(vacant_posts)}")

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
        st.error("Could not find a feasible schedule under the given constraints.")
