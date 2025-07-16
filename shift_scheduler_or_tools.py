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

# --- 2. PREPROCESSING: Determine shift demand ---
def plan_shifts(names, common_posts, c_only_posts, merged_groups):
    total_people = len(names)
    used_people = len(c_only_posts)  # for C-only posts
    post_plan = {}
    shift_demand = defaultdict(int)
    merged_map = {}
    merged_set = set()

    for group in merged_groups:
        parts = [p.strip() for p in group.split(",") if p.strip()]
        if len(parts) > 1:
            top = parts[0]
            for p in parts[1:]:
                merged_map[p] = top
                merged_set.add(p)

    post_list = []
    for p in common_posts:
        if p in merged_set:
            continue
        post_list.append(p)

    # Fill all C-only posts with C shift
    for p in c_only_posts:
        post_plan[p] = ["C"]
        shift_demand["C"] += 1

    # Try 8-hour shifts for common posts, else 12-hr, else unfilled
    for post in post_list[::-1]:  # bottom priority first
        if used_people + 3 <= total_people:
            post_plan[post] = ["A", "B", "C"]
            shift_demand["A"] += 1
            shift_demand["B"] += 1
            shift_demand["C"] += 1
            used_people += 3
        elif used_people + 2 <= total_people:
            post_plan[post] = ["Day12", "Night12"]
            shift_demand["Day12"] += 1
            shift_demand["Night12"] += 1
            used_people += 2
        else:
            post_plan[post] = []  # left unfilled

    return shift_demand, post_plan

# --- 3. SHIFT ASSIGNMENT ---
def assign_shifts_with_ortools(names, shift_demand, prev_shift_map, weekly_counts):
    random.shuffle(names)  # 🟢 Shuffle before model definition
    model = cp_model.CpModel()
    shifts = ["A", "B", "C", "Day12", "Night12", "Off"]
    shift_vars = {(p, s): model.NewBoolVar(f"{p}_{s}") for p in names for s in shifts}

    for p in names:
        model.AddExactlyOne(shift_vars[p, s] for s in shifts)

    for p in names:
        prev = prev_shift_map.get(p, "")
        if prev in ["C", "Night12"]:
            model.Add(shift_vars[p, "A"] == 0)
            model.Add(shift_vars[p, "Day12"] == 0)

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

    for s, count in shift_demand.items():
        model.Add(sum(shift_vars[p, s] for p in names) == count)

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
        result = {}
        for p in names:
            for s in shifts:
                if solver.Value(shift_vars[p, s]):
                    result[p] = s
        return result
    else:
        return None

# --- 4. FINAL OUTPUT ---
if st.button("Generate Schedule") and names:
    shift_demand, post_plan = plan_shifts(names, common_posts, c_only_posts, merge_input)
    shift_assignment = assign_shifts_with_ortools(names, shift_demand, prev_shift_map, weekly_counts)

    if shift_assignment:
        st.success("Schedule generated successfully!")
        doc = Document()
        doc.add_heading("Shift Schedule by Duty Post", 0)

        shifts_order = ["A", "B", "C", "Day12", "Night12"]
        post_shift_map = defaultdict(lambda: {s: "" for s in shifts_order})

        shift_to_people = defaultdict(list)
        for p, s in shift_assignment.items():
            shift_to_people[s].append(p)

        for s in shift_to_people:
            random.shuffle(shift_to_people[s])  # 🟢 Randomize assignment per shift

        for post, shifts in post_plan.items():
            for s in shifts:
                if shift_to_people[s]:
                    person = shift_to_people[s].pop()
                    post_shift_map[post][s] = person

        for post in post_plan:
            doc.add_paragraph(post, style='Heading 2')
            for s in shifts_order:
                if post_shift_map[post][s]:
                    doc.add_paragraph(f"Shift {s}: {post_shift_map[post][s]}", style='List Bullet')

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        st.download_button("Download Schedule", buffer, file_name="shift_schedule_by_post.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    else:
        st.error("No feasible schedule found. Try adjusting constraints or increasing staff.")
