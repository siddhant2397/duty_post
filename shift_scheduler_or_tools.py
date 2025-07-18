import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model
from docx import Document
from collections import defaultdict, OrderedDict
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

# --- 2. IMPROVED SHIFT PLANNING ---
def plan_shifts(names, common_posts, c_only_posts, merged_groups):
    total_people = len(names)
    shift_demand = defaultdict(int)
    merged_set = set()
    merged_map = {}
    post_plan = OrderedDict()

    for group in merged_groups:
        parts = [p.strip() for p in group.split(",") if p.strip()]
        if len(parts) > 1:
            top = parts[0]
            for p in parts[1:]:
                merged_map[p] = top
                merged_set.add(p)

    used_people = 0
    for post in c_only_posts:
        post_plan[post] = ["C"]
        shift_demand["C"] += 1
        used_people += 1

    reduced_common = []
    seen = set()
    for post in common_posts:
        if post in merged_set:
            continue
        if post in seen:
            continue
        reduced_common.append(post)
        seen.add(post)

    # Initially assign all as 8-hour
    post_status = OrderedDict((p, ["A", "B", "C"]) for p in reduced_common)

    def required_people(post_dict):
        return sum(3 if v == ["A", "B", "C"] else 2 for v in post_dict.values())

    posts = list(post_status.items())
    index = len(posts) - 1

    while True:
        total_required = used_people + required_people(post_status)
        if total_required <= total_people:
            break
        # Convert bottom post to 12-hour
        while index >= 0:
            post, shifts = posts[index]
            if post_status[post] == ["A", "B", "C"]:
                post_status[post] = ["Day12", "Night12"]
                break
            index -= 1
        else:
            # All converted to 12-hour, now start removing posts
            while True:
                post, shifts = posts.pop()
                post_status.pop(post)
                total_required = used_people + required_people(post_status)
                if total_required <= total_people:
                    break

    for post, shifts in post_status.items():
        post_plan[post] = shifts
        for s in shifts:
            shift_demand[s] += 1

    return shift_demand, post_plan

# --- 3. OR-TOOLS SHIFT ASSIGNMENT ---
def assign_shifts_with_ortools(names, shift_demand, prev_shift_map, weekly_counts):
    random.shuffle(names)
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

# --- 4. OUTPUT ---
if st.button("Generate Schedule") and names:
    shift_demand, post_plan = plan_shifts(names, common_posts, c_only_posts, merge_input)
    shift_assignment = assign_shifts_with_ortools(names, shift_demand, prev_shift_map, weekly_counts)

    if shift_assignment:
        st.success("Schedule generated successfully!")
        doc = Document()
        doc.add_heading("Shift Schedule by Duty Post", 0)

        shifts_order = ["A", "B", "C", "Day12", "Night12"]
        post_shift_map = OrderedDict()
        vacant_posts = []

        shift_to_people = defaultdict(list)
        for p, s in shift_assignment.items():
            shift_to_people[s].append(p)

        for s in shift_to_people:
            random.shuffle(shift_to_people[s])

        for post, shifts in post_plan.items():
            post_shift_map[post] = {}
            if not shifts:
                vacant_posts.append(post)
            for s in shifts:
                if shift_to_people[s]:
                    person = shift_to_people[s].pop()
                    post_shift_map[post][s] = person

        for post in post_plan:
            doc.add_paragraph(post, style='Heading 2')
            for s in shifts_order:
                if post_shift_map[post].get(s):
                    doc.add_paragraph(f"Shift {s}: {post_shift_map[post][s]}", style='List Bullet')

        if vacant_posts:
            doc.add_paragraph("\nUnfilled Posts (due to insufficient personnel):", style='Heading 2')
            for vp in vacant_posts:
                doc.add_paragraph(vp, style='List Bullet')

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        st.download_button("Download Schedule", buffer, file_name="shift_schedule_by_post.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    else:
        st.error("No feasible schedule found. Try adjusting constraints or increasing staff.")
