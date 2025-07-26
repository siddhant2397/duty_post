import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model
from docx import Document
from collections import defaultdict, OrderedDict
import io
import random

st.title("Smart Shift Scheduler with Technical & Gender Constraints (OR-Tools)")

# --- 1. INPUT SECTION ---
st.subheader("Upload Excel File with Names")
excel_file = st.file_uploader("Upload Excel with columns: Name, Gender (M/F), Technical (Yes/No)", type=[".xlsx"])
names, males, females, tech_males, tech_females = [], [], [], [], []
if excel_file:
    df = pd.read_excel(excel_file)
    for _, row in df.iterrows():
        name, gender, tech = row["Name"], row["Gender"], row.get("Technical", "No")
        names.append(name)
        if gender == "M":
            males.append(name)
            if tech == "Yes":
                tech_males.append(name)
        else:
            females.append(name)
            if tech == "Yes":
                tech_females.append(name)

tech_posts = [p.strip() for p in st.text_area("Enter Technical General Shift Posts (1 per line)").split("\n") if p.strip()]
male_gen_posts = [p.strip() for p in st.text_area("Enter Non-Technical General Shift Posts (Male)").split("\n") if p.strip()]
female_gen_posts = [p.strip() for p in st.text_area("Enter Non-Technical General Shift Posts (Female)").split("\n") if p.strip()]
male_common_posts = [p.strip() for p in st.text_area("Enter Common Posts (Male)").split("\n") if p.strip()]
female_common_posts = [p.strip() for p in st.text_area("Enter Common Posts (Female)").split("\n") if p.strip()]
c_only_posts = [p.strip() for p in st.text_area("Enter C Shift-Only Posts (Male Only)").split("\n") if p.strip()]
merge_input = st.text_area("Enter MERGE groups (comma-separated per line)").split("\n")

prev_shift_input = st.text_area("Enter previous shift info (Name: Shift per line)")
prev_shift_map = {}
if prev_shift_input.strip():
    for line in prev_shift_input.strip().split("\n"):
        if ":" in line:
            name, shift = line.split(":")
            prev_shift_map[name.strip()] = shift.strip()

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

# --- 2. PLAN POSTS ---
def plan_posts(all_posts, merge_input, c_only_posts):
    shift_demand = defaultdict(int)
    post_plan = OrderedDict()
    merged_set = set()
    merged_map = {}

    for group in merge_input:
        parts = [p.strip() for p in group.split(",") if p.strip()]
        if len(parts) > 1:
            top = parts[0]
            for p in parts[1:]:
                merged_map[p] = top
                merged_set.add(p)

    used = 0
    for post in c_only_posts:
        post_plan[post] = ["C"]
        shift_demand["C"] += 1
        used += 1

    common_posts = [p for p in all_posts if p not in merged_set]
    post_status = OrderedDict((p, ["A", "B", "C"]) for p in common_posts)

    def required_people(post_dict):
        return sum(3 if v == ["A", "B", "C"] else 2 for v in post_dict.values())

    posts = list(post_status.items())
    index = len(posts) - 1
    total_people = len(names)

    while True:
        total_required = used + required_people(post_status)
        if total_required <= total_people:
            break
        while index >= 0:
            post, shifts = posts[index]
            if post_status[post] == ["A", "B", "C"]:
                post_status[post] = ["Day12", "Night12"]
                break
            index -= 1
        else:
            while posts:
                post, _ = posts.pop()
                post_status.pop(post)
                total_required = used + required_people(post_status)
                if total_required <= total_people:
                    break

    for post, shifts in post_status.items():
        post_plan[post] = shifts
        for s in shifts:
            shift_demand[s] += 1

    return shift_demand, post_plan

# --- 3. OR-TOOLS SCHEDULER ---
def assign_shifts(names, shift_demand, prev_shift_map, weekly_counts):
    model = cp_model.CpModel()
    shifts = ["A", "B", "C", "Day12", "Night12", "Off"]
    vars = {(p, s): model.NewBoolVar(f"{p}_{s}") for p in names for s in shifts}

    for p in names:
        model.AddExactlyOne(vars[p, s] for s in shifts)
        prev = prev_shift_map.get(p, "")
        if prev in ["C", "Night12"]:
            model.Add(vars[p, "A"] == 0)
            model.Add(vars[p, "Day12"] == 0)

    max_4 = max(1, int(len(names) * 0.10))
    flags = []
    for p in names:
        wc = weekly_counts[p]
        total_night = model.NewIntVar(0, 10, f"tn_{p}")
        total_12 = model.NewIntVar(0, 10, f"t12_{p}")
        model.Add(total_night == wc["C"] + wc["Night12"] + vars[p, "C"] + vars[p, "Night12"])
        model.Add(total_12 == wc["Day12"] + wc["Night12"] + vars[p, "Day12"] + vars[p, "Night12"])
        model.Add(total_12 <= 3)
        flag = model.NewBoolVar(f"flag_{p}")
        gt3 = model.NewBoolVar(f"gt3_{p}")
        model.Add(total_night > 3).OnlyEnforceIf(gt3)
        model.Add(total_night <= 3).OnlyEnforceIf(gt3.Not())
        model.Add(flag == 1).OnlyEnforceIf(gt3)
        model.Add(flag == 0).OnlyEnforceIf(gt3.Not())
        flags.append(flag)
    model.Add(sum(flags) <= max_4)

    for s, c in shift_demand.items():
        model.Add(sum(vars[p, s] for p in names) == c)

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = random.randint(1, 9999)
    status = solver.Solve(model)
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return {p: s for p in names for s in shifts if solver.Value(vars[p, s])}
    else:
        return None

# --- 4. GENERATE ---
if st.button("Generate Schedule") and names:
    all_posts = tech_posts + male_gen_posts + female_gen_posts + male_common_posts + female_common_posts
    shift_demand, post_plan = plan_posts(all_posts, merge_input, c_only_posts)
    shift_assignment = assign_shifts(names, shift_demand, prev_shift_map, weekly_counts)

    if shift_assignment:
        st.success("✅ Schedule Generated")
        doc = Document()
        doc.add_heading("Shift Schedule", 0)

        shift_to_people = defaultdict(list)
        for p, s in shift_assignment.items():
            shift_to_people[s].append(p)

        for s in shift_to_people:
            random.shuffle(shift_to_people[s])

        post_shift_map = OrderedDict()
        vacant = []
        for post, shifts in post_plan.items():
            post_shift_map[post] = {}
            for s in shifts:
                if shift_to_people[s]:
                    person = shift_to_people[s].pop()
                    post_shift_map[post][s] = person
                else:
                    vacant.append(post)

        for post in post_shift_map:
            doc.add_paragraph(post, style='Heading 2')
            for s in ["A", "B", "C", "Day12", "Night12"]:
                if s in post_shift_map[post]:
                    doc.add_paragraph(f"{s}: {post_shift_map[post][s]}", style='List Bullet')

        if vacant:
            doc.add_paragraph("Unfilled Posts:", style='Heading 2')
            for p in vacant:
                doc.add_paragraph(p, style='List Bullet')

        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        st.download_button("Download Schedule", buf, "shift_schedule.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    else:
        st.error("❌ Could not generate feasible schedule. Try reducing constraints or increasing manpower.")
