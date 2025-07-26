import streamlit as st
import pandas as pd
from collections import defaultdict, OrderedDict
from ortools.sat.python import cp_model
from docx import Document
import io
import random

st.title("Smart Shift Scheduler (Gender + Technical + Constraints + OR-Tools)")

# --- 1. INPUT SECTION ---
st.subheader("Upload Excel File with Names, Gender(M/F), Technical(Yes/No)")
excel_file = st.file_uploader("Upload Excel File", type=[".xlsx"])

names_data = []
if excel_file:
    df = pd.read_excel(excel_file)
    for _, row in df.iterrows():
        name = str(row[0]).strip()
        gender = str(row[1]).strip().upper()
        tech = str(row[2]).strip().lower() == "yes"
        if name:
            names_data.append((name, gender, tech))

# Group people based on categories
males = [n for n, g, t in names_data if g == 'M']
females = [n for n, g, t in names_data if g == 'F']
tech_males = [n for n, g, t in names_data if g == 'M' and t]
tech_females = [n for n, g, t in names_data if g == 'F' and t]

# Posts Input
common_posts_male = [p.strip() for p in st.text_area("COMMON Duty Posts (Male)").split("\n") if p.strip()]
common_posts_female = [p.strip() for p in st.text_area("COMMON Duty Posts (Female)").split("\n") if p.strip()]
c_only_posts = [p.strip() for p in st.text_area("C-SHIFT ONLY Posts (Male Only)").split("\n") if p.strip()]
gen_posts_male = [p.strip() for p in st.text_area("NON-TECH General Shift Posts (Male)").split("\n") if p.strip()]
gen_posts_female = [p.strip() for p in st.text_area("NON-TECH General Shift Posts (Female)").split("\n") if p.strip()]
tech_posts = [p.strip() for p in st.text_area("TECHNICAL General Shift Posts").split("\n") if p.strip()]
merge_input = st.text_area("MERGE Groups (comma-separated per line)").split("\n")

# Previous shift info
prev_shift_input = st.text_area("Previous Shift (Name: Shift per line)")
prev_shift_map = {}
if prev_shift_input.strip():
    for line in prev_shift_input.strip().split("\n"):
        if ":" in line:
            name, shift = line.split(":")
            prev_shift_map[name.strip()] = shift.strip()

# Weekly history
hist_file = st.file_uploader("Upload Weekly Shift History Excel", type=[".xlsx"], key="weekly_hist")
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

# --- 2. POST ALLOCATION PLAN ---
def create_post_plan():
    post_plan = OrderedDict()
    shift_demand = defaultdict(int)
    merge_map = {}
    merged_set = set()

    for group in merge_input:
        parts = [p.strip() for p in group.split(",") if p.strip()]
        if len(parts) > 1:
            top = parts[0]
            for p in parts[1:]:
                merge_map[p] = top
                merged_set.add(p)

    def reduce(posts):
        reduced = []
        seen = set()
        for p in posts:
            if p not in seen and p not in merged_set:
                reduced.append(p)
                seen.add(p)
        return reduced

    reduced_common_male = reduce(common_posts_male)
    reduced_common_female = reduce(common_posts_female)

    for post in tech_posts:
        post_plan[post] = ["G-Tech"]
        shift_demand["G-Tech"] += 1

    for post in gen_posts_male:
        post_plan[post] = ["G-M"]
        shift_demand["G-M"] += 1

    for post in gen_posts_female:
        post_plan[post] = ["G-F"]
        shift_demand["G-F"] += 1

    for post in c_only_posts:
        post_plan[post] = ["C"]
        shift_demand["C"] += 1

    # Add common posts for males
    for post in reduced_common_male[::-1]:  # Reverse for converting from bottom
        post_plan[post] = ["A", "B", "C"]
        shift_demand["A"] += 1
        shift_demand["B"] += 1
        shift_demand["C"] += 1

    # Add common posts for females
    for post in reduced_common_female[::-1]:
        post_plan[post] = ["A", "B"]
        shift_demand["A"] += 1
        shift_demand["B"] += 1

    return post_plan, shift_demand

# --- 3. OR-TOOLS ASSIGNMENT ---
def assign_with_ortools():
    post_plan, shift_demand = create_post_plan()
    people = [n for n, _, _ in names_data]
    model = cp_model.CpModel()
    shifts = ["A", "B", "C", "Day12", "Night12", "G-M", "G-F", "G-Tech", "Off"]
    shift_vars = {(p, s): model.NewBoolVar(f"{p}_{s}") for p in people for s in shifts}

    # One shift per person
    for p in people:
        model.AddExactlyOne(shift_vars[p, s] for s in shifts)

    # Shift demand constraints
    for s, cnt in shift_demand.items():
        model.Add(sum(shift_vars[p, s] for p in people) == cnt)

    # Previous day constraint
    for p in people:
        prev = prev_shift_map.get(p, "")
        if prev in ["C", "Night12"]:
            for s in ["A", "Day12", "G-M", "G-F", "G-Tech"]:
                model.Add(shift_vars[p, s] == 0)

    # Weekly history constraints
    flags = []
    male_count = len([n for n, g, _ in names_data if g == 'M'])
    for p in people:
        wc = weekly_counts[p]
        total_night = model.NewIntVar(0, 10, f"night_{p}")
        total_12hr = model.NewIntVar(0, 10, f"day12_{p}")
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
    model.Add(sum(flags) <= max(1, int(male_count * 0.10)))

    # Eligibility constraints
    for p, g, t in names_data:
        if g == "F":
            model.Add(shift_vars[p, "C"] == 0)
            model.Add(shift_vars[p, "Night12"] == 0)
        if not t:
            model.Add(shift_vars[p, "G-Tech"] == 0)

    # Off shift definition
    for p in people:
        model.Add(shift_vars[p, "Off"] == 1).OnlyEnforceIf([
            shift_vars[p, s].Not() for s in shifts if s != "Off"
        ])

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.random_seed = random.randint(0, 10000)
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        assignment = {}
        for p in people:
            for s in shifts:
                if solver.Value(shift_vars[p, s]):
                    assignment[p] = s
        return assignment, post_plan
    else:
        return None, None

# --- 4. OUTPUT ---
if st.button("Generate Schedule") and names_data:
    result, post_plan = assign_with_ortools()
    if result:
        st.success("✅ Schedule generated successfully!")

        shift_to_people = defaultdict(list)
        for p, s in result.items():
            shift_to_people[s].append(p)

        doc = Document()
        doc.add_heading("Shift Schedule by Post", 0)
        for post, shifts in post_plan.items():
            doc.add_paragraph(post, style='Heading 2')
            for s in shifts:
                if shift_to_people[s]:
                    person = shift_to_people[s].pop()
                    doc.add_paragraph(f"Shift {s}: {person}", style='List Bullet')
        buffer = io.BytesIO()
        doc.save(buffer)
        st.download_button("Download Word Schedule", buffer.getvalue(), file_name="Shift_Schedule.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    else:
        st.error("❌ Could not generate a feasible schedule. Relax some constraints or increase staff.")
