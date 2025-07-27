import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
from collections import defaultdict, OrderedDict
from docx import Document
import random
import io

st.title("Smart Shift Scheduler (with OR-Tools, Gender, Tech & Fair Constraints)")

# --- INPUT SECTION ---
st.subheader("1. Upload Excel File with Names, Gender, Technical")
file = st.file_uploader("Excel file with Name, Gender(M/F), Technical(Yes/No)", type=["xlsx"])
names_data = []
if file:
    df = pd.read_excel(file)
    for _, row in df.iterrows():
        name = str(row[0]).strip()
        gender = str(row[1]).strip().upper()
        tech = str(row[2]).strip().lower() == "yes"
        if name:
            names_data.append((name, gender, tech))

# --- Grouping ---
males = [n for n, g, t in names_data if g == 'M']
females = [n for n, g, t in names_data if g == 'F']
tech_males = [n for n, g, t in names_data if g == 'M' and t]
tech_females = [n for n, g, t in names_data if g == 'F' and t]

# --- POST INPUT ---
st.subheader("2. Post Details")

common_male = [p.strip() for p in st.text_area("Common Male Duty Posts (priority order)").split("\n") if p.strip()]
common_female = [p.strip() for p in st.text_area("Common Female Duty Posts (priority order)").split("\n") if p.strip()]
c_only_posts = [p.strip() for p in st.text_area("C Shift Only Posts (Male Only)").split("\n") if p.strip()]

general_male = [p.strip() for p in st.text_area("General Shift Posts (Non-Tech, Male)").split("\n") if p.strip()]
general_female = [p.strip() for p in st.text_area("General Shift Posts (Non-Tech, Female)").split("\n") if p.strip()]
technical_posts = [p.strip() for p in st.text_area("Technical General Shift Posts").split("\n") if p.strip()]

merge_input = st.text_area("Merge Post Groups (Comma Separated, per line)").split("\n")
merged_map = {}
merged_set = set()

for group in merge_input:
    items = [x.strip() for x in group.split(",") if x.strip()]
    if len(items) > 1:
        top = items[0]
        for p in items[1:]:
            merged_map[p] = top
            merged_set.add(p)

# --- Previous Shift Input ---
st.subheader("3. Previous Day Shift Info")
prev_shift_input = st.text_area("Name: Shift per line (e.g. John: C)")
prev_shift_map = {}
if prev_shift_input:
    for line in prev_shift_input.strip().split("\n"):
        if ":" in line:
            name, shift = line.split(":")
            prev_shift_map[name.strip()] = shift.strip()

# --- Weekly Shift History Input ---
st.subheader("4. Weekly Shift History Upload")
hist_file = st.file_uploader("Upload Weekly History (Name, C, Night12, Day12)", type=["xlsx"])
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

# --- SHIFT ASSIGNMENT FUNCTION ---
def assign_shift_schedule():
    model = cp_model.CpModel()
    people = [n for n, _, _ in names_data]
    shifts = ["A", "B", "C", "Day12", "Night12", "General"]

    shift_vars = {(p, s): model.NewBoolVar(f"{p}_{s}") for p in people for s in shifts}
    for p in people:
        model.AddExactlyOne(shift_vars[p, s] for s in shifts)

    # -- Constraint: No A or General after C or Night12
    for p in people:
        prev = prev_shift_map.get(p, "")
        if prev in ["C", "Night12"]:
            model.Add(shift_vars[p, "A"] == 0)
            model.Add(shift_vars[p, "Day12"] == 0)
            model.Add(shift_vars[p, "General"] == 0)

    # -- Weekly Limits
    max_4_c = max(1, int(len(males) * 0.10))
    flags = []
    for p in people:
        wc = weekly_counts[p]
        total_night = model.NewIntVar(0, 10, f"tn_{p}")
        total_12 = model.NewIntVar(0, 10, f"t12_{p}")
        model.Add(total_night == wc["C"] + wc["Night12"] + shift_vars[p,"C"] + shift_vars[p,"Night12"])
        model.Add(total_12 == wc["Day12"] + wc["Night12"] + shift_vars[p,"Day12"] + shift_vars[p,"Night12"])
        model.Add(total_12 <= 3)

        flag = model.NewBoolVar(f"flag_{p}")
        over_c = model.NewBoolVar(f"overc_{p}")
        model.Add(total_night > 3).OnlyEnforceIf(over_c)
        model.Add(total_night <= 3).OnlyEnforceIf(over_c.Not())
        model.Add(flag == 1).OnlyEnforceIf(over_c)
        model.Add(flag == 0).OnlyEnforceIf(over_c.Not())
        flags.append(flag)
    model.Add(sum(flags) <= max_4_c)

    # -- Assign shifts to posts (simplified logic here)
    shift_demand = defaultdict(int)
    post_assignment = {}

    for post in technical_posts:
        if tech_males:
            post_assignment[post] = tech_males.pop()
            shift_demand["General"] += 1
        elif tech_females:
            post_assignment[post] = tech_females.pop()
            shift_demand["General"] += 1

    for post in general_male:
        if males:
            post_assignment[post] = males.pop()
            shift_demand["General"] += 1

    for post in general_female:
        if females:
            post_assignment[post] = females.pop()
            shift_demand["General"] += 1

    for post in c_only_posts:
        if males:
            post_assignment[post] = males.pop()
            shift_demand["C"] += 1

    # Common female first (A, B only for females, C covered by males)
    for post in common_female:
        if females:
            post_assignment[post+"_A"] = females.pop()
            shift_demand["A"] += 1
        if females:
            post_assignment[post+"_B"] = females.pop()
            shift_demand["B"] += 1
        if males:
            post_assignment[post+"_C"] = males.pop()
            shift_demand["C"] += 1

    # Common male posts (convert to 12-hr if insufficient)
    reduced_posts = [p for p in common_male if p not in merged_set]
    for post in reduced_posts:
        if len(males) >= 3:
            post_assignment[post+"_A"] = males.pop()
            post_assignment[post+"_B"] = males.pop()
            post_assignment[post+"_C"] = males.pop()
            shift_demand["A"] += 1
            shift_demand["B"] += 1
            shift_demand["C"] += 1
        elif len(males) >= 2:
            post_assignment[post+"_Day12"] = males.pop()
            post_assignment[post+"_Night12"] = males.pop()
            shift_demand["Day12"] += 1
            shift_demand["Night12"] += 1

    # Enforce shift demand
    for s in shift_demand:
        model.Add(sum(shift_vars[p, s] for p in people) == shift_demand[s])

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.random_seed = random.randint(1, 9999)
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        final = {}
        for p in people:
            for s in shifts:
                if solver.Value(shift_vars[p, s]):
                    final[p] = s
        return post_assignment, final
    else:
        return None, None

# --- GENERATE SCHEDULE ---
if st.button("Generate Schedule") and names_data:
    post_map, shift_map = assign_shift_schedule()
    if not post_map:
        st.error("Unable to generate feasible schedule. Relax constraints or check input.")
    else:
        st.success("Schedule Generated!")
        doc = Document()
        doc.add_heading("Shift Schedule", 0)
        for post, person in post_map.items():
            doc.add_paragraph(f"{post}: {person}")
        doc.add_heading("Individual Shift Assignment", level=1)
        for person, shift in shift_map.items():
            doc.add_paragraph(f"{person}: {shift}")
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        st.download_button("Download Word Schedule", buffer, "shift_schedule.docx")
