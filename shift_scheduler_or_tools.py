import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
from collections import defaultdict, OrderedDict
from docx import Document
import io
import random

st.title("💼 Smart Shift Scheduler (with OR-Tools, Gender & Tech Constraints)")

# 1. Upload Excel Input
st.subheader("Upload Excel with Name, Gender (M/F), Technical (Yes/No)")
excel_file = st.file_uploader("Upload Excel", type=["xlsx"])
names_data = []
if excel_file:
    df = pd.read_excel(excel_file)
    for _, row in df.iterrows():
        name = str(row[0]).strip()
        gender = str(row[1]).strip().upper()
        tech = str(row[2]).strip().lower() == 'yes'
        names_data.append((name, gender, tech))

# Split by category
males = [n for n, g, t in names_data if g == 'M']
females = [n for n, g, t in names_data if g == 'F']
tech_males = [n for n, g, t in names_data if g == 'M' and t]
tech_females = [n for n, g, t in names_data if g == 'F' and t]
nontech_males = [n for n in males if n not in tech_males]
nontech_females = [n for n in females if n not in tech_females]

# 2. Duty Post Inputs
st.subheader("Duty Post Inputs")
tech_posts = st.text_area("Technical General Shift Posts").splitlines()
gen_male_posts = st.text_area("General Shift Posts (Male)").splitlines()
gen_female_posts = st.text_area("General Shift Posts (Female)").splitlines()
c_only_posts = st.text_area("C Shift Only Posts (Male Only)").splitlines()
common_posts_female = st.text_area("Common Posts (Female)").splitlines()
common_posts_male = st.text_area("Common Posts (Male)").splitlines()
merge_groups = st.text_area("Merge Groups (e.g. Gate1, Gate2)").splitlines()

# Clean merge map
merge_map = {}
for group in merge_groups:
    items = [p.strip() for p in group.split(",")]
    top = items[0]
    for p in items[1:]:
        merge_map[p] = top

# 3. Generate Shift Demand
def build_shift_plan():
    post_plan = OrderedDict()
    shift_demand = defaultdict(int)
    total_used = 0

    # Assign tech posts
    for post in tech_posts:
        post_plan[post] = ["General"]
        shift_demand["General"] += 1
        total_used += 1

    # General shifts
    for post in gen_male_posts:
        post_plan[post] = ["General"]
        shift_demand["General"] += 1
        total_used += 1
    for post in gen_female_posts:
        post_plan[post] = ["General"]
        shift_demand["General"] += 1
        total_used += 1

    # C-shift only
    for post in c_only_posts:
        post_plan[post] = ["C"]
        shift_demand["C"] += 1
        total_used += 1

    # Common Female posts (A, B by female, C by male)
    for post in common_posts_female:
        post_plan[post + "_A"] = ["A"]
        post_plan[post + "_B"] = ["B"]
        post_plan[post + "_C"] = ["C"]
        shift_demand["A"] += 1
        shift_demand["B"] += 1
        shift_demand["C"] += 1
        total_used += 3

    # Common Male posts (start as 8-hr shifts)
    for post in common_posts_male:
        post_plan[post] = ["A", "B", "C"]

    # Handle merging & conversion to 12-hr if needed
    remaining = len(names_data) - total_used
    posts = [p for p in common_posts_male if p not in merge_map]
    index = len(posts) - 1
    while index >= 0:
        needed = 3 * len(posts)
        if needed <= remaining:
            break
        # convert last post to 12 hr
        post = posts[index]
        post_plan[post] = ["Day12", "Night12"]
        posts.pop(index)
        index -= 1

    for post in posts:
        post_plan[post] = ["A", "B", "C"]

    # Final shift demand update
    for post, shifts in post_plan.items():
        for s in shifts:
            shift_demand[s] += 1

    return post_plan, shift_demand

# 4. Solve with OR-Tools
def solve_with_ortools(post_plan, shift_demand):
    model = cp_model.CpModel()
    shifts = list(shift_demand.keys())
    shift_vars = {}
    people = [n for n, _, _ in names_data]
    for p in people:
        for s in shifts:
            shift_vars[(p, s)] = model.NewBoolVar(f"{p}_{s}")
        model.AddExactlyOne(shift_vars[(p, s)] for s in shifts)

    # Enforce shift counts
    for s in shifts:
        model.Add(sum(shift_vars[(p, s)] for p in people) == shift_demand[s])

    # Gender/Tech constraints
    for post, shifts_list in post_plan.items():
        for s in shifts_list:
            if s == "General":
                if post in tech_posts:
                    for p in people:
                        if p not in tech_males and p not in tech_females:
                            model.Add(shift_vars[(p, s)] == 0)
                elif post in gen_male_posts:
                    for p in people:
                        if p not in males:
                            model.Add(shift_vars[(p, s)] == 0)
                elif post in gen_female_posts:
                    for p in people:
                        if p not in females:
                            model.Add(shift_vars[(p, s)] == 0)
            elif post in c_only_posts or post.endswith("_C"):
                for p in people:
                    if p not in males:
                        model.Add(shift_vars[(p, s)] == 0)
            elif post.endswith("_A") or post.endswith("_B"):
                for p in people:
                    if p not in females:
                        model.Add(shift_vars[(p, s)] == 0)

    # No "Off" shift unless only 8-hr shifts used
    if "Day12" not in shift_demand and "Night12" not in shift_demand:
        shift_demand["Off"] = len(people) - sum(shift_demand.values())
        if shift_demand["Off"] > 0:
            for p in people:
                shift_vars[(p, "Off")] = model.NewBoolVar(f"{p}_Off")
            for p in people:
                model.AddExactlyOne(shift_vars[(p, s)] for s in list(shift_demand.keys()))
            model.Add(sum(shift_vars[(p, "Off")] for p in people) == shift_demand["Off"])

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
        assignment = {}
        for p in people:
            for s in shift_demand:
                if solver.Value(shift_vars[(p, s)]) == 1:
                    assignment[p] = s
        return assignment
    else:
        return None

# 5. Generate Schedule & Word file
if st.button("Generate Schedule") and names_data:
    post_plan, shift_demand = build_shift_plan()
    assignment = solve_with_ortools(post_plan, shift_demand)

    if assignment:
        st.success("✅ Schedule Generated!")
        doc = Document()
        doc.add_heading("Shift Schedule", 0)
        shifts = list(shift_demand.keys())
        shift_groups = defaultdict(list)
        for name, shift in assignment.items():
            shift_groups[shift].append(name)

        for s in shifts:
            doc.add_paragraph(f"Shift {s}", style='Heading 2')
            for name in shift_groups.get(s, []):
                doc.add_paragraph(name, style='List Bullet')

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        st.download_button("📄 Download Schedule", buffer, "shift_schedule.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    else:
        st.error("❌ Could not generate schedule. Try changing post counts or merge more.")
