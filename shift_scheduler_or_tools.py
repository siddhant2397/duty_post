import streamlit as st
import pandas as pd
from collections import defaultdict, OrderedDict
from ortools.sat.python import cp_model
from docx import Document
import io
import random

st.title("Fair Shift Scheduler (OR-Tools Based with Proper Off Logic)")

# --- INPUT: Excel with Name, Gender, Technical ---
st.subheader("Upload Excel File (Name, Gender[M/F], Technical[Yes/No])")
file = st.file_uploader("Upload", type=["xlsx"])
names_data = []
if file:
    df = pd.read_excel(file)
    for _, row in df.iterrows():
        name = str(row[0]).strip()
        gender = str(row[1]).strip().upper()
        tech = str(row[2]).strip().lower() == "yes"
        names_data.append((name, gender, tech))

# --- Split by gender/tech ---
males = [n for n, g, t in names_data if g == "M"]
females = [n for n, g, t in names_data if g == "F"]
tech_males = [n for n, g, t in names_data if g == "M" and t]
tech_females = [n for n, g, t in names_data if g == "F" and t]

# --- Post Inputs ---
common_male_posts = [p.strip() for p in st.text_area("COMMON MALE Duty Posts (priority)").split("\n") if p.strip()]
common_female_posts = [p.strip() for p in st.text_area("COMMON FEMALE Duty Posts (priority)").split("\n") if p.strip()]
c_only_posts = [p.strip() for p in st.text_area("C-SHIFT ONLY Duty Posts (Male Only)").split("\n") if p.strip()]
gen_male_posts = [p.strip() for p in st.text_area("GENERAL SHIFT Non-Tech MALE Posts").split("\n") if p.strip()]
gen_female_posts = [p.strip() for p in st.text_area("GENERAL SHIFT Non-Tech FEMALE Posts").split("\n") if p.strip()]
tech_posts = [p.strip() for p in st.text_area("GENERAL SHIFT TECHNICAL Posts").split("\n") if p.strip()]
merge_input = st.text_area("MERGE groups (comma-separated per line)").split("\n")

# --- Merge Map ---
merged_map = {}
for line in merge_input:
    parts = [p.strip() for p in line.split(",") if p.strip()]
    if len(parts) > 1:
        main = parts[0]
        for p in parts[1:]:
            merged_map[p] = main

# --- Step 1: Assign posts ---
post_plan = OrderedDict()
post_demand = defaultdict(int)

# 1. Technical Posts – assign only to tech people
for post in tech_posts:
    post_plan[post] = ["G"]  # General shift
    post_demand["G"] += 1

# 2. General Shift Non-Tech
for post in gen_male_posts:
    post_plan[post] = ["G"]
    post_demand["G"] += 1

for post in gen_female_posts:
    post_plan[post] = ["G"]
    post_demand["G"] += 1

# 3. C-shift only posts (Male)
for post in c_only_posts:
    post_plan[post] = ["C"]
    post_demand["C"] += 1

# 4. Common Female Duty Posts – A, B by female; C by male
for post in common_female_posts:
    post_plan[post+"_F_AB"] = ["A", "B"]
    post_plan[post+"_F_C"] = ["C"]
    post_demand["A"] += 1
    post_demand["B"] += 1
    post_demand["C"] += 1

# 5. Common Male Duty Posts – assign last (with fallback logic)
pending_common_male = [p for p in common_male_posts if p not in merged_map]

# Add as ABC (3-shift)
for post in pending_common_male:
    post_plan[post] = ["A", "B", "C"]
    post_demand["A"] += 1
    post_demand["B"] += 1
    post_demand["C"] += 1

# Now fallback logic: Merge & 12-hr shift conversion
def convert_posts(plan, demand, people_count):
    while True:
        total_demand = sum(demand.values())
        if total_demand <= people_count:
            break
        # Start converting from bottom priority
        for post in reversed(pending_common_male):
            if plan.get(post) == ["A", "B", "C"]:
                plan[post] = ["Day12", "Night12"]
                demand["A"] -= 1
                demand["B"] -= 1
                demand["C"] -= 1
                demand["Day12"] += 1
                demand["Night12"] += 1
                break
        else:
            break

# Apply conversion
convert_posts(post_plan, post_demand, len(males))

# --- Step 2: Build OR-Tools Model ---
st.subheader("Generate Schedule")
if st.button("Schedule") and names_data:
    shifts = ["A", "B", "C", "Day12", "Night12", "G"]
    model = cp_model.CpModel()
    shift_vars = {}
    for n, g, t in names_data:
        for s in shifts:
            shift_vars[n, s] = model.NewBoolVar(f"{n}_{s}")
        model.AddExactlyOne(shift_vars[n, s] for s in shifts)

    # Enforce shift demand
    for s in shifts:
        count = post_demand[s]
        model.Add(sum(shift_vars[n, s] for n, _, _ in names_data) == count)

    # Enforce qualification and gender
    for n, g, t in names_data:
        if n not in tech_males + tech_females:
            for post in tech_posts:
                model.Add(shift_vars[n, "G"] == 0)  # Non-tech can't take technical
        if g == "F":
            model.Add(shift_vars[n, "C"] == 0)  # Female can't take C
            model.Add(shift_vars[n, "Night12"] == 0)

    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        st.success("✅ Schedule generated successfully!")
        doc = Document()
        doc.add_heading("Shift Schedule by Duty Post", 0)
        shift_to_people = defaultdict(list)

        for n, _, _ in names_data:
            for s in shifts:
                if solver.Value(shift_vars[n, s]):
                    shift_to_people[s].append(n)

        # Assign people to posts
        post_assignment = OrderedDict()
        for post, required_shifts in post_plan.items():
            post_assignment[post] = {}
            for s in required_shifts:
                if shift_to_people[s]:
                    person = shift_to_people[s].pop()
                    post_assignment[post][s] = person

        # --- Output ---
        for post, shifts_dict in post_assignment.items():
            doc.add_paragraph(f"\n{post}", style='Heading 2')
            for s, p in shifts_dict.items():
                doc.add_paragraph(f"{s} : {p}", style='List Bullet')

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        st.download_button("Download Schedule", buffer, file_name="shift_schedule.docx")
    else:
        st.error("❌ Could not find feasible schedule. Try reducing posts or merging.")
