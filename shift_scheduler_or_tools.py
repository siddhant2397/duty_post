import streamlit as st
import pandas as pd
from collections import defaultdict, OrderedDict
from ortools.sat.python import cp_model
from docx import Document
import io
import random

st.title("Smart Shift Scheduler with Constraints (OR-Tools)")

# --- 1. INPUT SECTION ---
st.subheader("Upload Excel File (Name, Gender, Technical)")
excel_file = st.file_uploader("Upload Excel", type=["xlsx"])

names_data = []
if excel_file:
    df = pd.read_excel(excel_file)
    for _, row in df.iterrows():
        name = str(row[0]).strip()
        gender = str(row[1]).strip().upper()
        tech = str(row[2]).strip().lower() == "yes"
        if name:
            names_data.append((name, gender, tech))

# Gender/Skill based grouping
males = [n for n, g, t in names_data if g == 'M']
females = [n for n, g, t in names_data if g == 'F']
tech_males = [n for n, g, t in names_data if g == 'M' and t]
tech_females = [n for n, g, t in names_data if g == 'F' and t]

# --- Post Inputs ---
st.subheader("Enter Posts (One per line)")
tech_posts = st.text_area("Technical General Duty Posts").split("\n")
gen_male_posts = st.text_area("General Duty Posts (Male)").split("\n")
gen_female_posts = st.text_area("General Duty Posts (Female)").split("\n")
c_only_posts = st.text_area("C Shift Only Posts (Male Only)").split("\n")
common_male_posts = st.text_area("Common Posts (Male)").split("\n")
common_female_posts = st.text_area("Common Posts (Female)").split("\n")
merge_lines = st.text_area("Merge Groups (comma separated)").split("\n")

# Previous shift input
prev_shift_input = st.text_area("Previous Day Shifts (Name: Shift)")
prev_shift_map = {}
for line in prev_shift_input.split("\n"):
    if ":" in line:
        name, shift = line.strip().split(":")
        prev_shift_map[name.strip()] = shift.strip()

# Weekly History
hist_file = st.file_uploader("Weekly History (Name, C, Day12, Night12)", type=["xlsx"])
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

# --- 2. SHIFT DEMAND PLANNER ---
@st.cache_data
def plan_posts():
    posts = OrderedDict()
    shift_demand = defaultdict(int)

    # 1. TECH POSTS (General Shift)
    for p in tech_posts:
        posts[p] = ["G"]
        shift_demand["G"] += 1

    # 2. General Male/Female Posts (G shift)
    for p in gen_male_posts:
        posts[p] = ["G"]
        shift_demand["G"] += 1
    for p in gen_female_posts:
        posts[p] = ["G"]
        shift_demand["G"] += 1

    # 3. C-only posts
    for p in c_only_posts:
        posts[p] = ["C"]
        shift_demand["C"] += 1

    # 4. Common female posts (A, B by female, C by male)
    for p in common_female_posts:
        posts[p+"_A"] = ["A"]
        posts[p+"_B"] = ["B"]
        posts[p+"_C"] = ["C"]
        shift_demand["A"] += 1
        shift_demand["B"] += 1
        shift_demand["C"] += 1

    # 5. Common male posts (A, B, C initially)
    common_pool = [p for p in common_male_posts if p.strip()]
    for p in common_pool:
        posts[p] = ["A", "B", "C"]

    return posts, shift_demand, common_pool

def apply_merging_and_convert(common_posts, shift_demand, posts, total_available):
    merged = OrderedDict()
    merged_map = {}
    merge_dict = {}
    for line in merge_lines:
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if parts:
            top = parts[0]
            merge_dict[top] = parts

    reduced = [p for p in common_posts if p not in merged_map]
    index = len(reduced) - 1
    while True:
        total_required = sum(len(shifts) for shifts in posts.values())
        if total_required <= total_available:
            break
        while index >= 0:
            post = reduced[index]
            if posts.get(post) == ["A", "B", "C"]:
                posts[post+"_Day12"] = ["Day12"]
                posts[post+"_Night12"] = ["Night12"]
                del posts[post]
                shift_demand["Day12"] += 1
                shift_demand["Night12"] += 1
                break
            index -= 1
        else:
            break

    return posts, shift_demand

# --- 3. SOLVE WITH OR-TOOLS ---
def solve(names_data, posts, shift_demand):
    model = cp_model.CpModel()
    people = [n for n, _, _ in names_data]
    shifts = list(set(s for shift_list in posts.values() for s in shift_list) | {"Off"})
    var = {(p, s): model.NewBoolVar(f"{p}_{s}") for p in people for s in shifts}

    for p in people:
        model.AddExactlyOne(var[p, s] for s in shifts)

        prev = prev_shift_map.get(p, "")
        if prev in ["C", "Night12"]:
            model.Add(var[p, "A"] == 0)
            model.Add(var[p, "Day12"] == 0)
            model.Add(var[p, "G"] == 0)

        wc = weekly_counts[p]
        night_count = wc["C"] + wc["Night12"]
        model.Add(night_count + var[p, "C"] + var[p, "Night12"] <= 4)
        model.Add(wc["Day12"] + wc["Night12"] + var[p, "Day12"] + var[p, "Night12"] <= 3)

    for s, count in shift_demand.items():
        model.Add(sum(var[p, s] for p in people) == count)

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = random.randint(1, 10000)
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        result = {}
        for p in people:
            for s in shifts:
                if solver.Value(var[p, s]):
                    result[p] = s
        return result
    else:
        return None

# --- 4. OUTPUT ---
if st.button("Generate Schedule") and names_data:
    posts, shift_demand, common_pool = plan_posts()
    total_people = len(names_data)
    posts, shift_demand = apply_merging_and_convert(common_pool, shift_demand, posts, total_people)
    assignment = solve(names_data, posts, shift_demand)

    if assignment:
        st.success("Schedule generated successfully!")

        # Prepare DOC
        doc = Document()
        doc.add_heading("Shift Schedule", 0)

        shift_groups = defaultdict(list)
        for name, shift in assignment.items():
            shift_groups[shift].append(name)

        for shift in sorted(shift_groups.keys()):
            doc.add_paragraph(f"\nShift {shift}", style='Heading 2')
            for p in shift_groups[shift]:
                doc.add_paragraph(p, style='List Bullet')

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        st.download_button("Download Word Schedule", buffer, file_name="schedule.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    else:
        st.error("❌ Could not generate schedule. Try relaxing constraints or increasing available people.")
