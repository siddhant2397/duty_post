import streamlit as st
import pandas as pd
from collections import defaultdict, OrderedDict
from ortools.sat.python import cp_model
from docx import Document
import io
import random

st.title("Smart Shift Scheduler (with OR-Tools & Constraints)")

# --- INPUT SECTION ---
st.subheader("Upload Excel File with Name, Gender (M/F), Technical (Yes/No)")
excel_file = st.file_uploader("Upload Excel", type=["xlsx"])
people = []

if excel_file:
    df = pd.read_excel(excel_file)
    for _, row in df.iterrows():
        name = str(row[0]).strip()
        gender = str(row[1]).strip().upper()
        tech = str(row[2]).strip().lower() == 'yes'
        if name:
            people.append((name, gender, tech))

# Categorize
males = [p for p in people if p[1] == 'M']
females = [p for p in people if p[1] == 'F']
tech_males = [p[0] for p in males if p[2]]
tech_females = [p[0] for p in females if p[2]]
nontech_males = [p[0] for p in males if not p[2]]
nontech_females = [p[0] for p in females if not p[2]]
all_names = [p[0] for p in people]

# Post Inputs
tech_posts = [p.strip() for p in st.text_area("Technical General Shift Posts").split("\n") if p.strip()]
gen_male_posts = [p.strip() for p in st.text_area("General Shift Male Posts").split("\n") if p.strip()]
gen_female_posts = [p.strip() for p in st.text_area("General Shift Female Posts").split("\n") if p.strip()]
c_only_posts = [p.strip() for p in st.text_area("C-Shift-Only Posts (Males)").split("\n") if p.strip()]
common_posts_male = [p.strip() for p in st.text_area("Common Posts for Males (Priority Order)").split("\n") if p.strip()]
common_posts_female = [p.strip() for p in st.text_area("Common Posts for Females (Priority Order)").split("\n") if p.strip()]
merge_lines = [line.strip() for line in st.text_area("Merge Groups (comma-separated)").split("\n") if line.strip()]

# Merge logic
merged_map = {}
merged_set = set()
for line in merge_lines:
    parts = [p.strip() for p in line.split(",") if p.strip()]
    if len(parts) > 1:
        base = parts[0]
        for p in parts[1:]:
            merged_map[p] = base
            merged_set.add(p)

# Previous shift input
prev_shift_input = st.text_area("Previous Day Shift Info (Name: Shift)")
prev_shift_map = {}
if prev_shift_input:
    for line in prev_shift_input.splitlines():
        if ":" in line:
            n, s = line.split(":")
            prev_shift_map[n.strip()] = s.strip()

# Weekly history
hist_file = st.file_uploader("Upload Weekly History Excel", type=["xlsx"])
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

# Post demand plan
post_plan = OrderedDict()
shift_demand = defaultdict(int)

def add_post(p, shifts):
    for s in shifts:
        shift_demand[s] += 1
    post_plan[p] = shifts

# Assign posts in priority order
for post in tech_posts:
    add_post(post, ["GS"])

for post in gen_male_posts:
    add_post(post, ["GS"])

for post in gen_female_posts:
    add_post(post, ["GS"])

for post in c_only_posts:
    add_post(post, ["C"])

# Female common: A/B by females, C by males
for post in common_posts_female:
    add_post(post + "_AB", ["A", "B"])
    add_post(post + "_C", ["C"])

# Now try to cover male common posts with 3-shift
available_male = len([p for p in males if p[0] not in post_plan.values()])
post_candidates = [p for p in common_posts_male if p not in merged_set]
for post in post_candidates:
    add_post(post, ["A", "B", "C"])

# If shortage, merge and convert to 12-hr
def try_convert_12hr():
    shortage = sum(shift_demand.values()) - len(all_names)
    if shortage <= 0:
        return
    for post in reversed(common_posts_male):
        if post in post_plan and post_plan[post] == ["A", "B", "C"]:
            post_plan[post+"_Day12"] = ["Day12"]
            post_plan[post+"_Night12"] = ["Night12"]
            shift_demand["Day12"] += 1
            shift_demand["Night12"] += 1
            shift_demand["A"] -= 1
            shift_demand["B"] -= 1
            shift_demand["C"] -= 1
            del post_plan[post]
            shortage -= 1
            if shortage <= 0:
                break

try_convert_12hr()

# OR-Tools model
if st.button("Generate Schedule"):
    st.info("Building schedule with OR-Tools...")

    model = cp_model.CpModel()
    all_shifts = ["A", "B", "C", "Day12", "Night12", "GS", "Off"]
    shift_vars = {(n, s): model.NewBoolVar(f"{n}_{s}") for n in all_names for s in all_shifts}
    for n in all_names:
        model.AddExactlyOne(shift_vars[n, s] for s in all_shifts)

    for n in all_names:
        prev = prev_shift_map.get(n, "")
        if prev in ["C", "Night12"]:
            model.Add(shift_vars[n, "A"] == 0)
            model.Add(shift_vars[n, "Day12"] == 0)
            model.Add(shift_vars[n, "GS"] == 0)

    flags = []
    max4 = max(1, int(0.10 * len(all_names)))
    for n in all_names:
        wc = weekly_counts[n]
        c_night = wc["C"] + wc["Night12"]
        total_12hr = wc["Night12"] + wc["Day12"]
        c_flag = model.NewBoolVar(f"{n}_cflag")
        model.Add(c_night + shift_vars[n, "C"] + shift_vars[n, "Night12"] <= 3).OnlyEnforceIf(c_flag.Not())
        model.Add(c_night + shift_vars[n, "C"] + shift_vars[n, "Night12"] > 3).OnlyEnforceIf(c_flag)
        flags.append(c_flag)
        model.Add(total_12hr + shift_vars[n, "Day12"] + shift_vars[n, "Night12"] <= 3)

    model.Add(sum(flags) <= max4)

    for s in shift_demand:
        model.Add(sum(shift_vars[n, s] for n in all_names) == shift_demand[s])

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        assignment = {}
        for n in all_names:
            for s in all_shifts:
                if solver.Value(shift_vars[n, s]):
                    assignment[n] = s

        shift_to_people = defaultdict(list)
        for n, s in assignment.items():
            shift_to_people[s].append(n)

        doc = Document()
        doc.add_heading("Shift Schedule", 0)
        for post, shifts in post_plan.items():
            doc.add_paragraph(post, style='Heading 2')
            for s in shifts:
                if shift_to_people[s]:
                    p = shift_to_people[s].pop()
                    doc.add_paragraph(f"Shift {s}: {p}", style='List Bullet')

        buffer = io.BytesIO()
        doc.save(buffer)
        st.download_button("Download Schedule", buffer.getvalue(), file_name="shift_schedule.docx")
    else:
        st.error("❌ Unable to generate schedule. Try adjusting inputs or constraints.")
