import streamlit as st
import pandas as pd
from collections import defaultdict, OrderedDict
from ortools.sat.python import cp_model
from docx import Document
import io, random

st.title("Smart Shift Scheduler (OR-Tools | Gender & Tech Aware)")

# --- Upload Name List ---
st.subheader("Upload Excel with Name, Gender (M/F), Technical (Yes/No)")
excel = st.file_uploader("Upload Staff List", type=["xlsx"])
data = []
if excel:
    df = pd.read_excel(excel)
    for _, row in df.iterrows():
        name, gender, tech = row[0], str(row[1]).strip().upper(), str(row[2]).strip().lower()
        if name and gender in ['M', 'F']:
            data.append((name.strip(), gender, tech == 'yes'))

# --- Split by categories ---
males = [n for n, g, t in data if g == 'M']
females = [n for n, g, t in data if g == 'F']
tech_males = [n for n, g, t in data if g == 'M' and t]
tech_females = [n for n, g, t in data if g == 'F' and t]

# --- Text Inputs for Posts ---
tech_posts = [p.strip() for p in st.text_area("Technical Duty Posts (General Shift)").split("\n") if p.strip()]
gen_male_posts = [p.strip() for p in st.text_area("General Male Posts (9–5)").split("\n") if p.strip()]
gen_female_posts = [p.strip() for p in st.text_area("General Female Posts (9–5)").split("\n") if p.strip()]
c_only_posts = [p.strip() for p in st.text_area("C-Shift ONLY Posts (Male Only)").split("\n") if p.strip()]
common_male = [p.strip() for p in st.text_area("COMMON Posts (Male) [Priority]").split("\n") if p.strip()]
common_female = [p.strip() for p in st.text_area("COMMON Posts (Female) [Priority]").split("\n") if p.strip()]
merge_groups = st.text_area("MERGE groups (comma-separated, one per line)").split("\n")
merge_map = {}
merged_set = set()

for line in merge_groups:
    parts = [p.strip() for p in line.split(",") if p.strip()]
    if len(parts) > 1:
        top = parts[0]
        for p in parts[1:]:
            merge_map[p] = top
            merged_set.add(p)

# --- Previous Shift Map ---
prev_map = {}
prev_input = st.text_area("Previous Shifts (Name: Shift)")
for line in prev_input.split("\n"):
    if ":" in line:
        n, s = line.split(":")
        prev_map[n.strip()] = s.strip()

# --- Weekly Count Upload ---
hist = st.file_uploader("Weekly History Excel", type=["xlsx"])
weekly = defaultdict(lambda: {"C": 0, "Night12": 0, "Day12": 0})
if hist:
    df_hist = pd.read_excel(hist)
    for _, row in df_hist.iterrows():
        name = row[0]
        weekly[name] = {
            "C": int(row.get("C", 0)),
            "Night12": int(row.get("Night12", 0)),
            "Day12": int(row.get("Day12", 0))
        }

# --- Planning Logic ---
def generate_demand():
    post_plan = OrderedDict()
    demand = defaultdict(int)
    used = 0

    def add(post, shifts):
        post_plan[post] = shifts
        for s in shifts: demand[s] += 1

    for p in tech_posts:
        add(p, ["GenTech"])
        used += 1
    for p in gen_male_posts:
        add(p, ["GenMale"])
        used += 1
    for p in gen_female_posts:
        add(p, ["GenFemale"])
        used += 1
    for p in c_only_posts:
        add(p, ["C"])
        used += 1

    # Female Common: A/B by female, C by male
    for p in common_female:
        add(p+"_F", ["A", "B"])
        add(p+"_M", ["C"])
        used += 3

    # Male Common: Try 8-hr, then merge, then 12-hr
    reduced = [p for p in common_male if p not in merged_set]
    status = OrderedDict((p, ["A", "B", "C"]) for p in reduced)

    def req(status):
        return sum(3 if v == ["A","B","C"] else 2 for v in status.values())

    index = len(status) - 1
    while used + req(status) > len(males):
        while index >= 0:
            p, v = list(status.items())[index]
            if v == ["A","B","C"]:
                status[p] = ["Day12","Night12"]
                break
            index -= 1
        else:
            status.popitem()
            index = len(status)-1

    for p, shifts in status.items(): add(p, shifts)
    return demand, post_plan

# --- OR-Tools Solver ---
def solve(demand, post_plan):
    people = [n for n, _, _ in data]
    model = cp_model.CpModel()
    shifts = list(set([s for v in post_plan.values() for s in v] + ["Off"]))
    vars = {(p,s): model.NewBoolVar(f"{p}_{s}") for p in people for s in shifts}
    for p in people:
        model.AddExactlyOne(vars[p,s] for s in shifts)

    # No A/Day12/Gen after C/Night12
    for p in people:
        prev = prev_map.get(p,"")
        if prev in ["C","Night12"]:
            for s in ["A","Day12","GenMale","GenFemale","GenTech"]:
                if s in shifts: model.Add(vars[p,s] == 0)

    # Weekly Constraints
    flags = []
    max4 = max(1, int(len(males)*0.10))
    for p in people:
        wc = weekly[p]
        C_night = wc["C"] + wc["Night12"]
        Day12 = wc["Day12"]
        night_new = C_night + int("C" in shifts and vars[p,"C"]) + int("Night12" in shifts and vars[p,"Night12"])
        twelve_new = Day12 + int("Day12" in shifts and vars[p,"Day12"]) + int("Night12" in shifts and vars[p,"Night12"])
        if "C" in shifts: model.Add(night_new <= 4)
        if "Day12" in shifts: model.Add(twelve_new <= 3)

    for s, count in demand.items():
        model.Add(sum(vars[p,s] for p in people if s in vars[p]) == count)

    solver = cp_model.CpSolver()
    random.shuffle(people)
    status = solver.Solve(model)
    if status not in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
        return None

    assign = {}
    for p in people:
        for s in shifts:
            if solver.Value(vars[p,s]): assign[p] = s
    return assign

# --- Output Section ---
if st.button("Generate Schedule"):
    demand, plan = generate_demand()
    assignment = solve(demand, plan)

    if not assignment:
        st.error("No valid schedule. Try changing inputs or reduce posts.")
    else:
        st.success("✅ Schedule Generated!")
        doc = Document()
        doc.add_heading("Shift Assignment by Post", 0)
        shift_to_people = defaultdict(list)
        for p, s in assignment.items():
            shift_to_people[s].append(p)

        shift_copy = {k: v.copy() for k, v in shift_to_people.items()}
        post_map = OrderedDict()

        for post, shifts in plan.items():
            post_map[post] = {}
            for s in shifts:
                if shift_copy.get(s):
                    post_map[post][s] = shift_copy[s].pop()

        for post, sh in post_map.items():
            doc.add_paragraph(post, style="Heading 2")
            for s, p in sh.items():
                doc.add_paragraph(f"{s}: {p}", style="List Bullet")

        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        st.download_button("Download Word File", buf, file_name="shift_schedule.docx")
