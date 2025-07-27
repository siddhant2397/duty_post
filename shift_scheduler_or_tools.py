import streamlit as st
import pandas as pd
from collections import defaultdict, OrderedDict
from ortools.sat.python import cp_model
from docx import Document
import io
import random

st.title("Smart Shift Scheduler with Constraints")

# --- INPUT SECTION ---
st.subheader("Upload Excel: Name, Gender(M/F), Technical(Yes/No)")
excel_file = st.file_uploader("Upload Personnel File", type=[".xlsx"])
names_data = []

if excel_file:
    df = pd.read_excel(excel_file)
    for _, row in df.iterrows():
        name = str(row[0]).strip()
        gender = str(row[1]).strip().upper()
        tech = str(row[2]).strip().lower() == "yes"
        if name:
            names_data.append((name, gender, tech))

# Categorize
males = [n for n, g, t in names_data if g == "M"]
females = [n for n, g, t in names_data if g == "F"]
tech_males = [n for n, g, t in names_data if g == "M" and t]
tech_females = [n for n, g, t in names_data if g == "F" and t]

# Post Inputs
tech_posts = [p.strip() for p in st.text_area("TECH GENERAL Posts").split("\n") if p.strip()]
gen_posts_male = [p.strip() for p in st.text_area("General Shift (Male)").split("\n") if p.strip()]
gen_posts_female = [p.strip() for p in st.text_area("General Shift (Female)").split("\n") if p.strip()]
c_only_posts = [p.strip() for p in st.text_area("C-Shift Only Posts (Male)").split("\n") if p.strip()]
common_posts_male = [p.strip() for p in st.text_area("COMMON Posts (Male)").split("\n") if p.strip()]
common_posts_female = [p.strip() for p in st.text_area("COMMON Posts (Female)").split("\n") if p.strip()]
merge_input = [line.strip() for line in st.text_area("Merge Groups (comma-separated)").split("\n") if line.strip()]

# Previous Shift
prev_shift_input = st.text_area("Previous Shift (Name:Shift per line)")
prev_shift_map = {}
if prev_shift_input:
    for line in prev_shift_input.split("\n"):
        if ":" in line:
            name, shift = line.split(":")
            prev_shift_map[name.strip()] = shift.strip()

# Weekly Shift History
hist_file = st.file_uploader("Upload Weekly History", type=[".xlsx"], key="hist")
weekly_counts = defaultdict(lambda: {"C": 0, "Night12": 0, "Day12": 0})
if hist_file:
    hist_df = pd.read_excel(hist_file)
    for _, row in hist_df.iterrows():
        name = str(row[0])
        weekly_counts[name] = {
            "C": int(row.get("C", 0)),
            "Night12": int(row.get("Night12", 0)),
            "Day12": int(row.get("Day12", 0))
        }

# --- SHIFT PLANNING ---
def plan_duty_posts():
    shift_demand = defaultdict(int)
    post_plan = OrderedDict()
    merged_map = {}
    merged_set = set()

    # Process merge groups
    for group in merge_input:
        parts = [p.strip() for p in group.split(",") if p.strip()]
        if len(parts) > 1:
            top = parts[0]
            for p in parts[1:]:
                merged_map[p] = top
                merged_set.add(p)

    def register(post, shifts):
        for s in shifts:
            shift_demand[s] += 1
        post_plan[post] = shifts

    # Priority 1: TECH POSTS (General Shift)
    for post in tech_posts:
        register(post, ["Gen"])

    # Priority 2: NON-TECH GEN POSTS
    for post in gen_posts_male:
        register(post, ["Gen"])
    for post in gen_posts_female:
        register(post, ["Gen"])

    # Priority 3: C-SHIFT ONLY
    for post in c_only_posts:
        register(post, ["C"])

    # Priority 4: COMMON FEMALE (A,B by females, C by males)
    for post in common_posts_female:
        register(post+"_A", ["A"])
        register(post+"_B", ["B"])
        register(post+"_C", ["C"])

    # Priority 5: COMMON MALE (initially 3 shifts)
    common_final = []
    for post in common_posts_male:
        if post not in merged_set and post not in common_final:
            common_final.append(post)

    remaining = len(males)  # Estimate
    posts = [(p, ["A", "B", "C"]) for p in common_final]

    def required_people(pl):
        return sum(3 if v == ["A", "B", "C"] else 2 for _, v in pl)

    idx = len(posts) - 1
    while True:
        total = required_people(posts)
        if total + len(post_plan) <= remaining:
            break
        while idx >= 0:
            p, v = posts[idx]
            if v == ["A", "B", "C"]:
                posts[idx] = (p, ["Day12", "Night12"])
                break
            idx -= 1
        else:
            posts.pop()
            idx -= 1

    for p, v in posts:
        for s in v:
            shift_demand[s] += 1
        post_plan[p] = v

    return shift_demand, post_plan

# --- OR-TOOLS SCHEDULING ---
def assign_with_ortools(people, shift_demand):
    model = cp_model.CpModel()
    shifts = ["A", "B", "C", "Gen", "Day12", "Night12", "Off"]
    vars = {(p, s): model.NewBoolVar(f"{p}_{s}") for p in people for s in shifts}

    for p in people:
        model.AddExactlyOne(vars[p, s] for s in shifts)

    for p in people:
        prev = prev_shift_map.get(p, "")
        if prev in ["C", "Night12"]:
            model.Add(vars[p, "A"] == 0)
            model.Add(vars[p, "Day12"] == 0)
            model.Add(vars[p, "Gen"] == 0)

    flags = []
    max4 = max(1, int(len(people) * 0.1))
    for p in people:
        wc = weekly_counts[p]
        night_new = model.NewIntVar(0, 10, f"night_{p}")
        model.Add(night_new == wc["C"] + wc["Night12"] +
                  vars[p, "C"] + vars[p, "Night12"])

        day12_new = model.NewIntVar(0, 10, f"day12_{p}")
        model.Add(day12_new == wc["Day12"] + wc["Night12"] +
                  vars[p, "Day12"] + vars[p, "Night12"])
        model.Add(day12_new <= 3)

        flag = model.NewBoolVar(f"flag_{p}")
        model.Add(night_new > 3).OnlyEnforceIf(flag)
        model.Add(night_new <= 3).OnlyEnforceIf(flag.Not())
        flags.append(flag)

    model.Add(sum(flags) <= max4)

    for s, count in shift_demand.items():
        model.Add(sum(vars[p, s] for p in people) == count)

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = random.randint(0, 9999)
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        result = {}
        for p in people:
            for s in shifts:
                if solver.Value(vars[p, s]):
                    result[p] = s
        return result
    else:
        return None

# --- FINAL SCHEDULING AND EXPORT ---
if st.button("Generate Schedule") and names_data:
    all_names = [n for n, _, _ in names_data]
    demand, plan = plan_duty_posts()
    assignment = assign_with_ortools(all_names, demand)

    if assignment:
        st.success("Schedule Generated")
        doc = Document()
        doc.add_heading("Shift Schedule", 0)

        shift_to_people = defaultdict(list)
        for name, shift in assignment.items():
            shift_to_people[shift].append(name)

        for post, shifts in plan.items():
            doc.add_paragraph(post, style="Heading 2")
            for s in shifts:
                if shift_to_people[s]:
                    person = shift_to_people[s].pop()
                    doc.add_paragraph(f"{s}: {person}", style="List Bullet")

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        st.download_button("Download DOCX", buffer, "shift_schedule.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    else:
        st.error("No feasible schedule found.")
