# Streamlit-based Smart Shift Scheduler with Full OR-Tools Logic

import streamlit as st
import pandas as pd
import random
from ortools.sat.python import cp_model
from collections import defaultdict, OrderedDict
from docx import Document
import io

# --- 1. INPUT SECTION ---
st.title("Smart Shift Scheduler (OR-Tools + All Constraints)")

st.subheader("Upload Excel with Name, Gender(M/F), Technical(Yes/No)")
file = st.file_uploader("Upload staff file", type=[".xlsx"])

names_data = []
if file:
    df = pd.read_excel(file)
    for _, row in df.iterrows():
        name = str(row[0]).strip()
        gender = str(row[1]).strip().upper()
        tech = str(row[2]).strip().lower() == 'yes'
        names_data.append((name, gender, tech))

# Gender-based lists
males = [n for n, g, t in names_data if g == 'M']
females = [n for n, g, t in names_data if g == 'F']
tech_males = [n for n, g, t in names_data if g == 'M' and t]
tech_females = [n for n, g, t in names_data if g == 'F' and t]

# --- 2. DUTY POST INPUT ---
st.subheader("Enter Duty Posts")

gen_posts_male = [p.strip() for p in st.text_area("General Shift Posts (Non-Tech Male)").split("\n") if p.strip()]
gen_posts_female = [p.strip() for p in st.text_area("General Shift Posts (Non-Tech Female)").split("\n") if p.strip()]
tech_posts = [p.strip() for p in st.text_area("General Shift Technical Posts (M+F Tech)").split("\n") if p.strip()]
c_only_posts = [p.strip() for p in st.text_area("C-Shift Only Posts (Male Only)").split("\n") if p.strip()]
common_posts_female = [p.strip() for p in st.text_area("Common Duty Posts (Female)").split("\n") if p.strip()]
common_posts_male = [p.strip() for p in st.text_area("Common Duty Posts (Male)").split("\n") if p.strip()]
merge_input = st.text_area("Merge Groups (comma-separated per line)").split("\n")

# Previous Day Shift Info
st.subheader("Enter Previous Shift Info")
prev_input = st.text_area("Previous Shifts (Name: Shift per line)")
prev_shift_map = {}
if prev_input.strip():
    for line in prev_input.split("\n"):
        if ":" in line:
            name, shift = line.split(":")
            prev_shift_map[name.strip()] = shift.strip()

# Weekly History
st.subheader("Upload Weekly Shift History")
hist_file = st.file_uploader("Upload Weekly History Excel", type=[".xlsx"], key="hist")
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

# --- 3. SCHEDULING AND CONSTRAINT SOLVING ---
if st.button("Generate Schedule"):
    names = [n for n, _, _ in names_data]
    shift_types = ["A", "B", "C", "Day12", "Night12", "GS"]
    post_plan = OrderedDict()
    shift_demand = defaultdict(int)

    # 1. Technical General Shift Posts
    for post in tech_posts:
        post_plan[post] = ["GS"]
        shift_demand["GS"] += 1

    # 2. Non-Technical General Shift Posts
    for post in gen_posts_male:
        post_plan[post] = ["GS"]
        shift_demand["GS"] += 1
    for post in gen_posts_female:
        post_plan[post] = ["GS"]
        shift_demand["GS"] += 1

    # 3. C-Only Posts (Males Only)
    for post in c_only_posts:
        post_plan[post] = ["C"]
        shift_demand["C"] += 1

    # 4. Common Female Posts (AB by females, C by males)
    for post in common_posts_female:
        post_plan[post] = ["A", "B", "C"]
        shift_demand["A"] += 1
        shift_demand["B"] += 1
        shift_demand["C"] += 1

    # 5. Common Male Posts (handle merging + 12-hour logic later)
    reduced_male_posts = []
    merged = set()
    merge_map = {}
    for line in merge_input:
        group = [p.strip() for p in line.split(",") if p.strip()]
        if len(group) > 1:
            top = group[0]
            for p in group[1:]:
                merge_map[p] = top
                merged.add(p)

    for p in common_posts_male:
        if p not in merged:
            reduced_male_posts.append(p)

    # Assign as A,B,C shifts initially
    temp_post_status = OrderedDict((p, ["A", "B", "C"]) for p in reduced_male_posts)

    def required_people(status_map):
        return sum(3 if v == ["A", "B", "C"] else 2 for v in status_map.values())

    index = len(reduced_male_posts) - 1
    total_people = len(males)
    used = sum(shift_demand[s] for s in shift_demand)
    while used + required_people(temp_post_status) > total_people:
        # Convert to 12-hour
        while index >= 0:
            p = reduced_male_posts[index]
            if temp_post_status[p] == ["A", "B", "C"]:
                temp_post_status[p] = ["Day12", "Night12"]
                break
            index -= 1
        else:
            break
        used = sum(shift_demand[s] for s in shift_demand)

    for p, shifts in temp_post_status.items():
        post_plan[p] = shifts
        for s in shifts:
            shift_demand[s] += 1

    # --- OR-TOOLS SOLVER ---
    model = cp_model.CpModel()
    shifts_all = shift_types + ["Off"]
    shift_vars = {(n, s): model.NewBoolVar(f"{n}_{s}") for n in names for s in shifts_all}

    for n in names:
        model.AddExactlyOne(shift_vars[n, s] for s in shifts_all)

    # Previous shift constraints
    for n in names:
        prev = prev_shift_map.get(n, "")
        if prev in ["C", "Night12"]:
            for banned in ["A", "Day12", "GS"]:
                model.Add(shift_vars[n, banned] == 0)

    # Weekly limits
    flags = []
    max_4 = max(1, int(len(males) * 0.10))
    for n in names:
        wc = weekly_counts[n]
        total_night = model.NewIntVar(0, 10, f"tn_{n}")
        total_12hr = model.NewIntVar(0, 10, f"t12_{n}")
        model.Add(total_night == wc["C"] + wc["Night12"] + shift_vars[n,"C"] + shift_vars[n,"Night12"])
        model.Add(total_12hr == wc["Day12"] + wc["Night12"] + shift_vars[n,"Day12"] + shift_vars[n,"Night12"])
        model.Add(total_12hr <= 3)
        flag = model.NewBoolVar(f"flag_{n}")
        model.Add(flag == 1).OnlyEnforceIf(total_night > 3)
        model.Add(flag == 0).OnlyEnforceIf(total_night <= 3)
        flags.append(flag)
    model.Add(sum(flags) <= max_4)

    # Shift demand
    for s, cnt in shift_demand.items():
        model.Add(sum(shift_vars[n, s] for n in names) == cnt)

    # Off constraint
    for n in names:
        others = [shift_vars[n, s].Not() for s in shift_types]
        model.Add(shift_vars[n, "Off"] == 1).OnlyEnforceIf(others)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.random_seed = random.randint(1, 9999)
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        result = {}
        for n in names:
            for s in shifts_all:
                if solver.Value(shift_vars[n, s]):
                    result[n] = s

        # Assign to posts
        assigned = defaultdict(list)
        people_per_shift = defaultdict(list)
        for n, s in result.items():
            people_per_shift[s].append(n)

        for s in people_per_shift:
            random.shuffle(people_per_shift[s])

        final_plan = OrderedDict()
        for post, shifts in post_plan.items():
            final_plan[post] = {}
            for s in shifts:
                if people_per_shift[s]:
                    final_plan[post][s] = people_per_shift[s].pop()

        # Output to Word
        doc = Document()
        doc.add_heading("Shift Schedule", 0)
        for post in final_plan:
            doc.add_paragraph(post, style='Heading 2')
            for s, p in final_plan[post].items():
                doc.add_paragraph(f"{s}: {p}", style='List Bullet')

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        st.download_button("Download Schedule", buffer, file_name="shift_schedule.docx")
    else:
        st.error("No feasible schedule found.")
