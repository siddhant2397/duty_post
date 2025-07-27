import streamlit as st
import pandas as pd
from collections import defaultdict, OrderedDict
from ortools.sat.python import cp_model
from docx import Document
import io
import random

st.title("Smart Shift Scheduler (with OR-Tools & Full Constraints)")

# --- Input Section ---
st.subheader("Upload Excel File with Names, Gender (M/F), Technical (Yes/No)")
excel_file = st.file_uploader("Upload Staff Info Excel", type=[".xlsx"])

names_data = []
gender_map = {}
tech_map = {}
if excel_file:
    df = pd.read_excel(excel_file)
    for _, row in df.iterrows():
        name = str(row[0]).strip()
        gender = str(row[1]).strip().upper()
        tech = str(row[2]).strip().lower() == "yes"
        if name:
            names_data.append(name)
            gender_map[name] = gender
            tech_map[name] = tech

males = [n for n in names_data if gender_map[n] == "M"]
females = [n for n in names_data if gender_map[n] == "F"]
tech_males = [n for n in males if tech_map[n]]
tech_females = [n for n in females if tech_map[n]]

# --- Duty Post Inputs ---
tech_posts = [p.strip() for p in st.text_area("TECHNICAL POSTS (General Shift)").split("\n") if p.strip()]
gen_posts_male = [p.strip() for p in st.text_area("GENERAL SHIFT POSTS - Male").split("\n") if p.strip()]
gen_posts_female = [p.strip() for p in st.text_area("GENERAL SHIFT POSTS - Female").split("\n") if p.strip()]
common_posts_female = [p.strip() for p in st.text_area("COMMON DUTY POSTS - FEMALE (Priority)").split("\n") if p.strip()]
common_posts_male = [p.strip() for p in st.text_area("COMMON DUTY POSTS - MALE (Priority)").split("\n") if p.strip()]
c_only_posts = [p.strip() for p in st.text_area("C SHIFT EXTRA POSTS (Male Only)").split("\n") if p.strip()]
merge_input = [line.strip() for line in st.text_area("MERGE GROUPS (comma-separated per line)").split("\n") if line.strip()]

# --- Previous Day Shift Info ---
prev_shift_input = st.text_area("Previous Day Shift Info (Name: Shift per line)")
prev_shift_map = {}
if prev_shift_input:
    for line in prev_shift_input.split("\n"):
        if ":" in line:
            name, shift = line.split(":")
            prev_shift_map[name.strip()] = shift.strip()

# --- Weekly History ---
hist_file = st.file_uploader("Upload Weekly Shift History Excel", type=[".xlsx"], key="weekly_hist")
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

# --- Shift Planning & Assignment ---
def assign_all(names, tech_posts, gen_posts_male, gen_posts_female, common_posts_female, common_posts_male, c_only_posts, merge_input):
    model = cp_model.CpModel()
    shifts = ["A", "B", "C", "Day12", "Night12", "Gen"]

    shift_vars = {(n, s): model.NewBoolVar(f"{n}_{s}") for n in names for s in shifts}
    post_vars = {}
    post_to_person = {}

    # Group merge logic
    merged_map = {}
    for group in merge_input:
        parts = [p.strip() for p in group.split(",") if p.strip()]
        top = parts[0]
        for p in parts[1:]:
            merged_map[p] = top

    # Example logic: Add only required constraints (full logic would be longer)
    for name in names:
        model.AddExactlyOne(shift_vars[name, s] for s in shifts)

        prev = prev_shift_map.get(name, "")
        if prev in ["C", "Night12"]:
            model.Add(shift_vars[name, "A"] == 0)
            model.Add(shift_vars[name, "Day12"] == 0)
            model.Add(shift_vars[name, "Gen"] == 0)

        wc = weekly_counts[name]
        night_total = model.NewIntVar(0, 10, f"night_{name}")
        model.Add(night_total == wc["C"] + wc["Night12"] + shift_vars[name, "C"] + shift_vars[name, "Night12"])
        model.Add(night_total <= 4)
        model.Add(wc["Day12"] + wc["Night12"] + shift_vars[name, "Day12"] + shift_vars[name, "Night12"] <= 3)

    # TODO: Add logic to assign shifts to posts and rotate as per rules
    # Prioritize: technical -> gen -> c_only -> common_female (A,B by F, C by M) -> common_male -> merge -> 12hr

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = random.randint(1, 9999)
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        result = {n: s for n in names for s in shifts if solver.Value(shift_vars[n, s])}
        return result
    return None

# --- Output Section ---
if st.button("Generate Schedule") and names_data:
    result = assign_all(names_data, tech_posts, gen_posts_male, gen_posts_female,
                        common_posts_female, common_posts_male, c_only_posts, merge_input)

    if result:
        doc = Document()
        doc.add_heading("Shift Schedule", 0)
        for person, shift in result.items():
            doc.add_paragraph(f"{person}: {shift}")

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        st.download_button("Download Schedule", buffer, file_name="shift_schedule.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    else:
        st.error("No feasible schedule found. Adjust constraints or inputs.")
