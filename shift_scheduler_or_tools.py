import streamlit as st
import pandas as pd
from collections import defaultdict, OrderedDict
from ortools.sat.python import cp_model
import random
from docx import Document
import io

st.title("Smart Shift Scheduler (Streamlit Version)")

# --- Upload Excel File ---
st.subheader("Upload Excel File: Name | Gender (M/F) | Technical (Yes/No)")
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

if names_data:
    # Categorize
    males = [n for n, g, t in names_data if g == 'M']
    females = [n for n, g, t in names_data if g == 'F']
    tech_males = [n for n, g, t in names_data if g == 'M' and t]
    tech_females = [n for n, g, t in names_data if g == 'F' and t]

    # --- Text Inputs ---
    st.subheader("Enter Posts")
    common_posts_male = st.text_area("COMMON duty posts for MALE (priority order)").split("\n")
    common_posts_female = st.text_area("COMMON duty posts for FEMALE (priority order)").split("\n")
    c_only_posts = st.text_area("C-SHIFT-ONLY duty posts (Male Only)").split("\n")
    gen_posts_male = st.text_area("NON-TECH GENERAL SHIFT duty posts (Male)").split("\n")
    gen_posts_female = st.text_area("NON-TECH GENERAL SHIFT duty posts (Female)").split("\n")
    tech_posts = st.text_area("TECHNICAL GENERAL SHIFT duty posts").split("\n")

    merge_input = st.text_area("MERGE groups (comma-separated per line)").split("\n")
    merge_map = {}
    for line in merge_input:
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) > 1:
            main = parts[0]
            for p in parts[1:]:
                merge_map[p] = main

    if st.button("Generate Schedule"):
        shift_options = ['A', 'B', 'C', 'Day12', 'Night12', 'General']
        person_roles = {}
        post_assignments = OrderedDict()

        tech_all = tech_males + tech_females
        for post in tech_posts:
            if tech_all:
                person = tech_all.pop(0)
                person_roles[person] = 'Tech'
                post_assignments[post] = (person, 'General')

        for post in gen_posts_male:
            if males:
                person = males.pop(0)
                person_roles[person] = 'GenMale'
                post_assignments[post] = (person, 'General')

        for post in gen_posts_female:
            if females:
                person = females.pop(0)
                person_roles[person] = 'GenFemale'
                post_assignments[post] = (person, 'General')

        for post in c_only_posts:
            if males:
                person = males.pop(0)
                person_roles[person] = 'COnly'
                post_assignments[post] = (person, 'C')

        for post in common_posts_female:
            for shift in ['A', 'B']:
                if females:
                    person = females.pop(0)
                    person_roles[person] = 'CommonF'
                    post_assignments[f"{post}_{shift}"] = (person, shift)

        for post in common_posts_female:
            if males:
                person = males.pop(0)
                person_roles[person] = 'CommonF_C'
                post_assignments[f"{post}_C"] = (person, 'C')

        merged_set = set(merge_map.keys())
        reduced_posts = []
        seen = set()

        for post in common_posts_male:
            if post in merged_set or post in seen:
                continue
            reduced_posts.append(post)
            seen.add(post)

        def people_needed(posts_dict):
            return sum(3 if v == ['A', 'B', 'C'] else 2 for v in posts_dict.values())

        post_status = OrderedDict((p, ['A', 'B', 'C']) for p in reduced_posts)
        for p_from, p_to in merge_map.items():
            if p_from in reduced_posts:
                reduced_posts.remove(p_from)
                post_status[p_to] = ['A', 'B', 'C']

        while people_needed(post_status) > len(males):
            last_post = list(post_status.keys())[-1]
            post_status[last_post] = ['Day12', 'Night12']

        for post, shifts in post_status.items():
            for s in shifts:
                if males:
                    person = males.pop(0)
                    person_roles[person] = 'CommonM'
                    post_assignments[f"{post}_{s}"] = (person, s)

        # --- Display ---
        st.subheader("📋 Final Schedule")
        for post, (person, shift) in post_assignments.items():
            st.text(f"{post:25s} --> {person:15s} | Shift: {shift}")

        # --- Download as Word File ---
        doc = Document()
        doc.add_heading("Shift Schedule", 0)
        for post, (person, shift) in post_assignments.items():
            doc.add_paragraph(f"{post} --> {person} | Shift: {shift}")

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        st.download_button("Download Schedule as Word", buffer, file_name="shift_schedule.docx")
