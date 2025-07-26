import streamlit as st
from collections import defaultdict, OrderedDict
import pandas as pd
import random
from ortools.sat.python import cp_model
from docx import Document
import io

st.title("Shift Scheduler with Technical, Gender & Shift Constraints")

# ---------- INPUT SECTION ----------

male_names = st.text_area("Enter Male Names (comma separated)", "John, Mike, Raj")
female_names = st.text_area("Enter Female Names (comma separated)", "Anita, Priya, Neha")
males = [name.strip() for name in male_names.split(",") if name.strip()]
females = [name.strip() for name in female_names.split(",") if name.strip()]
names = males + females

tech_males = st.text_area("Enter Technically Qualified Male Names (comma separated)", "Mike, Raj")
tech_females = st.text_area("Enter Technically Qualified Female Names (comma separated)", "Priya")
tech_males = [name.strip() for name in tech_males.split(",") if name.strip()]
tech_females = [name.strip() for name in tech_females.split(",") if name.strip()]
tech_all = tech_males + tech_females

tech_posts = [p.strip() for p in st.text_area("Enter Technical General Shift Posts", "Control Room, Server Room").split(',') if p.strip()]
male_gen_posts = [p.strip() for p in st.text_area("Enter Non-Technical General Shift Posts (Male)", "Reception, Security").split(',') if p.strip()]
female_gen_posts = [p.strip() for p in st.text_area("Enter Non-Technical General Shift Posts (Female)", "Helpdesk, HR Desk").split(',') if p.strip()]
male_common_posts = [p.strip() for p in st.text_area("Enter Common Posts (Male)", "Gate, Lobby, CCTV").split(',') if p.strip()]
female_common_posts = [p.strip() for p in st.text_area("Enter Common Posts (Female)", "Gate, Lobby").split(',') if p.strip()]
c_only_posts = [p.strip() for p in st.text_area("Enter C Shift-Only Posts (Male Only)", "Night Patrol, Watch Tower").split(',') if p.strip()]
merge_input = st.text_area("Enter Merge Groups (comma-separated per line)").split("\n")

prev_shift_input = st.text_area("Enter Previous Day Shifts (Name: Shift per line)")
prev_shift_map = {}
if prev_shift_input.strip():
    for line in prev_shift_input.strip().split("\n"):
        if ":" in line:
            name, shift = line.split(":")
            prev_shift_map[name.strip()] = shift.strip()

weekly_log_file = st.file_uploader("Upload Weekly Log CSV (Name, C, Day12, Night12)", type="csv")
weekly_shift_log = defaultdict(lambda: {'C': 0, 'Day12': 0, 'Night12': 0})
if weekly_log_file:
    df = pd.read_csv(weekly_log_file)
    for _, row in df.iterrows():
        weekly_shift_log[row['Name']] = {'C': row['C'], 'Day12': row['Day12'], 'Night12': row['Night12']}

# ---------- PLANNING AND ASSIGNMENT ----------

def blocked_today(person):
    return prev_shift_map.get(person, '') in ['C', 'Night12']

def can_take_c_shift(person):
    wc = weekly_shift_log[person]
    c_count = wc['C']
    return (c_count < 3 or (c_count < 4 and sum(1 for m in males if weekly_shift_log[m]['C'] >= 4) < len(males)*0.1))

def can_take_12hr(person):
    wc = weekly_shift_log[person]
    return wc['Day12'] + wc['Night12'] < 3

def assign_post(post_list, candidates, used, condition=None):
    for p in candidates:
        if p in used:
            continue
        if blocked_today(p):
            continue
        if condition and not condition(p):
            continue
        used.add(p)
        return p
    return None

shift_demand = defaultdict(int)
post_plan = OrderedDict()
used_people = set()

# Assign General Posts First
for post in tech_posts:
    person = assign_post([p for p in tech_all], tech_all, used_people, condition=can_take_12hr)
    if person:
        post_plan[post] = ['General']
        shift_demand['General'] += 1
        weekly_shift_log[person]['Day12'] += 1

for post in male_gen_posts:
    person = assign_post(males, males, used_people)
    if person:
        post_plan[post] = ['General']
        shift_demand['General'] += 1

for post in female_gen_posts:
    person = assign_post(females, females, used_people)
    if person:
        post_plan[post] = ['General']
        shift_demand['General'] += 1

# Assign C-Only Posts
for post in c_only_posts:
    person = assign_post(males, males, used_people, condition=can_take_c_shift)
    if person:
        post_plan[post] = ['C']
        shift_demand['C'] += 1
        weekly_shift_log[person]['C'] += 1

# Prepare merged set
merged_map = {}
merged_set = set()
for group in merge_input:
    parts = [p.strip() for p in group.split(',') if p.strip()]
    if len(parts) > 1:
        top = parts[0]
        for p in parts[1:]:
            merged_map[p] = top
            merged_set.add(p)

# Filter unique posts
common_posts_all = list(OrderedDict.fromkeys(male_common_posts + female_common_posts))
filtered_common = [p for p in common_posts_all if p not in merged_set]

# Initially assign all as 8-hr shift
common_status = OrderedDict((p, ['A', 'B', 'C']) for p in filtered_common)

def required_people(post_status):
    return sum(3 if v == ['A', 'B', 'C'] else 2 for v in post_status.values())

posts = list(common_status.items())
index = len(posts) - 1
while True:
    total_required = len(post_plan) + required_people(common_status)
    if total_required <= len(names):
        break
    while index >= 0:
        post, shifts = posts[index]
        if common_status[post] == ['A', 'B', 'C']:
            common_status[post] = ['Day12', 'Night12']
            break
        index -= 1
    else:
        while posts:
            post, _ = posts.pop()
            del common_status[post]
            total_required = len(post_plan) + required_people(common_status)
            if total_required <= len(names):
                break

for post, shifts in common_status.items():
    post_plan[post] = shifts
    for s in shifts:
        shift_demand[s] += 1

# ---------- OR-TOOLS ASSIGNMENT ----------
def assign_shifts_ortools():
    model = cp_model.CpModel()
    shifts = list(set(s for shift_list in post_plan.values() for s in shift_list)) + ['Off']
    shift_vars = {(p, s): model.NewBoolVar(f"{p}_{s}") for p in names for s in shifts}

    for p in names:
        model.AddExactlyOne(shift_vars[p, s] for s in shifts)

    for p in names:
        prev = prev_shift_map.get(p, '')
        if prev in ['C', 'Night12']:
            for s in ['A', 'Day12', 'General']:
                if (p, s) in shift_vars:
                    model.Add(shift_vars[p, s] == 0)

    flags = []
    for p in names:
        wc = weekly_shift_log[p]
        total_night = model.NewIntVar(0, 10, f"tn_{p}")
        total_12hr = model.NewIntVar(0, 10, f"t12_{p}")
        model.Add(total_night == wc['C'] + wc['Night12'] + shift_vars[p,"C"] + shift_vars[p,"Night12"] if (p,"Night12") in shift_vars else 0)
        model.Add(total_12hr == wc['Day12'] + wc['Night12'] + shift_vars[p,"Day12"] + shift_vars[p,"Night12"] if (p,"Day12") in shift_vars else 0)
        model.Add(total_12hr <= 3)
        flag = model.NewBoolVar(f"flag_{p}")
        gt3 = model.NewBoolVar(f"gt3_{p}")
        model.Add(total_night > 3).OnlyEnforceIf(gt3)
        model.Add(total_night <= 3).OnlyEnforceIf(gt3.Not())
        model.Add(flag == 1).OnlyEnforceIf(gt3)
        model.Add(flag == 0).OnlyEnforceIf(gt3.Not())
        flags.append(flag)
    model.Add(sum(flags) <= max(1, int(len(males)*0.10)))

    for s, count in shift_demand.items():
        model.Add(sum(shift_vars[p, s] for p in names if (p, s) in shift_vars) == count)

    for p in names:
        if (p, 'Off') in shift_vars:
            model.Add(shift_vars[p, 'Off'] == 1).OnlyEnforceIf([
                shift_vars[p, s].Not() for s in shifts if s != 'Off'
            ])

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = random.randint(1, 10000)
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        result = {}
        for p in names:
            for s in shifts:
                if solver.Value(shift_vars[p, s]):
                    result[p] = s
        return result
    return None

# ---------- OUTPUT ----------
if st.button("Generate Schedule"):
    result = assign_shifts_ortools()
    if result:
        st.success("✅ Schedule generated!")
        doc = Document()
        doc.add_heading("Shift Schedule by Duty Post", 0)

        shifts_order = ['General', 'A', 'B', 'C', 'Day12', 'Night12']
        shift_to_people = defaultdict(list)
        for p, s in result.items():
            shift_to_people[s].append(p)

        for s in shift_to_people:
            random.shuffle(shift_to_people[s])

        post_shift_map = OrderedDict()
        vacant_posts = []
        for post, shifts in post_plan.items():
            post_shift_map[post] = {}
            for s in shifts:
                if shift_to_people[s]:
                    post_shift_map[post][s] = shift_to_people[s].pop()
                else:
                    vacant_posts.append(post)

        for post in post_plan:
            doc.add_paragraph(post, style='Heading 2')
            for s in shifts_order:
                if post_shift_map[post].get(s):
                    doc.add_paragraph(f"Shift {s}: {post_shift_map[post][s]}", style='List Bullet')

        if vacant_posts:
            doc.add_paragraph("\nUnfilled Posts:", style='Heading 2')
            for vp in vacant_posts:
                doc.add_paragraph(vp, style='List Bullet')

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        st.download_button("Download Schedule", buffer, file_name="shift_schedule.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    else:
        st.error("❌ No feasible schedule found. Adjust constraints or increase staff.")
