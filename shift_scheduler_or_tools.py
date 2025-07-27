import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import numpy as np

st.title("Advanced Shift Scheduler with Proper Merging & 12-hr Conversion Logic")

# == Input Step 1: Upload Staff Excel ==
file = st.file_uploader(
    "Upload Staff Excel (Name, Gender (M/F), Technical (Yes/No))",
    type="xlsx",
    help="Excel file with columns: Name, Gender, Technical"
)

def to_bool_tech(x):
    if isinstance(x, str):
        return x.strip().lower() == "yes"
    return bool(x)

if file:
    staff = pd.read_excel(file)
    if not all(col in staff.columns for col in ['Name', 'Gender', 'Technical']):
        st.error("Excel must have 'Name', 'Gender', and 'Technical' columns")
        st.stop()

    staff = staff.dropna(subset=['Name', 'Gender', 'Technical'])
    staff['Name'] = staff['Name'].astype(str).str.strip()
    staff['Gender'] = staff['Gender'].astype(str).str.upper().str.strip()
    staff['Technical'] = staff['Technical'].apply(to_bool_tech)
    staff = staff.reset_index(drop=True)
    st.write(f"Uploaded {len(staff)} personnel")
    st.dataframe(staff)

    # == Input Step 2: Define posts and merges ==
    st.header("Define Duty Posts and Merge Groups")

    num_days = st.number_input("Number of Days to Schedule", value=7, min_value=1, max_value=31)

    posts_common_male = st.text_area(
        "Male Common Duty Posts (Priority order from top to bottom, one per line)",
        height=150,
        help="Will be merged and converted as needed"
    ).strip().splitlines()

    posts_common_female = st.text_area(
        "Female Common Duty Posts (one per line, A/B females, C males)",
        height=150
    ).strip().splitlines()

    posts_c_only = st.text_area(
        "C-shift Only Posts (Male only)",
        height=80
    ).strip().splitlines()

    posts_gen_male = st.text_area(
        "General Shift Posts (Non-Tech Males only)",
        height=80
    ).strip().splitlines()

    posts_gen_female = st.text_area(
        "General Shift Posts (Non-Tech Females only)",
        height=80
    ).strip().splitlines()

    posts_tech = st.text_area(
        "Technical General Shift Posts (all technical personnel eligible)",
        height=80
    ).strip().splitlines()

    merge_lines = st.text_area(
        "Mergable Male Common Duty Posts Groups (comma separated, priority order top-down)",
        height=150,
        help="Example:\nMCom2,MCom3\nMCom5,MCom6"
    ).strip().splitlines()

    merge_groups = []
    for line in merge_lines:
        group = [p.strip() for p in line.split(",") if p.strip()]
        if len(group) > 1:
            merge_groups.append(group)

    if st.button("Generate Schedule"):

        names = staff["Name"].tolist()
        is_male = staff["Gender"] == "M"
        is_female = staff["Gender"] == "F"
        is_tech = staff["Technical"]

        shifts8 = ["A", "B", "C"]
        shifts12 = ["Day12", "Night12"]

        # --- Helper functions ---
        def merge_posts(posts, merges):
            post_to_group = {}
            groups = []
            for group in merges:
                merged_name = group[0]
                groups.append((merged_name, set(group)))
                for p in group:
                    post_to_group[p] = merged_name
            unmerged = [p for p in posts if p not in post_to_group]
            for p in unmerged:
                groups.append((p, {p}))
                post_to_group[p] = p

            def priority(g):
                return min(posts.index(p) for p in g[1])

            groups.sort(key=priority)
            merged_dict = {name: members for name, members in groups}
            return merged_dict, [name for name, _ in groups]

        def convert_priority_list_to_shifts(merged_posts_list, eight_hr_count):
            if eight_hr_count > len(merged_posts_list):
                eight_hr_count = len(merged_posts_list)
            eight_hr_posts = merged_posts_list[:eight_hr_count]
            twelve_hr_posts = merged_posts_list[eight_hr_count:]
            return eight_hr_posts, twelve_hr_posts

        # ==== Main iterative scheduling loop ====
        max_merges = len(merge_groups) + 1

        feasible_solution = None
        solution_info = ""
        solution_found = False

        # Phase 1: Exhaust merges with only 8-hour shifts (no conversions)
        for merge_idx in range(max_merges):
            applied_merges = merge_groups[:merge_idx]
            m_dict, m_priority_list = merge_posts(posts_common_male, applied_merges)
            num_merged_posts = len(m_priority_list)

            eight_hr_posts = m_priority_list
            twelve_hr_posts = []

            st.write(f"Trying merges={merge_idx} with no 12hr conversion")
            st.write(f"Merged list: {m_priority_list}")

            # Build entries with no 12 hr shifts - your existing code for entries here
            entries = []

            tech_indices = set(staff[is_tech].index)
            for post in posts_tech:
                entries.append((post, ["General"], tech_indices))
            male_indices = set(staff[is_male].index)
            for post in posts_gen_male:
                entries.append((post, ["General"], male_indices))
            female_indices = set(staff[is_female].index)
            for post in posts_gen_female:
                entries.append((post, ["General"], female_indices))
            for post in posts_c_only:
                entries.append((post, ["C"], male_indices))
            for post in posts_common_female:
                entries.append((f"{post}_A", ["A"], female_indices))
                entries.append((f"{post}_B", ["B"], female_indices))
                entries.append((f"{post}_C", ["C"], male_indices))
            for merged_post in eight_hr_posts:
                for shift in shifts8:
                    entries.append((f"{merged_post}_{shift}", [shift], male_indices))

            # Model build and solve (your existing code)
            model = cp_model.CpModel()
            slot_list = []
            for post, shifts, elig in entries:
                for day in range(num_days):
                    for s in shifts:
                        slot_list.append((post, day, s, elig))

            X = {}
            for slot_id, (post, day, shift, elig) in enumerate(slot_list):
                for p in elig:
                    X[p, slot_id] = model.NewBoolVar(f"assign_{names[p]}_{post}_d{day}_s{shift}")

            for slot_id, (_, _, _, elig) in enumerate(slot_list):
                model.AddExactlyOne([X[p, slot_id] for p in elig])

            all_shifts = shifts8 + shifts12 + ["General"]
            for p in staff.index:
                for day in range(num_days):
                    slots_same_day = [sid for sid, (_, day_, shift_, elig_) in enumerate(slot_list)
                                      if day_ == day and p in elig_]
                    if len(slots_same_day) > 1:
                        model.Add(sum(X[p, sid] for sid in slots_same_day) <= 1)

            n_staff_10pc = max(1, int(np.ceil(0.1 * len(staff))))
            extra_night_staff = [model.NewBoolVar(f'ext_night_{names[p]}') for p in staff.index]
            for idx, p in enumerate(staff.index):
                night_slots = [sid for sid, (_, _, shift_, elig_) in enumerate(slot_list)
                               if shift_ in ("C", "Night12") and p in elig_]
                tot_nights = model.NewIntVar(0, num_days, f"nightcount_{names[p]}")
                model.Add(tot_nights == sum(X[p, sid] for sid in night_slots))
                model.Add(tot_nights <= 3 + extra_night_staff[idx] * num_days)
            model.Add(sum(extra_night_staff) <= n_staff_10pc)

            for p in staff.index:
                slots_12h = [sid for sid, (_, _, shift_, elig_) in enumerate(slot_list)
                             if shift_ in ("Day12", "Night12") and p in elig_]
                tot_12h = model.NewIntVar(0, num_days, f"tot12_{names[p]}")
                model.Add(tot_12h == sum(X[p, sid] for sid in slots_12h))
                model.Add(tot_12h <= 3)

            for p in staff.index:
                for day in range(num_days - 1):
                    c_night_sids = [sid for sid, (_, day_, shift_, elig_) in enumerate(slot_list)
                                    if day_ == day and shift_ in ("C", "Night12") and p in elig_]
                    next_sids = [sid for sid, (_, day_, shift_, elig_) in enumerate(slot_list)
                                 if day_ == day + 1 and shift_ in ("A", "Day12") and p in elig_]
                    for s1 in c_night_sids:
                        for s2 in next_sids:
                            model.AddBoolOr([X[p, s1].Not(), X[p, s2].Not()])

            eight_hr_sids = [sid for sid, (_, _, shift_, _) in enumerate(slot_list) if shift_ in shifts8]
            model.Maximize(sum(X[p, sid] for sid in eight_hr_sids for p in staff.index if (p, sid) in X))

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 60.0
            status = solver.Solve(model)

            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                feasible_solution = {
                    "solver": solver,
                    "X": X,
                    "slot_list": slot_list,
                    "staff": staff,
                    "names": names,
                    "merged_dict": m_dict,
                    "merged_priority_list": m_priority_list,
                    "eight_hr_posts": eight_hr_posts,
                    "twelve_hr_posts": twelve_hr_posts,
                    "merge_idx": merge_idx,
                    "num_convert_12h": 0,
                    "num_days": num_days
                }
                solution_info = f"Solution found with {merge_idx} merge groups and NO 12-hour shifts."
                solution_found = True
                break
            else:
                solution_found = False

        if not solution_found:
            st.write("No feasible solution found with merges and all 8-hour shifts.")
            # Phase 2 - start conversions with full merging
            applied_merges = merge_groups[:]  # all merges
            m_dict, m_priority_list = merge_posts(posts_common_male, applied_merges)
            num_merged_posts = len(m_priority_list)

            for num_convert_12h in range(1, num_merged_posts + 1):
                eight_hr_posts, twelve_hr_posts = convert_priority_list_to_shifts(
                    m_priority_list, num_merged_posts - num_convert_12h
                )
                st.write(f"Trying {num_convert_12h} 12-hour converted posts after full merging")

                entries = []

                tech_indices = set(staff[is_tech].index)
                for post in posts_tech:
                    entries.append((post, ["General"], tech_indices))
                male_indices = set(staff[is_male].index)
                for post in posts_gen_male:
                    entries.append((post, ["General"], male_indices))
                female_indices = set(staff[is_female].index)
                for post in posts_gen_female:
                    entries.append((post, ["General"], female_indices))
                for post in posts_c_only:
                    entries.append((post, ["C"], male_indices))
                for post in posts_common_female:
                    entries.append((f"{post}_A", ["A"], female_indices))
                    entries.append((f"{post}_B", ["B"], female_indices))
                    entries.append((f"{post}_C", ["C"], male_indices))
                for merged_post in eight_hr_posts:
                    for shift in shifts8:
                        entries.append((f"{merged_post}_{shift}", [shift], male_indices))
                for merged_post in twelve_hr_posts:
                    for shift in shifts12:
                        entries.append((f"{merged_post}_{shift}", [shift], male_indices))

                # Build & solve model same way...

                model = cp_model.CpModel()
                slot_list = []
                for post, shifts, elig in entries:
                    for day in range(num_days):
                        for s in shifts:
                            slot_list.append((post, day, s, elig))

                X = {}
                for slot_id, (post, day, shift, elig) in enumerate(slot_list):
                    for p in elig:
                        X[p, slot_id] = model.NewBoolVar(f"assign_{names[p]}_{post}_d{day}_s{shift}")

                for slot_id, (_, _, _, elig) in enumerate(slot_list):
                    model.AddExactlyOne([X[p, slot_id] for p in elig])

                for p in staff.index:
                    for day in range(num_days):
                        slots_same_day = [sid for sid, (_, day_, shift_, elig_) in enumerate(slot_list)
                                          if day_ == day and p in elig_]
                        if len(slots_same_day) > 1:
                            model.Add(sum(X[p, sid] for sid in slots_same_day) <= 1)

                n_staff_10pc = max(1, int(np.ceil(0.1 * len(staff))))
                extra_night_staff = [model.NewBoolVar(f'ext_night_{names[p]}') for p in staff.index]
                for idx, p in enumerate(staff.index):
                    night_slots = [sid for sid, (_, _, shift_, elig_) in enumerate(slot_list)
                                   if shift_ in ("C", "Night12") and p in elig_]
                    tot_nights = model.NewIntVar(0, num_days, f"nightcount_{names[p]}")
                    model.Add(tot_nights == sum(X[p, sid] for sid in night_slots))
                    model.Add(tot_nights <= 3 + extra_night_staff[idx] * num_days)
                model.Add(sum(extra_night_staff) <= n_staff_10pc)

                for p in staff.index:
                    slots_12h = [sid for sid, (_, _, shift_, elig_) in enumerate(slot_list)
                                 if shift_ in ("Day12", "Night12") and p in elig_]
                    tot_12h = model.NewIntVar(0, num_days, f"tot12_{names[p]}")
                    model.Add(tot_12h == sum(X[p, sid] for sid in slots_12h))
                    model.Add(tot_12h <= 3)

                for p in staff.index:
                    for day in range(num_days - 1):
                        c_night_sids = [sid for sid, (_, day_, shift_, elig_) in enumerate(slot_list)
                                        if day_ == day and shift_ in ("C", "Night12") and p in elig_]
                        next_sids = [sid for sid, (_, day_, shift_, elig_) in enumerate(slot_list)
                                     if day_ == day + 1 and shift_ in ("A", "Day12") and p in elig_]
                        for s1 in c_night_sids:
                            for s2 in next_sids:
                                model.AddBoolOr([X[p, s1].Not(), X[p, s2].Not()])

                eight_hr_sids = [sid for sid, (_, _, shift_, _) in enumerate(slot_list) if shift_ in shifts8]
                model.Maximize(sum(X[p, sid] for sid in eight_hr_sids for p in staff.index if (p, sid) in X))

                solver = cp_model.CpSolver()
                solver.parameters.max_time_in_seconds = 60.0
                status = solver.Solve(model)

                if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    feasible_solution = {
                        "solver": solver,
                        "X": X,
                        "slot_list": slot_list,
                        "staff": staff,
                        "names": names,
                        "merged_dict": m_dict,
                        "merged_priority_list": m_priority_list,
                        "eight_hr_posts": eight_hr_posts,
                        "twelve_hr_posts": twelve_hr_posts,
                        "merge_idx": merge_idx,
                        "num_convert_12h": num_convert_12h,
                        "num_days": num_days
                    }
                    solution_info = f"Used full merges and converted {num_convert_12h} posts to 12h shifts."
                    break
                if feasible_solution is not None:
                    break

        if feasible_solution is None:
            st.error("No feasible schedule found after merging and conversions. Try adjusting inputs.")
            st.stop()

        # Continue with your output formatting and off assignments as before...
        # ...


        # === Build final priority list reflecting stepwise merged posts ===
        # Priority order:
        post_priority_order = []
        post_priority_order.extend(posts_tech)
        post_priority_order.extend(posts_gen_male)
        post_priority_order.extend(posts_gen_female)
        post_priority_order.extend(posts_c_only)
        post_priority_order.extend(posts_common_female)
        post_priority_order.extend(feasible_solution["merged_priority_list"])

        # === Function to display grouped schedule per day ===
        def display_schedule_grouped(priority_order, slot_list, X, solver, staff, day=0):
            names = staff['Name'].tolist()
            assignments = {post: {} for post in priority_order}

            for slot_id, (post, slot_day, shift, elig) in enumerate(slot_list):
                if slot_day != day:
                    continue
                for p in elig:
                    if solver.Value(X[p, slot_id]):
                        # Remove shift suffix if base post exists in priority list
                        base_post = post
                        if '_' in post:
                            candidate = post.rsplit('_', 1)[0]
                            if candidate in priority_order:
                                base_post = candidate
                        if base_post not in assignments:
                            assignments[base_post] = {}
                        assignments[base_post][shift] = names[p]

            st.subheader(f"Shift Assignments for Day {day + 1}")
            for post in priority_order:
                if post in assignments and assignments[post]:
                    shifts = assignments[post]
                    shift_person = [f"{shift}: {person}" for shift, person in sorted(shifts.items())]
                    shifts_str = "; ".join(shift_person)
                    st.write(f"**{post}** → {shifts_str}")
                else:
                    st.write(f"**{post}** → No assignment")

        # === Display all days schedule grouped ===
        for day in range(num_days):
            display_schedule_grouped(post_priority_order, feasible_solution["slot_list"], feasible_solution["X"], feasible_solution["solver"], staff, day=day)

        # === OFF assignments logic (same as before) ===
        off_data = []

        # Female OFF logic
        female_shift_needed_posts = []
        for post in posts_gen_female:
            female_shift_needed_posts.append((post, "General"))
        for post in posts_common_female:
            female_shift_needed_posts.append((f"{post}_A", "A"))
            female_shift_needed_posts.append((f"{post}_B", "B"))
        females_indices = staff[is_female].index
        assigned_records = []
        X = feasible_solution["X"]
        slot_list = feasible_solution["slot_list"]
        solver = feasible_solution["solver"]
        for slot_id, (post, day, shift, elig) in enumerate(slot_list):
            for p in elig:
                if solver.Value(X.get((p, slot_id), 0)) == 1:
                    assigned_records.append([
                        staff.loc[p, "Name"],
                        post,
                        day + 1,
                        shift
                    ])
        
        df_assign = pd.DataFrame(assigned_records, columns=["Name", "Post", "Day", "Shift"])

        for p in females_indices:
            assigned_shifts_count = df_assign[(df_assign["Name"] == staff.loc[p, "Name"]) &
                                              (df_assign["Post"].isin([ps for ps, _ in female_shift_needed_posts]))].shape[0]
            total_needed_shifts = len(female_shift_needed_posts) * num_days
            if assigned_shifts_count >= total_needed_shifts:
                for day in range(1, num_days + 1):
                    off_data.append([staff.loc[p, "Name"], "OFF", day, "OFF"])

        # Male off logic: only if all common male posts covered by 8hr shifts (no 12h)
        eight_hr_posts = feasible_solution["eight_hr_posts"]
        twelve_hr_posts = feasible_solution["twelve_hr_posts"]
        if len(twelve_hr_posts) == 0:
            male_common_needed_shifts = []
            for merged_post in eight_hr_posts:
                for shift in shifts8:
                    male_common_needed_shifts.append(f"{merged_post}_{shift}")
            males_indices = staff[is_male].index
            for p in males_indices:
                assigned = df_assign[(df_assign["Name"] == staff.loc[p, "Name"]) &
                                     (df_assign["Post"].isin(male_common_needed_shifts))]
                assigned_days = set(assigned["Day"].tolist())
                for day in range(1, num_days + 1):
                    if day not in assigned_days:
                        off_data.append([staff.loc[p, "Name"], "OFF", day, "OFF"])

        df_off = pd.DataFrame(off_data, columns=["Name", "Post", "Day", "Shift"])
        conflicts = pd.merge(df_off, df_assign, on=["Name", "Day"], how="inner")
        if not conflicts.empty:
            st.warning("Persons assigned both shifts and OFF on same day found:")
            st.dataframe(conflicts)
        else:
            st.warning("Empty")
        

        if not df_off.empty:
            st.subheader("OFF Assignments")
            st.dataframe(df_off.sort_values(by=["Day", "Name"]))
            st.download_button("Download OFF Assignments CSV", df_off.to_csv(index=False), "off_assignments.csv")
        st.write(f"Merge iteration {merge_idx}, applied merges: {applied_merges}")
        st.write("Merged priority list:", m_priority_list)
        st.write("Number of merged posts:", num_merged_posts)
        st.write(f"Converting {num_convert_12h} posts to 12-hour shifts.")
        eight_hr_posts, twelve_hr_posts = convert_priority_list_to_shifts(m_priority_list, max_8h_posts - num_convert_12h)
        st.write("8-hour shifts posts:", eight_hr_posts)
        st.write("12-hour shifts posts:", twelve_hr_posts)

        # Download shift assignments as CSV (flattened)
        df_assign = pd.DataFrame(assigned_records, columns=["Name", "Post", "Day", "Shift"])
        st.download_button("Download Shift Assignments CSV", df_assign.to_csv(index=False), "shift_assignments.csv")

else:
    st.info("Please upload your staff Excel file to start the scheduling process.")
