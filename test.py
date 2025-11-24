import pandas as pd
import sys
from ortools.sat.python import cp_model

# --- 1. Initial Data Loading and Preparation (from your script) ---

eventCapacity = 129
numTeams = 33 # This will be dynamically determined by the data

# Define the filename and the columns we want to keep
filename = 'd4gcdata_rows.csv'
columns_to_keep = [
    'user_id',
    'team_id',
    'data_skill',
    'cloud_skill',
    'status',
    'work_preferrence' # Using the exact spelling from the CSV header
    
]

# Load the CSV file into a pandas DataFrame
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Please ensure it's in the same directory.")
    sys.exit()

df = df[df['status'] == 'confirmed']
# Select only the specified columns
filtered_df = df[columns_to_keep].copy() # Use .copy() to avoid SettingWithCopyWarning

# Fill any empty 'team_id' values with a placeholder
filtered_df.loc[:, 'team_id'] = filtered_df['team_id'].fillna('No Team')

# Map skill strings to 0–4 integers (unknown/missing -> 0; beginner=1, intermediate=2, advanced=3, expert=4)
def _map_skill_series(s: pd.Series) -> pd.Series:
    mapping = {
        'beginner': 0,
        'intermediate': 1,
        'advanced': 2,
        'expert': 3,
    }
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).clip(0, 4).astype(int)
    s_norm = s.fillna('').astype(str).str.strip().str.lower()
    mapped = s_norm.map(mapping)
    numeric = pd.to_numeric(s_norm, errors='coerce')
    mapped = mapped.where(mapped.notna(), numeric)
    return mapped.fillna(0).clip(0, 4).astype(int)

# Map skill strings to 0–4 integers (unknown/missing -> 0; beginner=1, intermediate=2, advanced=3, expert=4)
def _map_work_series(s: pd.Series) -> pd.Series:
    mapping = {
        'technical': -1,
        'balanced': 0,
        'strategic': 1,
    }
    s_norm = s.fillna('').astype(str).str.strip().str.lower()
    mapped = s_norm.map(mapping)
    numeric = pd.to_numeric(s_norm, errors='coerce')
    mapped = mapped.where(mapped.notna(), numeric)
    return mapped.fillna(0).clip(0, 4).astype(int)

filtered_df.loc[:, 'data_skill'] = _map_skill_series(filtered_df['data_skill'])
filtered_df.loc[:, 'cloud_skill'] = _map_skill_series(filtered_df['cloud_skill'])
filtered_df.loc[:, 'work_preferrence'] = _map_work_series(filtered_df['work_preferrence'])

# For demonstration, limit the number of participants
filtered_df = filtered_df.head(eventCapacity)
print(filtered_df)
print("--- Filtered Registration Data ---")
print(f"Processing {len(filtered_df)} confirmed participants.\n")

# --- 2. Separate Pre-assigned and Unassigned Participants ---

# Participants who need a team
unassigned_df = filtered_df[filtered_df['team_id'] == 'No Team'].copy()

# Participants already in a team
pre_assigned_df = filtered_df[filtered_df['team_id'] != 'No Team'].copy()

# Get a list of unique team IDs that exist, excluding 'No Team'
existing_team_ids = pre_assigned_df['team_id'].unique().tolist()
existingTeams = len(existing_team_ids)
if existingTeams < numTeams:
    existing_team_ids += [f'AutoTeam_{i+1}' for i in range(numTeams - existingTeams)]
print(f"Existing teams found: {existing_team_ids}")

if numTeams == 0:
    print("No pre-existing teams found. Cannot distribute players.")
    sys.exit()

print(f"Found {len(unassigned_df)} participants to assign to {numTeams} existing teams.")

# --- 3. Calculate Initial State of Existing Teams ---
initial_teams_summary = {}
for team_id in existing_team_ids:
    members = pre_assigned_df[pre_assigned_df['team_id'] == team_id]
    initial_teams_summary[team_id] = {
        'initial_size': len(members),
        'initial_data_skill': members['data_skill'].sum(),
        'initial_cloud_skill': members['cloud_skill'].sum(),
        'initial_work_pref': members['work_preferrence'].sum(),
        'initial_members': members['user_id'].tolist()
    }
    
print("\n--- Initial Teams Summary ---")
for team_id, summary in initial_teams_summary.items():
    print(f"Team ID: {team_id}")
    print(f"  Initial Size: {summary['initial_size']}")
    print(f"  Initial Data Skill Total: {summary['initial_data_skill']}")
    print(f"  Initial Cloud Skill Total: {summary['initial_cloud_skill']}")
    print(f"  Initial Work Preference Total: {summary['initial_work_pref']}")
    print()
# --- 4. Setup the CP-SAT Solver Model ---
model = cp_model.CpModel()

# Create a mapping for easier indexing
unassigned_participants = unassigned_df['user_id'].tolist()
team_ids = existing_team_ids

# Decision variables: assignment_vars[p][t] is true if participant p is assigned to team t
assignment_vars = {}
for p_idx, p_id in enumerate(unassigned_participants):
    for t_idx, t_id in enumerate(team_ids):
        assignment_vars[(p_idx, t_idx)] = model.NewBoolVar(f'assign_{p_idx}_to_{t_idx}')

# --- 5. Define Constraints ---

# Constraint 1: Each unassigned participant must be assigned to exactly one team.
for p_idx in range(len(unassigned_participants)):
    model.AddExactlyOne([assignment_vars[(p_idx, t_idx)] for t_idx in range(numTeams)])

# Constraint 2: Team size must not exceed 4.
max_team_size = 4
for t_idx, t_id in enumerate(team_ids):
    initial_size = initial_teams_summary[t_id]['initial_size']
    newly_assigned_count = sum(assignment_vars[(p_idx, t_idx)] for p_idx in range(len(unassigned_participants)))
    model.Add(initial_size + newly_assigned_count <= max_team_size)

# --- 6. Define the Objective Function (Fairness) ---

# Calculate final skill totals for each team
final_data_skills = []
#final_cloud_skills = []
final_work_prefs = []

for t_idx, t_id in enumerate(team_ids):
    # Sum of skills from newly assigned members
    new_data_skill = sum(assignment_vars[(p_idx, t_idx)] * unassigned_df.iloc[p_idx]['data_skill'] for p_idx in range(len(unassigned_participants)))
    #new_cloud_skill = sum(assignment_vars[(p_idx, t_idx)] * unassigned_df.iloc[p_idx]['cloud_skill'] for p_idx in range(len(unassigned_participants)))
    new_work_pref = sum(assignment_vars[(p_idx, t_idx)] * unassigned_df.iloc[p_idx]['work_preferrence'] for p_idx in range(len(unassigned_participants)))

    # Final skill is initial + new
    final_data_skills.append(initial_teams_summary[t_id]['initial_data_skill'] + new_data_skill)
    #final_cloud_skills.append(initial_teams_summary[t_id]['initial_cloud_skill'] + new_cloud_skill)
    #final_work_prefs.append(initial_teams_summary[t_id]['initial_work_pref'] + new_work_pref)

# Calculate target average skills to balance towards
total_participants = len(filtered_df)
avg_data_skill = filtered_df['data_skill'].sum() / numTeams
#avg_cloud_skill = filtered_df['cloud_skill'].sum() / numTeams
# The ideal work preference is 0 (perfectly balanced)
avg_work_pref = 0 
avg_cloud_skill = filtered_df['cloud_skill'].sum() / numTeams

# Create deviation variables for each team and skill
total_deviation = []
for t_idx in range(numTeams):
    # Deviation = abs(final_skill - average_skill)
    data_dev = model.NewIntVar(0, 100, f'data_dev_{t_idx}')
    #cloud_dev = model.NewIntVar(0, 100, f'cloud_dev_{t_idx}')
    work_pref_dev = model.NewIntVar(0, 100, f'work_pref_dev_{t_idx}')

    model.AddAbsEquality(data_dev, final_data_skills[t_idx] - int(avg_data_skill))
    #model.AddAbsEquality(cloud_dev, final_cloud_skills[t_idx] - int(avg_cloud_skill))
    # We want work preference to be close to 0
    #model.AddAbsEquality(work_pref_dev, final_work_prefs[t_idx] - int(avg_work_pref))

    # We can weigh deviations. Let's treat them equally for now.
    total_deviation.extend([data_dev])

# Objective: Minimize the sum of all deviations to make teams as similar as possible.
model.Minimize(sum(total_deviation))

# --- 7. Solve the Model and Display Results ---
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30.0 # Set a time limit
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("\n--- Solution Found! Distributing unassigned participants... ---\n")
    
    # Reconstruct the final teams
    final_teams = {t_id: data['initial_members'][:] for t_id, data in initial_teams_summary.items()}
    changed_teams = {}
    for p_idx, p_id in enumerate(unassigned_participants):
        for t_idx, t_id in enumerate(team_ids):
            if solver.Value(assignment_vars[(p_idx, t_idx)]) == 1:
                final_teams[t_id].append(p_id)
                if t_id not in changed_teams:
                    changed_teams[t_id] = [p_id]
                else:
                    changed_teams[t_id].append(p_id)
                break # Move to the next participant

    # Print the final team compositions
    for i, (team_id, members) in enumerate(final_teams.items()):
        print(f"--- Team {i + 1} (ID: {team_id}) --- | Changed: {'Yes' if team_id in changed_teams.keys() else 'No'}")
        
        if members:
            for user_id in members:
                print(f"  User ID: {user_id} | Data Skill: {filtered_df[filtered_df['user_id'] == user_id]['data_skill'].values[0]} | Cloud Skill: {filtered_df[filtered_df['user_id'] == user_id]['cloud_skill'].values[0]} | Work Preference: {filtered_df[filtered_df['user_id'] == user_id]['work_preferrence'].values[0]} | was changed: {'Yes' if team_id in changed_teams.keys() and user_id in changed_teams[team_id] else 'No'}")
        else:
            print("  No members in this team.")
        print()
        
    # Print a summary DataFrame for verification
    summary_data = []
    for t_id, members in final_teams.items():
        if t_id not in changed_teams.keys():
            continue
        team_df = filtered_df[filtered_df['user_id'].isin(members)]
        summary_data.append({
            'Team ID': t_id,
            'Team Size': len(team_df),
            'Total Data Skill': team_df['data_skill'].sum(),
            'Total Cloud Skill': team_df['cloud_skill'].sum(),
            'Work Pref Balance': team_df['work_preferrence'].sum()
        })
    summary_df = pd.DataFrame(summary_data)
    print("\n--- Final Team Balance Summary ---")
    print(summary_df.to_string())

else:
    print("Could not find a feasible solution. Consider relaxing constraints (e.g., increasing max team size).")