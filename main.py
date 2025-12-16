from sklearn.datasets import fetch_kddcup99
import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt 
import numpy as np

data = fetch_openml(name='titanic', version=1, as_frame=True, parser='pandas')
df = data.frame

features = list(df.columns)
print("Available features:", features)
selected_features = ['pclass', 'survived','sex', 'age','sibsp','parch',]
print("Selected features: ", selected_features)

# 1. Create the Age Groups (Bins)
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 120]
labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']

# 2. Prepare the data
df_plot = df.copy()

# Use pd.cut to group ages, then convert to string to avoid the Float vs Str error
df_plot['age_group'] = pd.cut(df_plot['age'], bins=bins, labels=labels, right=False)
df_plot['age_group'] = df_plot['age_group'].astype(str).replace('nan', 'Unknown')

# 3. Plotting
fig, axs = plt.subplots(1, len(selected_features), figsize=(20, 5), squeeze=False)
axs_flat = axs.flatten()

for ax, f in zip(axs_flat, selected_features):
    # Use the grouped age column if we are looking at 'age'
    col_name = 'age_group' if f == 'age' else f
    
    # Get counts and convert index to string immediately to stop the comparison error
    counts = df_plot[col_name].value_counts(dropna=False)
    counts.index = counts.index.astype(str)
    
    # Sort them so they appear in order (0-10, 10-20, etc.)
    # We move 'Unknown' or 'nan' to the very end manually
    sorted_labels = sorted([i for i in counts.index if i not in ['Unknown', 'nan']])
    if 'Unknown' in counts.index: sorted_labels.append('Unknown')
    if 'nan' in counts.index: sorted_labels.append('nan')
    
    counts = counts.reindex(sorted_labels)
    
    # Color coding
    colors = ['orange' if x in ['Unknown', 'nan'] else 'skyblue' for x in counts.index]
    
    ax.bar(counts.index, counts.values, color=colors, edgecolor='black')
    ax.set_title(f"Dist of {f.capitalize()}")
    plt.setp(ax.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()


# I picked the survival feature as it likely the most important one to show corealtions in the data, for example you can see that the people in their 20-30 had better chances thna the rest, or the fact female passngers had a better chance.



# 1. Setup Age Binning
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 120]
labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']

df_plot = df.copy()
df_plot['age'] = pd.cut(df_plot['age'], bins=bins, labels=labels, right=False)
df_plot['age'] = df_plot['age'].astype(str).replace('nan', 'Unknown')

# 2. Define the features you want to look at
# We exclude 'survived' from the list because it's our target
features_to_check = ['pclass', 'sex', 'age', 'sibsp', 'embarked']

# 3. Create the figure
fig, axs = plt.subplots(1, len(features_to_check), figsize=(25, 6), squeeze=False)
axs_flat = axs.flatten()

for ax, f in zip(axs_flat, features_to_check):
    # Create the counts table (Crosstab)
    # Rows = Feature (Age, Sex, etc.), Columns = Survived (0, 1)
    ct = pd.crosstab(df_plot[f], df_plot['survived'])
    
    # Optional: Sort the age groups so they don't appear randomly
    if f == 'age':
        ordered = [l for l in labels if l in ct.index] + ['Unknown']
        ct = ct.reindex([i for i in ordered if i in ct.index])

    # 4. Plot as Stacked Bars
    ct.plot(kind='bar', stacked=True, ax=ax, color=['#e74c3c', '#2ecc71'], edgecolor='black')
    
    ax.set_title(f"Survival by {f.capitalize()}", fontsize=14)
    ax.set_xlabel(f)
    ax.set_ylabel("Count")
    ax.legend(['Died', 'Lived'], title='Status', prop={'size': 8})
    plt.setp(ax.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()

df_scatter = df.copy()
# Ensure survival is integer
df_scatter['survived'] = pd.to_numeric(df_scatter['survived'], errors='coerce').fillna(0).astype(int)

# Create a list of features to plot, excluding 'survived' as it's the y-axis
features_to_plot_against_survival = [f for f in selected_features if f != 'survived']

fig, axs = plt.subplots(1, len(features_to_plot_against_survival), figsize=(25, 6), squeeze=False)
axs_flat = axs.flatten()

for ax, f in zip(axs_flat, features_to_plot_against_survival): # Use the new list here
    temp = df_scatter[[f, 'survived']].dropna().copy()

    # --- DATA CONVERSION ---
    if f == 'sex':
        temp['x_vals'] = temp[f].map({'female': 0, 'male': 1})
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Female', 'Male'])
    else:
        # Now f will never be 'survived', so temp[f] will always be a Series.
        temp['x_vals'] = pd.to_numeric(temp[f], errors='coerce')

    temp = temp.dropna(subset=['x_vals'])

    # --- COUNTER LOGIC (For Pclass and Sex) ---
    if f in ['pclass', 'sex']:
        # Get unique categories (1,2,3 or Female,Male)
        categories = temp[f].unique()
        for cat in categories:
            # Count Survivors and Dead for this specific group
            group = temp[temp[f] == cat]
            alive = len(group[group['survived'] == 1])
            dead = len(group[group['survived'] == 0])

            # Determine X position for the text
            x_pos = 0 if cat == 'female' else (1 if cat == 'male' else cat)

            # Place the text: Green count at the top, Red count at the bottom
            ax.text(x_pos, 1.2, f"Lived: {alive}", color='#27ae60', ha='center', fontweight='bold')
            ax.text(x_pos, -0.25, f"Died: {dead}", color='#c0392b', ha='center', fontweight='bold')

    # --- COLOR MAP ---
    point_colors = ['#2ecc71' if s == 1 else '#e74c3c' for s in temp['survived']]

    # --- JITTER ---
    y_jitter = temp['survived'].astype(float) + np.random.uniform(-0.12, 0.12, size=len(temp))
    x_jitter = temp['x_vals'].astype(float)
    if f in ['pclass', 'sex', 'sibsp']:
        x_jitter = x_jitter + np.random.uniform(-0.1, 0.1, size=len(temp))

    # --- PLOT ---
    ax.scatter(x_jitter, y_jitter, c=point_colors, alpha=0.4, s=40, edgecolor='white', lw=0.5)

    ax.set_xlabel(f.capitalize(), fontsize=12)
    ax.set_ylim(-0.5, 1.5) # Extra space for the counters
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Died (0)', 'Lived (1)'])
    ax.set_title(f"{f.capitalize()} vs Survival", pad=20)
plt.savefig('correlation_plot.png')
plt.tight_layout()
plt.show()


#ai was used to create this scatter plot and histogram and to show correclations clearly in the scatter plot. 
