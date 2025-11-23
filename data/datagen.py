import numpy as np
import pandas as pd

np.random.seed(42)
n_colleges = 500

# Unique rank distribution
overall_rank = np.arange(1, n_colleges + 1)
np.random.shuffle(overall_rank)
rank_norm = (overall_rank - 1) / (n_colleges - 1)

# Faculty features
faculty_phd = np.random.randint(50, 250, n_colleges) * (1 - 0.2 * rank_norm) + np.random.normal(0, 5, n_colleges)
faculty_phd = np.clip(faculty_phd, 5, 250).astype(int)

faculty_other = np.random.randint(10, 80, n_colleges)
faculty_other = np.clip(faculty_other, 0, 100).astype(int)

frac_research_phd = np.random.uniform(0.3, 0.9, n_colleges)
frac_research_other = np.random.uniform(0.1, 0.6, n_colleges)

# === Normalized research budget (in millions) ===
research_budget = np.random.uniform(1, 5000, n_colleges) * (1 - 0.4 * rank_norm)
research_budget = np.clip(research_budget, 1, 5000)   # 1M to 5000M
research_budget_million = research_budget  # keep in millions

centres_of_excellence_count = np.random.randint(0, 15, n_colleges)

teaching_load_per_faculty = np.random.uniform(2, 15, n_colleges) + 2 * rank_norm
teaching_load_per_faculty = np.clip(teaching_load_per_faculty, 2, 15)

diversity_index = np.random.uniform(0.0, 1.0, n_colleges) * (1 - 0.3 * rank_norm)
diversity_index = np.clip(diversity_index, 0, 1)

industry_collaborations = np.random.randint(0, 100, n_colleges)

student_count = np.random.randint(100, 10000, n_colleges)
num_courses = np.random.randint(10, 300, n_colleges) * (1 - 0.1 * rank_norm)

# Research FTE
research_FTE = faculty_phd * frac_research_phd + faculty_other * frac_research_other

# === Softer nonlinear research outputs ===
papers_t1 = (
    0.1 * np.log1p(research_FTE) ** 2
    + 0.05 * np.sqrt(research_budget_million)
    + 1.0 * centres_of_excellence_count
    - 0.25 * teaching_load_per_faculty
    + np.random.normal(0, 6, n_colleges)
)
papers_t1 = np.clip(papers_t1, 10, 5000).astype(int)

citation_percentile_avg = (
    0.4
    + 0.08 * diversity_index
    + 0.12 * np.log10(research_budget_million)
    - 0.15 * rank_norm
    + np.random.normal(0, 0.03, n_colleges)
)
citation_percentile_avg = np.clip(citation_percentile_avg, 0, 1)

# === More stable cost-efficiency output ===
impact_score = papers_t1 * citation_percentile_avg
cost_efficiency = research_budget_million / (impact_score + 5)   # avoid huge spikes

# === Final dataset ===
df = pd.DataFrame({
    "college_id": [f"C{i+1:03d}" for i in range(n_colleges)],
    "faculty_phd": faculty_phd,
    "faculty_other": faculty_other,
    "frac_research_phd": frac_research_phd,
    "frac_research_other": frac_research_other,
    "research_budget_million": research_budget_million,   # normalized
    "centres_of_excellence_count": centres_of_excellence_count,
    "teaching_load_per_faculty": teaching_load_per_faculty,
    "diversity_index": diversity_index,
    "industry_collaborations": industry_collaborations,
    "student_count": student_count,
    "num_courses": num_courses.astype(int),
    "papers_t1": papers_t1,
    "citation_percentile_avg": citation_percentile_avg,
    "overall_rank": overall_rank,
    "cost_efficiency": cost_efficiency
})

df.to_csv("department_research_data.csv", index=False)
print("New dataset generated with normalized scaling and weaker correlations.")
print(df.head())
