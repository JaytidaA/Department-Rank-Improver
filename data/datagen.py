import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

n_colleges = 521

# Generate synthetic data
faculty_phd = np.random.randint(10, 150, n_colleges)
faculty_other = np.random.randint(5, 80, n_colleges)
frac_research_phd = np.random.uniform(0.3, 0.8, n_colleges)
frac_research_other = np.random.uniform(0.1, 0.5, n_colleges)
research_budget = 10 ** np.random.uniform(5, 9, n_colleges)  # INR (1e5–1e9)
centres_of_excellence_count = np.random.randint(0, 10, n_colleges)
teaching_load_per_faculty = np.random.uniform(6, 16, n_colleges)
diversity_index = np.random.uniform(0.2, 1.0, n_colleges)
industry_collaborations = np.random.randint(0, 15, n_colleges)
student_count = np.random.randint(200, 2500, n_colleges)
num_courses = np.random.randint(10, 120, n_colleges)

# Derived: research_FTE
research_FTE = faculty_phd * frac_research_phd + faculty_other * frac_research_other

# Simulate Tier-1 papers and citation percentile based on realistic correlations
papers_t1 = (
    0.05 * research_FTE
    + 0.000002 * research_budget
    + 0.8 * centres_of_excellence_count
    - 0.3 * teaching_load_per_faculty
    + np.random.normal(0, 2, n_colleges)
)
papers_t1 = np.clip(papers_t1, 0, None).astype(int)

citation_percentile_avg = (
    0.5
    + 0.05 * (centres_of_excellence_count / 10)
    + 0.1 * (diversity_index - 0.5)
    + 0.05 * np.log10(research_budget / 1e5)
    - 0.02 * (teaching_load_per_faculty / 15)
)
citation_percentile_avg = np.clip(citation_percentile_avg, 0, 1)

# Overall rank (lower = better)
overall_rank = (
    1000 / (1 + papers_t1)
    + 100 * (1 - citation_percentile_avg)
    + np.random.uniform(0, 50, n_colleges)
).astype(int)

# Assemble dataframe
df = pd.DataFrame({
    "college_id": [f"C{i+1:03d}" for i in range(n_colleges)],
    "faculty_phd": faculty_phd,
    "faculty_other": faculty_other,
    "frac_research_phd": frac_research_phd,
    "frac_research_other": frac_research_other,
    "research_budget": research_budget,
    "centres_of_excellence_count": centres_of_excellence_count,
    "teaching_load_per_faculty": teaching_load_per_faculty,
    "diversity_index": diversity_index,
    "industry_collaborations": industry_collaborations,
    "student_count": student_count,
    "num_courses": num_courses,
    "papers_t1": papers_t1,
    "citation_percentile_avg": citation_percentile_avg,
    "overall_rank": overall_rank
})

# Save CSV
df.to_csv("department_research_data.csv", index=False)
print("CSV generated: department_research_data.csv")
print(df.head())
