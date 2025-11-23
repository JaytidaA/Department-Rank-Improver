# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from libRION.libRION import RionOptimiser


def define_objective_functions():
    """
    Define three objective functions for RION-Opt.
    All functions take a pandas Series (row) and return a float.
    """
    
    def objective_1_productivity(individual: pd.Series) -> float:
        """
        Objective 1: Minimize -(papers_t1 / research_FTE)
        Higher productivity is better, so we negate for minimization.
        """
        papers_t1 = individual.get('papers_t1', 10)
        faculty_phd = individual.get('faculty_phd', 20)
        faculty_other = individual.get('faculty_other', 5)
        frac_research_phd = individual.get('frac_research_phd', 0.5)
        frac_research_other = individual.get('frac_research_other', 0.2)
        
        research_FTE = faculty_phd * frac_research_phd + faculty_other * frac_research_other
        research_FTE = max(research_FTE, 1.0)
        
        productivity = papers_t1 / research_FTE
        return -productivity
    
    def objective_2_quality(individual: pd.Series) -> float:
        """
        Objective 2: Minimize -(citation_percentile_avg * papers_t1 / research_FTE)
        Higher quality is better, so we negate for minimization.
        """
        papers_t1 = individual.get('papers_t1', 10)
        citation_percentile_avg = individual.get('citation_percentile_avg', 0.5)
        faculty_phd = individual.get('faculty_phd', 20)
        faculty_other = individual.get('faculty_other', 5)
        frac_research_phd = individual.get('frac_research_phd', 0.5)
        frac_research_other = individual.get('frac_research_other', 0.2)
        
        research_FTE = faculty_phd * frac_research_phd + faculty_other * frac_research_other
        research_FTE = max(research_FTE, 1.0)
        
        # Impact score: weighted by citation percentile
        impact_score = papers_t1 * citation_percentile_avg
        quality = impact_score / research_FTE
        return -quality
    
    def objective_3_cost_efficiency(individual: pd.Series) -> float:
        """
        Objective 3: Minimize (research_budget / impact_score)
        Lower cost per impact is better.
        """
        research_budget = individual.get('research_budget', 1000000)
        papers_t1 = individual.get('papers_t1', 10)
        citation_percentile_avg = individual.get('citation_percentile_avg', 0.5)
        
        impact_score = papers_t1 * citation_percentile_avg
        if impact_score < 1.0:
            return 1e5
        cost_per_impact = research_budget / (impact_score + 1e-6)
        return cost_per_impact
    
    return [objective_1_productivity, objective_2_quality, objective_3_cost_efficiency]


def define_constraints():
    """
    Define constraints as [[min, max], [min, max], ...]
    One constraint pair for each column in the dataset (excluding college_id).
    
    Columns in order:
    1. faculty_phd
    2. faculty_other
    3. frac_research_phd
    4. frac_research_other
    5. research_budget
    6. centres_of_excellence_count
    7. teaching_load_per_faculty
    8. diversity_index
    9. industry_collaborations
    10. student_count
    11. num_courses
    12. papers_t1
    13. citation_percentile_avg
    14. overall_rank
    """
    
    constraints = [
        [5, 250],             # faculty_phd
        [0, 100],             # faculty_other
        [0.1, 0.9],           # frac_research_phd
        [0.05, 0.8],          # frac_research_other
        [1e5, 5e9],           # research_budget (100K to 5B INR)
        [0, 15],              # centres_of_excellence_count
        [2.0, 15.0],          # teaching_load_per_faculty
        [0.0, 1.0],           # diversity_index
        [0, 100],             # industry_collaborations
        [100, 10000],         # student_count
        [10, 300],            # num_courses
        [0, 5000],            # papers_t1
        [0.0, 1.0],           # citation_percentile_avg
        [1, 200],             # overall_rank
    ]
    
    return constraints


def main():
    """Main execution function"""
    
    print("="*70)
    print("RION-Opt: Multi-Objective Optimization")
    print("="*70)
    
    # Load dataset
    data_path = "./data/department_research_data.csv"
    print(f"\nLoading dataset: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        
        # Drop college_id column if it exists
        if 'college_id' in df.columns:
            df = df.drop(columns=['college_id'])
            print(f" Dropped 'college_id' column")
        
        print(f" Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
        print(f"\nFirst row sample:")
        print(df.head(1).to_string())
    except Exception as e:
        print(f" Error loading dataset: {e}")
        return
    
    # Define objective functions
    objective_funcs = define_objective_functions()
    print(f"\n Defined {len(objective_funcs)} objective functions:")
    print(f"  1. Research Productivity (papers/FTE)")
    print(f"  2. Research Quality (impact/FTE)")
    print(f"  3. Cost Efficiency (budget/impact)")
    
    # Define constraints
    constraints = define_constraints()
    print(f"\nDefined {len(constraints)} constraints matching dataset columns")
    
    # Initialize RION Optimizer
    print("\n" + "="*70)
    print("Initializing RION Optimizer")
    print("="*70)
    
    optimizer = RionOptimiser(
        generations=40,
        pop_size=50,
        inertia=0.5,
        self_confidence=1.5,
        cross_confidence=1.5,
        pso_iterations=10
    )
    
    print("Configuration:")
    print(f"  Generations:     40")
    print(f"  Population Size: 50")
    print(f"  PSO Inertia:     0.5")
    print(f"  PSO Cognitive:   1.5")
    print(f"  PSO Social:      1.5")
    print(f"  PSO Iterations:  10")
    
    # Run optimization
    print("\n" + "="*70)
    print("Running Optimization...")
    print("="*70 + "\n")
    
    optimized_population = optimizer.optimise(
        df=df,
        optimiser_funcs=objective_funcs,
        constraints=constraints
    )
    
    # Save results
    print("\n" + "="*70)
    print("Optimization Complete!")
    print("="*70)
    
    if optimized_population is not None:
        output_file = "./results/optimized_population.csv"
        import os
        os.makedirs("./results/", exist_ok=True)
        optimized_population.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
        print(f"  Final population size: {len(optimized_population)}")
        print(f"\nFirst few optimized solutions:")
        print(optimized_population.head())
        
        # Calculate and display objective values for top solutions
        print(f"\nObjective values for top 5 solutions:")
        print(f"{'Solution':<10s} {'Productivity':<15s} {'Quality':<15s} {'Cost/Impact':<15s}")
        print("-"*55)
        for idx in range(min(5, len(optimized_population))):
            row = optimized_population.iloc[idx]
            obj_vals = [func(row) for func in objective_funcs]
            print(f"{idx+1:<10d} {-obj_vals[0]:<15.3f} {-obj_vals[1]:<15.3f} {obj_vals[2]:<15.2e}")
    
    print("\n" + "="*70)

        # ===== Additional Pareto Visualization & Metrics =====

    print("\nCalculating Pareto Front and Metrics...")

    # Compute objective values matrix for final population
    objective_values = np.array([[func(ind) for func in objective_funcs] 
                                 for _, ind in optimized_population.iterrows()])

    # Extract Pareto front (front 0)
    fronts, ranks = optimizer._fast_non_dominated_sort(optimized_population)
    pareto_front_indices = fronts[0]
    pareto_front = objective_values[pareto_front_indices]

    # Convert maximization objectives (1 & 2) to negative for plotting
    plot_values = pareto_front.copy()
    plot_values[:,0] = -plot_values[:,0]  # productivity max
    plot_values[:,1] = -plot_values[:,1]  # quality max

    # -------- 3D Pareto Scatter Plot --------
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(-objective_values[:,0], -objective_values[:,1], objective_values[:,2], alpha=0.4, label="All Solutions")
    ax.scatter(plot_values[:,0], plot_values[:,1], plot_values[:,2], color='red', label="Pareto Front")

    ax.set_title("Pareto Front (3 Objectives)")
    ax.set_xlabel("Research Productivity (Max)")
    ax.set_ylabel("Research Quality (Max)")
    ax.set_zlabel("Cost Efficiency (Min)")

    plt.legend()
    plt.show()

    # -------- Diversity Metric --------
    def diversity_metric(pareto):
        pareto = pareto[pareto[:,0].argsort()]
        distances = np.linalg.norm(pareto[1:] - pareto[:-1], axis=1)
        return np.std(distances)

    diversity = diversity_metric(pareto_front)
    print(f"\nDiversity Spread Metric: {diversity:.4f}")

    # -------- Hypervolume --------
    ref_point = np.max(plot_values, axis=0) + 1.0
    def hypervolume(front, ref):
        hv = 0
        front = front[front[:,0].argsort()]
        for i in range(len(front) - 1):
            hv += abs(front[i+1][0] - front[i][0]) * \
                  abs(ref[1] - front[i][1]) * \
                  abs(ref[2] - front[i][2])
        return hv

    hv = hypervolume(plot_values, ref_point)
    print(f"Hypervolume: {hv:.4f}")



if __name__ == "__main__":
    main()
