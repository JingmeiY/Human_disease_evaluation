import json
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score
from collections import Counter
from scipy import stats
import sys
import os

# Add parent directory to path for importing utility functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility import load_json_file

def load_comparison_data():
    deepseek_path = "./Result/LLM_Judge/deepseek-reasoner/model_comparison_results_parsed_samples.json"
    gemini_path = "./Result/LLM_Judge/gemini-2.5-pro/model_comparison_results_parsed_samples.json"
    
    deepseek_data = load_json_file(deepseek_path)
    gemini_data = load_json_file(gemini_path)
    
    return deepseek_data, gemini_data

def calculate_win_rates(data):
    """Calculate win rates for Q1 and Q2"""
    q1_winners = Counter(item['q1_winner'] for item in data)
    q2_winners = Counter(item['q2_winner'] for item in data)
    
    total = len(data)
    
    return {
        'q1_counts': q1_winners,
        'q2_counts': q2_winners,
        'q1_rates': {winner: (count/total)*100 for winner, count in q1_winners.items()},
        'q2_rates': {winner: (count/total)*100 for winner, count in q2_winners.items()},
        'total': total
    }

def calculate_cohen_kappa_ci(y1, y2, confidence=0.95):
    """Calculate Cohen's kappa with confidence interval"""
    kappa = cohen_kappa_score(y1, y2)
    n = len(y1)
    se_kappa = np.sqrt((1 - kappa**2) / n)
    z_critical = stats.norm.ppf(1 - (1-confidence)/2)
    ci_lower = kappa - z_critical * se_kappa
    ci_upper = kappa + z_critical * se_kappa
    return kappa, ci_lower, ci_upper

def calculate_agreement_rate(y1, y2):
    """Calculate simple agreement rate"""
    return sum(1 for a, b in zip(y1, y2) if a == b) / len(y1) * 100

def perform_chi_square_test(dist1, dist2, categories):
    """Perform chi-square test"""
    contingency = [
        [dist1.get(cat, 0) for cat in categories],
        [dist2.get(cat, 0) for cat in categories]
    ]
    _, p_value, _, _ = chi2_contingency(contingency)
    return p_value

def create_sample_lookup(data_list):
    """Convert list to dictionary with sample_id as key"""
    return {item['sample_id']: item for item in data_list}

def analyze_inter_judge_agreement(deepseek_data, gemini_data):
    """Calculate inter-judge agreement metrics"""
    deepseek_lookup = create_sample_lookup(deepseek_data)
    gemini_lookup = create_sample_lookup(gemini_data)
    
    common_ids = list(set(deepseek_lookup.keys()) & set(gemini_lookup.keys()))
    
    # Extract Q1 and Q2 winners for common samples
    ds_q1 = [deepseek_lookup[sid]['q1_winner'] for sid in common_ids]
    gm_q1 = [gemini_lookup[sid]['q1_winner'] for sid in common_ids]
    ds_q2 = [deepseek_lookup[sid]['q2_winner'] for sid in common_ids]
    gm_q2 = [gemini_lookup[sid]['q2_winner'] for sid in common_ids]
    
    # Get all possible categories
    all_q1_cats = list(set(ds_q1 + gm_q1))
    all_q2_cats = list(set(ds_q2 + gm_q2))
    
    # Calculate metrics
    kappa_q1, ci_lower_q1, ci_upper_q1 = calculate_cohen_kappa_ci(ds_q1, gm_q1)
    kappa_q2, ci_lower_q2, ci_upper_q2 = calculate_cohen_kappa_ci(ds_q2, gm_q2)
    
    agreement_q1 = calculate_agreement_rate(ds_q1, gm_q1)
    agreement_q2 = calculate_agreement_rate(ds_q2, gm_q2)
    
    # Chi-square tests
    ds_q1_dist = Counter(ds_q1)
    gm_q1_dist = Counter(gm_q1)
    ds_q2_dist = Counter(ds_q2)
    gm_q2_dist = Counter(gm_q2)
    
    chi2_q1 = perform_chi_square_test(ds_q1_dist, gm_q1_dist, all_q1_cats)
    chi2_q2 = perform_chi_square_test(ds_q2_dist, gm_q2_dist, all_q2_cats)
    
    return {
        'q1': {'kappa': kappa_q1, 'ci_lower': ci_lower_q1, 'ci_upper': ci_upper_q1, 
               'agreement': agreement_q1, 'chi2_p': chi2_q1},
        'q2': {'kappa': kappa_q2, 'ci_lower': ci_lower_q2, 'ci_upper': ci_upper_q2, 
               'agreement': agreement_q2, 'chi2_p': chi2_q2},
        'n_samples': len(common_ids)
    }

def analyze_by_domain(data, domains):
    """Analyze win rates by domain"""
    domain_results = {}
    for domain in domains:
        domain_data = [item for item in data if item['domain'] == domain]
        if domain_data:
            domain_results[domain] = calculate_win_rates(domain_data)
    return domain_results

def write_win_rate_table(f, title, ds_rates, gm_rates, models):
    """Write structured win rate table"""
    f.write(f"{title}\n")
    f.write("-" * 90 + "\n")
    f.write(f"{'Judge':<10} {'Question':<12} {'Model':<20} {'Count':<8} {'%':<8} ")
    f.write("-" * 90 + "\n")
    
    for judge_name, rates in [('DeepSeek', ds_rates), ('Gemini', gm_rates)]:
        for q_name, q_key in [('Q1 (Risk)', 'q1'), ('Q2 (Reason)', 'q2')]:
            counts = rates[f'{q_key}_counts']
            percentages = rates[f'{q_key}_rates']
            
            model_names = list(models)

                
            for i, model in enumerate(model_names):
                count = counts.get(model, 0)
                pct = percentages.get(model, 0)
                f.write(f"{judge_name:<10} {q_name:<12} {model:<20} {count:<8} {pct:<7.2f}%\n")
                judge_name = ""  # Only show judge name once
                q_name = ""      # Only show question name once
    f.write("\n")

def main():
    deepseek_data, gemini_data = load_comparison_data()
    
    # Calculate win rates for both judges
    ds_rates = calculate_win_rates(deepseek_data)
    gm_rates = calculate_win_rates(gemini_data)
    
    # Calculate inter-judge agreement
    agreement = analyze_inter_judge_agreement(deepseek_data, gemini_data)
    
    # Get unique domains and models
    domains = list(set(item['domain'] for item in deepseek_data))
    all_models = set()
    for item in deepseek_data:
        all_models.add(item['q1_winner'])
        all_models.add(item['q2_winner'])
    all_models = sorted(list(all_models))
    
    # Domain-specific analysis
    ds_domain_results = analyze_by_domain(deepseek_data, domains)
    gm_domain_results = analyze_by_domain(gemini_data, domains)
    
    # Output results
    with open("./model_comparison_analysis_results.txt", "w") as f:
        f.write("MODEL COMPARISON ANALYSIS RESULTS\n")
        f.write("=" * 90 + "\n\n")
        
        # Overall Win Rate Table
        write_win_rate_table(f, "OVERALL WIN RATES", ds_rates, gm_rates, all_models)
        
        # Inter-Judge Agreement Table
        f.write("INTER-JUDGE AGREEMENT\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Question':<15} {'Cohen κ':<10} {'95% CI':<18} {'Agreement%':<12} {'Chi² p-value':<12}\n")
        f.write("-" * 80 + "\n")
        
        for q, q_name in [('q1', 'Q1 (Risk)'), ('q2', 'Q2 (Reasoning)')]:
            kappa = agreement[q]['kappa']
            ci_lower = agreement[q]['ci_lower']
            ci_upper = agreement[q]['ci_upper']
            agree_rate = agreement[q]['agreement']
            chi2_p = agreement[q]['chi2_p']
            f.write(f"{q_name:<15} {kappa:<9.3f} [{ci_lower:.3f}-{ci_upper:.3f}] {agree_rate:<11.2f}% {chi2_p:<11.4f}\n")
        
        f.write(f"\nTotal samples analyzed: {agreement['n_samples']}\n\n")
        
        # Domain-specific analysis
        f.write("DOMAIN-SPECIFIC ANALYSIS\n")
        f.write("=" * 90 + "\n\n")
        
        for domain in sorted(domains):
            if domain in ds_domain_results and domain in gm_domain_results:
                ds_domain = ds_domain_results[domain]
                gm_domain = gm_domain_results[domain]
                
                write_win_rate_table(f, f"Domain: {domain}", ds_domain, gm_domain, all_models)
    
    print("Results saved to model_comparison_analysis_results.txt")

if __name__ == "__main__":
    main()
