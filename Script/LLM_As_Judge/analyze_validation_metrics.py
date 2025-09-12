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
from utility import load_json_file, save_json_file

def load_validation_data():
    deepseek_path = "./Result/LLM_Judge/deepseek-reasoner/reference_validation_results_parsed_samples.json"
    gemini_path = "./Result/LLM_Judge/gemini-2.5-pro/reference_validation_results_parsed_samples.json"
    
    deepseek_data = load_json_file(deepseek_path)
    gemini_data = load_json_file(gemini_path)
    
    return deepseek_data, gemini_data

def process_q2_mapping(score):
    return 'sound' if score in ['B', 'C'] else 'poor'

def calculate_cohen_kappa_ci(y1, y2, confidence=0.95):
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

def create_contingency_table(dist1, dist2, categories):
    """Create contingency table for chi-square test"""
    return [
        [dist1.get(cat, 0) for cat in categories],
        [dist2.get(cat, 0) for cat in categories]
    ]

def perform_chi_square_test(dist1, dist2, categories):
    """Perform chi-square test and return p-value"""
    contingency = create_contingency_table(dist1, dist2, categories)
    _, p_value, _, _ = chi2_contingency(contingency)
    return p_value

def create_sample_lookup(data_list):
    """Convert list of samples to dictionary with sample_id as key"""
    return {item['sample_id']: item for item in data_list}

def analyze_by_domain(deepseek_data, gemini_data, domains):
    """Analyze validation score distributions by domain"""
    domain_results = {}
    
    for domain in domains:
        ds_domain = [item for item in deepseek_data if item['domain'] == domain]
        gm_domain = [item for item in gemini_data if item['domain'] == domain]
        
        if ds_domain and gm_domain:
            # Extract Q1 and Q2 scores for this domain
            ds_q1 = [item['q1_normalized'] for item in ds_domain]
            gm_q1 = [item['q1_normalized'] for item in gm_domain]
            ds_q2 = [process_q2_mapping(item['q2_normalized']) for item in ds_domain]
            gm_q2 = [process_q2_mapping(item['q2_normalized']) for item in gm_domain]
            
            # Calculate distributions
            ds_q1_dist = Counter(ds_q1)
            gm_q1_dist = Counter(gm_q1)
            ds_q2_dist = Counter(ds_q2)
            gm_q2_dist = Counter(gm_q2)
            
            domain_results[domain] = {
                'q1': {'ds_dist': ds_q1_dist, 'gm_dist': gm_q1_dist, 'ds_total': len(ds_q1), 'gm_total': len(gm_q1)},
                'q2': {'ds_dist': ds_q2_dist, 'gm_dist': gm_q2_dist, 'ds_total': len(ds_q2), 'gm_total': len(gm_q2)}
            }
    
    return domain_results

def write_score_distribution_table(f, title, ds_dist, gm_dist, categories, total_ds, total_gm):
    """Write structured score distribution table"""
    f.write(f"{title}\n")
    f.write("-" * 90 + "\n")
    f.write(f"{'Judge':<10} {'Category':<15} {'Count':<8} {'%':<8}\n")
    f.write("-" * 90 + "\n")
    
    # DeepSeek distributions
    for i, cat in enumerate(categories):
        count = ds_dist.get(cat, 0)
        pct = (count/total_ds)*100 if total_ds > 0 else 0
        judge_name = "DeepSeek" if i == 0 else ""
        f.write(f"{judge_name:<10} {cat:<15} {count:<8} {pct:<7.2f}%\n")
    
    f.write("-" * 40 + "\n")
    
    # Gemini distributions  
    for i, cat in enumerate(categories):
        count = gm_dist.get(cat, 0)
        pct = (count/total_gm)*100 if total_gm > 0 else 0
        judge_name = "Gemini" if i == 0 else ""
        f.write(f"{judge_name:<10} {cat:<15} {count:<8} {pct:<7.2f}%\n")
    
    f.write("\n")

def write_domain_breakdown_table(f, title, q1_data, q2_data):
    """Write domain-specific score breakdown table"""
    f.write(f"{title}\n")
    f.write("-" * 90 + "\n")
    f.write(f"{'Judge':<10} {'Question':<12} {'Category':<15} {'Count':<8} {'%':<8}\n")
    f.write("-" * 90 + "\n")
    
    # Q1 breakdown
    for judge_name, dist, total in [('DeepSeek', q1_data['ds_dist'], q1_data['ds_total']), 
                                   ('Gemini', q1_data['gm_dist'], q1_data['gm_total'])]:
        for i, cat in enumerate(['A', 'B', 'C']):
            count = dist.get(cat, 0)
            pct = (count/total)*100 if total > 0 else 0
            q_name = "Q1 (Score)" if i == 0 else ""
            judge_display = judge_name if i == 0 else ""
            f.write(f"{judge_display:<10} {q_name:<12} {cat:<15} {count:<8} {pct:<7.2f}%\n")
        if judge_name == 'DeepSeek':
            f.write("-" * 60 + "\n")
    
    f.write("-" * 90 + "\n")
    
    # Q2 breakdown
    for judge_name, dist, total in [('DeepSeek', q2_data['ds_dist'], q2_data['ds_total']), 
                                   ('Gemini', q2_data['gm_dist'], q2_data['gm_total'])]:
        for i, cat in enumerate(['poor', 'sound']):
            count = dist.get(cat, 0)
            pct = (count/total)*100 if total > 0 else 0
            q_name = "Q2 (Reason)" if i == 0 else ""
            judge_display = judge_name if i == 0 else ""
            f.write(f"{judge_display:<10} {q_name:<12} {cat:<15} {count:<8} {pct:<7.2f}%\n")
        if judge_name == 'DeepSeek':
            f.write("-" * 60 + "\n")
    
    f.write("\n")

def main():
    deepseek_data, gemini_data = load_validation_data()
    
    # Extract Q1 and Q2 scores from new data structure
    deepseek_q1 = [item['q1_normalized'] for item in deepseek_data]
    gemini_q1 = [item['q1_normalized'] for item in gemini_data]
    
    ds_q1_dist = Counter(deepseek_q1)
    gm_q1_dist = Counter(gemini_q1)
    
    # Q2 distributions (mapped)
    deepseek_q2 = [process_q2_mapping(item['q2_normalized']) for item in deepseek_data]
    gemini_q2 = [process_q2_mapping(item['q2_normalized']) for item in gemini_data]
    
    ds_q2_dist = Counter(deepseek_q2)
    gm_q2_dist = Counter(gemini_q2)
    
    # Chi-square tests using reusable function
    p_q1 = perform_chi_square_test(ds_q1_dist, gm_q1_dist, ['A', 'B', 'C'])
    p_q2 = perform_chi_square_test(ds_q2_dist, gm_q2_dist, ['poor', 'sound'])
    
    # Cohen's Kappa - create lookups for matching samples
    deepseek_lookup = create_sample_lookup(deepseek_data)
    gemini_lookup = create_sample_lookup(gemini_data)
    
    common_ids = list(set(deepseek_lookup.keys()) & set(gemini_lookup.keys()))
    ds_q1_common = [deepseek_lookup[sample_id]['q1_normalized'] for sample_id in common_ids]
    gm_q1_common = [gemini_lookup[sample_id]['q1_normalized'] for sample_id in common_ids]
    ds_q2_common = [process_q2_mapping(deepseek_lookup[sample_id]['q2_normalized']) for sample_id in common_ids]
    gm_q2_common = [process_q2_mapping(gemini_lookup[sample_id]['q2_normalized']) for sample_id in common_ids]
    
    kappa_q1, ci_lower_q1, ci_upper_q1 = calculate_cohen_kappa_ci(ds_q1_common, gm_q1_common)
    kappa_q2, ci_lower_q2, ci_upper_q2 = calculate_cohen_kappa_ci(ds_q2_common, gm_q2_common)
    
    # Calculate agreement rates
    agreement_q1 = calculate_agreement_rate(ds_q1_common, gm_q1_common)
    agreement_q2 = calculate_agreement_rate(ds_q2_common, gm_q2_common)
    
    # Get unique domains for domain analysis
    domains = list(set(item['domain'] for item in deepseek_data))
    domain_results = analyze_by_domain(deepseek_data, gemini_data, domains)
    
    # Output to file
    with open("./validation_analysis_results.txt", "w") as f:
        f.write("VALIDATION ANALYSIS RESULTS\n")
        f.write("=" * 90 + "\n\n")
        
        # Overall Score Distribution Tables
        write_score_distribution_table(f, "Q1 SCORE APPROPRIATENESS DISTRIBUTION", 
                                     ds_q1_dist, gm_q1_dist, ['A', 'B', 'C'], 
                                     len(deepseek_q1), len(gemini_q1))
        
        write_score_distribution_table(f, "Q2 REASONING QUALITY DISTRIBUTION (MAPPED)", 
                                     ds_q2_dist, gm_q2_dist, ['poor', 'sound'], 
                                     len(deepseek_q2), len(gemini_q2))
        
        # Inter-Judge Agreement Table
        f.write("INTER-JUDGE AGREEMENT\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Question':<15} {'Cohen κ':<10} {'95% CI':<18} {'Agreement%':<12} {'Chi² p-value':<12}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Q1 (Score)':<15} {kappa_q1:<9.3f} [{ci_lower_q1:.3f}-{ci_upper_q1:.3f}] {agreement_q1:<11.2f}% {p_q1:<11.4f}\n")
        f.write(f"{'Q2 (Reasoning)':<15} {kappa_q2:<9.3f} [{ci_lower_q2:.3f}-{ci_upper_q2:.3f}] {agreement_q2:<11.2f}% {p_q2:<11.4f}\n")
        f.write(f"\nTotal samples analyzed: {len(common_ids)}\n\n")
        
        # Domain-specific analysis - showing score breakdowns, not agreement
        f.write("DOMAIN-SPECIFIC ANALYSIS\n")
        f.write("=" * 90 + "\n\n")
        
        for domain in sorted(domains):
            if domain in domain_results:
                result = domain_results[domain]
                write_domain_breakdown_table(f, f"Domain: {domain}", result['q1'], result['q2'])
    
    print("Results saved to validation_analysis_results.txt")

if __name__ == "__main__":
    main()


