import evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_text_generation_metrics(
    predictions,
    references,
    use_mae=False,
    use_mse=False,
    use_exact_match=False,
    use_bleu=False,
    use_rouge=False,
    use_meteor=False,
    use_bertscore=False):
    """
    Evaluate text generation metrics for predictions vs references.
    
    Args:
        predictions: List of predicted values/texts
        references: List of reference/ground truth values/texts
        use_mae: Whether to calculate Mean Absolute Error (for numeric values)
        use_mse: Whether to calculate Mean Squared Error (for numeric values)
        use_exact_match: Whether to calculate exact match score
        use_bleu: Whether to calculate BLEU score
        use_rouge: Whether to calculate ROUGE scores
        use_meteor: Whether to calculate METEOR score
        use_bertscore: Whether to calculate BERTScore
    
    Returns:
        Dictionary containing the calculated metrics
    """
    scores = {}

    # --- L1 Loss MAE ---
    if use_mae:
        scores["mae"] = mean_absolute_error(references, predictions)

    # --- L2 Loss MSE ---
    if use_mse:
        scores["mse"] = mean_squared_error(references, predictions)

    # --- Exact Match ---
    if use_exact_match:
        exact_match = evaluate.load("exact_match")
        # Ensure predictions and references are strings and strip leading/trailing whitespace
        predictions = [str(p).strip() for p in predictions]
        references = [str(r).strip() for r in references]
        exact_match_results = exact_match.compute(predictions=predictions, references=references)
        scores["exact_match"] = exact_match_results["exact_match"]

    # --- BLEU ---
    if use_bleu:
        bleu = evaluate.load("bleu")
        bleu_results = bleu.compute(
            predictions=predictions,
            references=references
        )
        scores["bleu"] = bleu_results["bleu"]

    # --- ROUGE ---
    if use_rouge:
        rouge = evaluate.load("rouge")
        rouge_results = rouge.compute(
            predictions=predictions,
            references=references,
            use_aggregator=True
        )
        scores["rouge1"] = rouge_results["rouge1"]
        scores["rouge2"] = rouge_results["rouge2"]
        scores["rougeL"] = rouge_results["rougeL"]
        scores["rougeLsum"] = rouge_results["rougeLsum"]

    # --- METEOR ---
    if use_meteor:
        meteor = evaluate.load("meteor")
        meteor_results = meteor.compute(predictions=predictions, references=references)
        scores["meteor"] = meteor_results["meteor"]

    # --- BERTScore ---
    if use_bertscore:
        bertscore = evaluate.load("bertscore")
        bertscore_results = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en"
        )
        scores["bertscore_precision"] = sum(bertscore_results["precision"]) / len(bertscore_results["precision"])
        scores["bertscore_recall"]    = sum(bertscore_results["recall"])    / len(bertscore_results["recall"])
        scores["bertscore_f1"]        = sum(bertscore_results["f1"])        / len(bertscore_results["f1"])

    # Scale percentages to 0-100 range, keep MAE/MSE as-is
    return {key: round(value * 100, 2) if key not in ["mae", "mse"] else round(value, 2) for key, value in scores.items()}