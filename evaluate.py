import argparse
import json

def calculate_metrics(pred_data, truth_data):
    total_truth_answers = 0
    total_pred_answers = 0
    correct_preds = 0
    
    for q_id, q_info in truth_data.items():
        truth_ans = set(q_info.get("answer", []))
        total_truth_answers += len(truth_ans)
        
        if q_id in pred_data:
            pred_ans = set(pred_data[q_id].get("answer", []))
            total_pred_answers += len(pred_ans)
            
            # Đếm số kết quả dự đoán đúng
            correct_preds += len(truth_ans.intersection(pred_ans))
            
    precision = correct_preds / total_pred_answers if total_pred_answers > 0 else 0
    recall = correct_preds / total_truth_answers if total_truth_answers > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Evaluate Legal IR results")
    parser.add_argument("--pred", type=str, required=True, help="Đường dẫn file submission.json")
    parser.add_argument("--ground_truth", type=str, required=True, help="Đường dẫn file ground truth (VD: data/warmup.json)")
    args = parser.parse_args()
    
    print(f"Đọc file kết quả dự đoán từ {args.pred}...")
    with open(args.pred, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
        
    print(f"Đọc file ground truth từ {args.ground_truth}...")
    with open(args.ground_truth, 'r', encoding='utf-8') as f:
        truth_data = json.load(f)
        
    p, r, f1 = calculate_metrics(pred_data, truth_data)
    
    print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-Score:  {f1:.4f}")

if __name__ == "__main__":
    main()
