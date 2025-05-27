import torch
import numpy as np
from typing import Dict, Optional, List
from transformers import EvalPrediction, BertTokenizerFast
from torchmetrics import Accuracy
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
import nltk

# --- Download nltk punkt if not already downloaded (needed by rouge_scorer) ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading nltk punkt tokenizer...")
    nltk.download('punkt', quiet=True)
# --- End NLTK check ---

# --- NLVR2 指标计算 ---
nlvr2_accuracy_calculator = Accuracy(task="binary")

def compute_nlvr2_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    计算 NLVR2 任务的指标 (Accuracy)。

    Args:
        eval_pred (EvalPrediction): 包含 predictions (logits) 和 label_ids (binary labels) 的对象。

    Returns:
        Dict[str, float]: 包含准确率的字典。
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    if logits is None or labels is None:
        print("Warning: Missing predictions or labels in compute_nlvr2_metrics.")
        return {}

    # 确保是 Tensor
    logits_tensor = torch.from_numpy(logits) if isinstance(logits, np.ndarray) else logits
    labels_tensor = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels
    labels_tensor = labels_tensor.long() # 确保标签是 Long 类型

    # 移动到设备
    nlvr2_accuracy_calculator = nlvr2_accuracy_calculator.to(logits_tensor.device)

    # 从 logits 获取预测类别 (假设 logits 形状为 [N, 2])
    preds = torch.argmax(logits_tensor, dim=-1)

    # 更新指标状态
    nlvr2_accuracy_calculator.update(preds, labels_tensor)

    # 计算最终准确率
    try:
        accuracy = nlvr2_accuracy_calculator.compute()
        return {"accuracy": accuracy.item()}
    except Exception as e:
        print(f"Error computing NLVR2 accuracy: {e}")
        return {}

# --- Retrieval 指标计算 ---
def compute_retrieval_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Retrieval 任务的占位符指标计算函数。
    实际的 Recall@K 计算通常在 Trainer 外部完成。
    """
    return {} # 返回空字典，表示不由 Trainer 计算主要指标

# --- Caption/Generation 指标计算 ---
def compute_caption_metrics(eval_pred: EvalPrediction, tokenizer: Optional[BertTokenizerFast] = None) -> Dict[str, float]:
    """
    计算生成任务 (如 Captioning, Generative VQA) 的指标，包括 BLEU-4 和 ROUGE-L。

    Args:
        eval_pred (EvalPrediction): 包含 predictions (generated token IDs) 和 label_ids (reference token IDs) 的对象。
                                    label_ids 应该包含 -100 用于填充。
        tokenizer (Optional[BertTokenizerFast]): 用于解码 token IDs。

    Returns:
        Dict[str, float]: 包含生成指标 (bleu, rougeL) 的字典。
    """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    if predictions is None or labels is None:
        print("Warning: Missing predictions or labels in compute_caption_metrics.")
        return {}
    if tokenizer is None:
        print("Warning: Tokenizer not provided to compute_caption_metrics. Cannot decode sequences.")
        return {}

    # 确保 predictions 是 numpy 数组 (Hugging Face Trainer 通常输出 numpy)
    if isinstance(predictions, tuple): # Handle cases where predictions might be nested
        predictions = predictions[0]
    if not isinstance(predictions, np.ndarray):
        try:
            predictions = np.array(predictions)
        except Exception as e:
            print(f"Error converting predictions to numpy array: {e}")
            return {}

    # 将 -100 替换为 pad_token_id 以便解码
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

    # 解码生成的 token IDs 和参考 token IDs
    try:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # --- ROUGE 需要进行 NLTK 分词预处理 ---
        decoded_preds_rouge = decoded_preds
        decoded_labels_rouge = decoded_labels

    except Exception as e:
        print(f"Error decoding tokens in compute_caption_metrics: {e}")
        return {}

    # --- 计算 BLEU-4 ---
    try:
        bleu_calculator = BLEU()
        bleu_score = bleu_calculator.corpus_score(decoded_preds, [decoded_labels]).score
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        bleu_score = 0.0

    # --- 计算 ROUGE-L ---
    try:
        rouge_calculator = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rougeL_scores = []
        for pred, label in zip(decoded_preds_rouge, decoded_labels_rouge):
            scores = rouge_calculator.score(label, pred)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        avg_rougeL = np.mean(rougeL_scores) * 100 if rougeL_scores else 0.0
    except Exception as e:
        print(f"Error calculating ROUGE-L score: {e}")
        avg_rougeL = 0.0

    # --- 返回计算出的指标 ---
    return {"bleu": bleu_score, "rougeL": avg_rougeL}
