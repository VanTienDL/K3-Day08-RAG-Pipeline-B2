import json
import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.task10_generation import generate_with_citation
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--full", action="store_true", help="Run on all 16 questions instead of just 3")
    args = parser.parse_args()

    dataset_path = project_root / "group_project" / "evaluation" / "golden_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Nếu có cờ --full, chạy toàn bộ. Nếu không, chỉ lấy 3 câu đầu để test.
    if args.full:
        test_data = data
        print("Mode: FULL (16 questions)")
    else:
        test_data = data[:3]
        print("Mode: TEST (3 questions) - Use --full to run all")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    print(f"Generating answers for {len(test_data)} questions...")
    for item in test_data:
        query = item["question"]
        print(f"\nQ: {query}")
        result = generate_with_citation(query)
        
        ans = result["answer"]
        print(f"A: {ans[:100]}...")
        
        chunks = [src.get("content", "") for src in result.get("sources", [])]
        
        questions.append(query)
        answers.append(ans)
        contexts.append(chunks)
        ground_truths.append(item["expected_answer"])

    # Tạo dataset (cả key v0.1 và v0.2 để backward/forward compatible)
    dataset_dict = {
        "question": questions,
        "user_input": questions,
        "answer": answers,
        "response": answers,
        "contexts": contexts,
        "retrieved_contexts": contexts,
        "ground_truth": ground_truths,
        "reference": ground_truths
    }
    eval_dataset = Dataset.from_dict(dataset_dict)
    
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
    # For evaluate, Ragas typically works better with a proper LLM model. Llama might be weak for evaluation, 
    # but we will use the default config.
    model_name = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    
    llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
    embeddings = OpenAIEmbeddings()
    
    # Ragas 0.2.x wrapper
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    except ImportError:
        ragas_llm = llm
        ragas_embeddings = embeddings
        
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    
    print("\nBắt đầu chạy RAGAS evaluation...")
    results = evaluate(
        eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False
    )
    
    print("\n=== KẾT QUẢ ĐÁNH GIÁ (3 CÂU) ===")
    print(results)
    
    df = results.to_pandas()
    # Handle both column name variations
    col_q = 'user_input' if 'user_input' in df.columns else 'question'
    cols_to_show = [col_q, 'faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    
    print("\nChi tiết từng câu:")
    print(df[cols_to_show])
    
    # Lưu kết quả ra file tạm
    output_path = project_root / "group_project" / "evaluation" / "temp_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nĐã lưu kết quả chi tiết ra: {output_path}")

if __name__ == "__main__":
    main()
