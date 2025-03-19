import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from tqdm import tqdm

# Fixed capability description
CAPABILITY = """The model achieves accuracy 50.0% on the task of "Economics, Market Dynamics, Cost Structures, Utility Theory, Production Possibilities", accuracy 25.0% on the task of "International Law, Human Rights, Jurisdiction, Treaty Obligations, Maritime Delimitation", accuracy 55.0% on the task of "Nutrition, Metabolism, Biochemistry, Health, Exercise", accuracy 35.0% on the task of "Management Functions, Business Philosophy, Decision-Making, Corporate Social Responsibility, Training Methodologies", accuracy 65.0% on the task of "Religious Studies, Sikhism, Buddhism, Jainism, Islam", accuracy 45.0% on the task of "Biology, Evolution, Genetics, Physiology, Cellular Biology", accuracy 35.0% on the task of "Machine Learning, Probability & Statistics, Overfitting & Regularization, Bayesian Networks, Model Architectures (e.g., RoBERTa, ResNeXt)", accuracy 70.0% on the task of "Multidisciplinary, Evaluation, Scientific concepts, Psychological assessments, Societal norms", accuracy 50.0% on the task of "Ethics, Philosophy, Morality, Justice, Autonomy", accuracy 60.0% on the task of "Mathematics, Equations, Geometry, Probability, Functions", accuracy 70.0% on the task of "Philosophy, Ethics, Moral Philosophy, Philosophers, Principles", accuracy 20.0% on the task of "Accounting, Auditing, Tax, Financial Reporting, Business Decisions", accuracy 55.0% on the task of "Morality, Ethics, Behavior, Societal Norms, Hypothetical Scenarios", accuracy 30.0% on the task of "Mathematics, Problem-Solving, Algebra, Geometry, Calculus", accuracy 30.0% on the task of "Virology, Healthcare Systems, Research Ethics, Epidemiology, Immunology", accuracy 30.0% on the task of "Public Relations, Crisis Management, Communication Models, Opinion Leadership, Licensing and Regulations", accuracy 65.0% on the task of "Computer Science, Programming (Python), Data Structures and Algorithms, Digital Systems, Conditional Logic", accuracy 70.0% on the task of "Sociology, Social Structure, Demography, Political Science, Family Dynamics", accuracy 55.0% on the task of "Jurisprudence, Legal Theories, Philosophers, Judicial Decision-Making, Critical Race Theory", accuracy 35.0% on the task of "Politics, Government, Legislation, Elections, Federalism", accuracy 55.0% on the task of "Economics, Trade and Policy, Monetary and Fiscal Policy, Employment and Inflation, GDP and Standard of Living", accuracy 80.0% on the task of "Mathematics, Arithmetic, Problem-solving, Geometry, Fractions", accuracy 60.0% on the task of "Medicine, Physiology, Diagnostics, Exercise Science, Cardiology", accuracy 70.0% on the task of "Genetics, Mutations, Inheritance, Genetic Disorders, DNA/RNA Pairing", accuracy 30.0% on the task of "Statistics, Demographics, Global Trends, Historical Data, Social Issues", accuracy 50.0% on the task of "Legal principles, Scenarios, Property law, Criminal law, Civil procedure", accuracy 65.0% on the task of "Electronics, Oscillators, Binary/Hexadecimal Conversion, Errors and Losses, Oscilloscope", accuracy 80.0% on the task of "Chemistry, Physics, Quantum Mechanics, Molecular Geometry, Buffer Solutions", accuracy 60.0% on the task of "Trivia, Knowledge, Variety, Accuracy, Specificity", accuracy 50.0% on the task of "Marketing/Business Concepts, Consumer Behavior, External Marketing Environment, Pricing Strategies, Buying Situations", accuracy 60.0% on the task of "Computer Science, Algorithms, Memory Management, Debugging, Logic Systems", accuracy 40.0% on the task of "Statistics, Hypothesis Testing, Probability, Distributions, Data Analysis", accuracy 60.0% on the task of "Physics, Kinematics, Dynamics, Optics, Thermodynamics", accuracy 90.0% on the task of "Geography, Migration, Religion, Urbanization, Socioeconomic factors", accuracy 80.0% on the task of "Reproductive Health, Hormonal Regulation, Sexual Behavior, Menstrual Cycle, Contraception", accuracy 80.0% on the task of "Predicate Logic, Truth Tables, Logical Validity, Philosophical Argument, Symbolization", accuracy 55.0% on the task of "Biology, Genetics, Physiology, Ecology, Developmental Biology", accuracy 65.0% on the task of "Astronomy, Solar System, Planets, Comets, Measurement", accuracy 45.0% on the task of "Psychology, Ethics, Memory, Communication, Development", accuracy 70.0% on the task of "Aging, Well-being, Memory, Personality, Retirement", accuracy 60.0% on the task of "Logical Fallacies, Argumentation, Critical Thinking, Reasoning Errors, Philosophy", accuracy 50.0% on the task of "Physics, Problem-solving, Thermodynamics, Relativity, Quantum Mechanics", accuracy 60.0% on the task of "Anatomy, Physiology, Nerve Pathways, Developmental Biology, Clinical Procedures (Dental and General)", accuracy 65.0% on the task of "Psychology, Memory and Learning, Mental Health, Development and Milestones, Psychological Theories", accuracy 85.0% on the task of "Business Ethics, Corporate Social Responsibility, Consumer Behavior, Employment Practices, Civil Society", accuracy 55.0% on the task of "Group Theory, Abstract Algebra, Abelian Group, Homomorphism, Vector Space", accuracy 20.0% on the task of "Time Series Analysis, Stationary Processes, Econometrics Models, Residual Autocorrelation, Model Specification", accuracy 70.0% on the task of "Foreign Policy, U.S. Government, International Relations, Natural Resources, Historical Doctrines", accuracy 70.0% on the task of "Historical Events, Political Movements, Government Policies, Social Reforms, U.S. Presidential Addresses", accuracy 90.0% on the task of "Chemistry, Reactions, Calculations, Principles, Thermodynamics"."""

# Customizable prompt template
PROMPT_TEMPLATE = """Based on the model description and the test sample, predict whether the model can handle test sample by indicating 'Yes' or 'No'.

Model capability: {capability}

Test sample: {question}

Answer:"""

def load_model_and_tokenizer(base_model_path, adapter_path):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=False,
        trust_remote_code=True,
        add_eos_token=True
    )
    
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir="/NS/ssdecl/work"
    )
    
    print("Loading adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    # Get the token ID for "Yes" - matching training
    yes_token_id = tokenizer('Yes', add_special_tokens=False)['input_ids'][-1]
        
    return model, tokenizer, yes_token_id

def get_prediction(model, tokenizer, question, yes_token_id, device):
    # Use the fixed capability and prompt template
    prompt = PROMPT_TEMPLATE.format(capability=CAPABILITY, question=question)
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get the logits for the last token position
        last_token_logits = logits[0, -1, :]
        
        # Get probability for "Yes" token
        yes_prob = torch.softmax(last_token_logits, dim=0)[yes_token_id].item()
        print(yes_prob)
        
        # Use 0.5 as threshold for Yes/No decision
        if yes_prob > 0.5:
            return "Yes", yes_prob
        else:
            return "No", yes_prob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--adapter_path", type=str, default="/NS/ssdecl/work/CapaitLLMRec/test/checkpoint-20")
    parser.add_argument("--test_file", type=str, default="/NS/ssdecl/work/CapaitLLMRec/test/mmlu_test.json")
    parser.add_argument("--output_file", type=str, default="/NS/ssdecl/work/CapaitLLMRec/test/mmlu_results.json")
    args = parser.parse_args()
    
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, yes_token_id = load_model_and_tokenizer(args.base_model, args.adapter_path)
    
    # Load test data
    with open(args.test_file, "r") as f:
        test_data = json.load(f)
    
    # Process each test sample
    results = []
    test_data = test_data[:10]
    for sample in tqdm(test_data, desc="Processing test samples"):
        prediction, confidence = get_prediction(
            model,
            tokenizer,
            sample["question"],
            yes_token_id,
            device
        )
        
        results.append({
            "question": sample["question"],
            "prediction": prediction,
            "confidence": confidence
        })
    
    # Save results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary statistics
    yes_count = sum(1 for r in results if r["prediction"] == "Yes")
    no_count = sum(1 for r in results if r["prediction"] == "No")
    
    print("\nResults Summary:")
    print(f"Total samples: {len(results)}")
    print(f"Yes predictions: {yes_count}")
    print(f"No predictions: {no_count}")
    print(f"Average confidence: {sum(r['confidence'] for r in results)/len(results):.3f}")

if __name__ == "__main__":
    main() 