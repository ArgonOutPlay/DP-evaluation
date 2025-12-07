import json
import argparse

#program to generate no gt dataset in txt from .json gt dataset
def main():
    parser = argparse.ArgumentParser(description="Evaluator for RAG.")
    parser.add_argument("--input_path",
                type=str,
                help="Path to question file (.json)")
    parser.add_argument("--output_path",
                type=str,
                help="Path to question file (.json)")
    
    args = parser.parse_args()
    in_path = args.input_path
    out_path = args.output_path

    questions = []
    #read data with GT
    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entry in data:
            if 'question' in entry:
                questions.append(entry['question'])

    #return TXT question withou GT
    with open(out_path, 'w', encoding='utf-8') as o:
        for q in questions:
            o.write(q + '\n')

if __name__ == "__main__":
    main()