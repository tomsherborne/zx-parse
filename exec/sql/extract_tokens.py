"""
Extract the tokens from an AllenNLP predict phase output
"""
import sys, json

INPUT_QUERIES=sys.argv[1]
OUTPUT_FILE=sys.argv[2]

with open(INPUT_QUERIES, 'r') as input_fh, open(OUTPUT_FILE, 'w') as out_fh:
	for line in input_fh:
		prediction = json.loads(line)
		if "predicted_tokens" in prediction:
			predicted_tokens = prediction["predicted_tokens"]
		elif "sql_predicted_tokens" in prediction:
			predicted_tokens = prediction["sql_predicted_tokens"]
		else:
			raise ValueError(f"Prediction keys did not contain valid prediction: keys were {prediction.keys()}")
		predicted_str = " ".join(predicted_tokens) + "\n"
		out_fh.write(predicted_str)
