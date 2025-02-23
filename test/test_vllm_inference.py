from vllm import LLM, SamplingParams

# Define model path (update this to match your directory)
MODEL_PATH = "/outputs/checkpoint-60/"

# Initialize vLLM with your fine-tuned model
llm = LLM(model=MODEL_PATH)

# Define sampling parameters (adjust as needed)
sampling_params = SamplingParams(
    temperature=0.7,  # Controls randomness
    top_p=0.9,        # Nucleus sampling
    max_tokens=200,   # Max tokens to generate
)

# Run inference
prompt = "Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?"
outputs = llm.generate([prompt], sampling_params)

# Print results
for output in outputs:
    print("Generated Text:", output.outputs[0].text)
