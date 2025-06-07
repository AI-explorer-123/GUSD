import torch
import json
from tqdm import tqdm

print("Loadind data...")
nodes = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/node.json'))
movie_id = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/movie_id.json'))
print("Loading finished.")

prompts = {}
i = 0
for node in tqdm(nodes):
    if node['type'] == 'review':
        prompt = f"""
        Background:
        A spoiler is an element of a disseminated summary or description of a media narrative that reveals significant plot elements, with the implication that the experience of discovering the plot naturally, as the creator intended it, has been robbed of its full effect.There are three types of spoilers: short spoilers, long spoilers, and thematic spoilers. Short spoilers reveal the plot ending in a very brief and less detailed manner, without any summary or explanation of themes in the story, typically spanning one to three sentences. Long spoilers usually provide more context and range between two and five sentences. They provide a summary and reveal the ending of a story. Lastly, thematic spoilers reveal a story's unifying theme as well as provifing a synopsis of the plot and revealing the ending. They range from three to six sentences in length. The general consensus is that there are only negative effects of spoilers. However, research shows that it is short and long spoilers that can cause negative effects, while thematic spoilers generally have a possitive effect.
        
        Task:
        Based on the information of a movie and a review of it, determine whether the review is a spoiler. 
        i.e. whether it reveals the major plot of the movie so that will affect the experience of people who have not seen the film. \
        You should give me your prediction and give a detailed explanation of your answer. 

        Input:
        The summary of the movie plot: {nodes[movie_id[node['movie_id']]]['feature']['semantic']}
        The content of the review: {node['feature']['semantic']} 

        Requirements:
        1. You are supposed to think in a step by step manner. 
        Firstly, analyze the summary of the movie to grasp the major plot. 
        Then, consider the content of the review and summarize what does the review mainly talks about. \
        Many of the comments only mentions the acting, filming, special effects, costumes, music, etc., \
        which can not be considered as a spoiler. In fact, only a small number of reviews are real spoilers.
        Finally, give your prediction and explain it. Notice that your prediction must be consistent with your explanation.
        2. Please return me your answer in json format with two keys: prediction and explanation. \
        Your prediction should be "True" or "False", where "True" means you think the review is a spoiler and "False" means the opposite.
        For example: {{"prediction": False, "explanation": "I think the review doesn't seem to be a spoiler because ..."}}

        Output:
        Here is my answer:
        """
        prompts[i] = prompt
        i += 1
        
with open('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/llm/prompt.json', 'w') as f:
    json.dump(prompts, f, indent=4)