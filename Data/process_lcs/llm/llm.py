import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    '/data3/whr/zhk/huggingface/Aquila2-7b', trust_remote_code=True, offload_folder='offload', device_map='cuda:1', torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(
    '/data3/whr/zhk/huggingface/Aquila2-7b')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# lm_model = AutoModel.from_pretrained(
#     '/data3/whr/zhk/huggingface/bge-large-en-v1.5').to('cuda:1')
# lm_model.eval()
# lm_tokenizer = AutoTokenizer.from_pretrained(
#     '/data3/whr/zhk/huggingface/bge-large-en-v1.5')
print("Loading finished.")

print("Loading data..")
users = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Dataset/LCS/user.json'))
nodes = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/node.json'))
movie_id = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/movie_id.json'))
print("Loading finished.")

all_expl = []
# expl_embed = torch.tensor([]).to('cuda:1')

sum = 0
for user_id in users.keys():
    information = ''
    for id, review in enumerate(users[user_id]['reviews']):
        information += f"""
                Review {id+1}:
                    The summary of the movie plot: {nodes[movie_id[review['movie_id']]]
                    ['feature']['semantic']}
                    The content of the review: {review['content']} 
    """
    prompt = f"""
    Background:
        A spoiler is an element of a disseminated summary or description of movies that reveals significant plot elements and twists, with the implication that the experience of discovering the plot naturally, as the creator intended it, has been robbed of its full effect.
        
    Task:
        Based on the information of a user and all the reviews of the user, determine whether the user tends to post spoiler reviews. 
        i.e. whether the user tends to post reviews which reveal the major plot of the relevant movie so that will affect the experience of people who have not seen the film. 
        You should give me your judgement and give a detailed explanation of your answer.
        
    Requirements:
        1. Let's think step-by-step.
        2. Many of the comments only mentions the acting, filming, special effects, costumes, music, etc., which can not be considered as a spoiler.
        3. Some users will tell if this review is a spoiler when posting reviews, which is definitely a spoiler.
        4. Note that some user will express their feelings about the movie without talking about the plot, which can't be judged as a spoiler. However, if a user talks about the ending or twisting of the movie, they are definitely spoilers.
        5. Finally, give your judgement and explain it in detail. Notice that your judgement must be consistent with your explanation.
        6. You should answer in the way of the two given examples, and don't answer somethng irrelevant.  

    Example 1:
        Input:
            All the reviews of the user1:
                Review 1:
                    The summary of the movie plot: 'A small-town beauty pageant turns deadly as it becomes clear that someone will go to any lengths to win.'
                    The content of the review: 'This could have been really awesome, but in the end, Drop Dead Gorgeous is the kind of movie that happens when people who don't really know anything about a subject decide to make a satire about it. Given that the focus of the movie, beauty pageants, is so deserving of ridicule, the script for this film is woefully unfunny or interesting. Further, the writers do a poor imitation of Fargo by casting every character as an imbecile with a Minnesota accent -- it just doesn't work. I've seen worse films -- but not too many more. I laughed out loud maybe twice, giggled on six occasions, and looked at my watch over two dozen times, especially in the last 30 minutes.'
            
        Output:
            Here is my answer: Based on the information provided, it appears that the user tends to post spoiler reviews. While the review includes general criticism of the film, it also reveals specific plot-related details and hints at potential surprises, which could spoil the experience for others who haven't seen the movie.In this review, the first paragraph provides a brief summary of the movie's plot, which sets the premise of a small-town beauty pageant turning deadly. While this summary alone may not be considered a major spoiler, the subsequent content of the review reveals specific criticisms and references plot developments, potentially spoiling key surprises for those who haven't seen the film.
    Example 2:
        Input:
            All the reviews of the user2:
                Review 1:
                The summary of the movie plot: "When her rich oilman father is killed, Bingo, raised in the wilds of South America, inherits the company. Her guardians Ben and Howard send her to New York for civilizing but on the way she meets Andy, wonderful in every way but wealth. He can't live off her money, he says, as he turns to Marjory. Uncivilized Bingo, who hits anyone she disagrees with, shoots Andy in the arm. Now it's okay for him to marry her."
                The content of the review: 'Fun to watch an early "talking" but the acting is marginal and the fight scene laughable.  Fun seeing Montgomery and Crawford in the earliest part of their careers.  But you can tell western electric was still playing around with sound trying to get the levels right.  Sometimes background music overpowered dialogue.', "When her rich oilman father is killed, Bingo, raised in the wilds of South America, inherits the company. Her guardians Ben and Howard send her to New York for civilizing but on the way she meets Andy, wonderful in every way but wealth. He can't live off her money, he says, as he turns to Marjory. Uncivilized Bingo, who hits anyone she disagrees with, shoots Andy in the arm. Now it's okay for him to marry her.'
        Output:
            Here is my answer: Based on the information provided, it appears that the user tends to post non-spoiler reviews. The review does touch upon some plot points, but they do not reveal critical plot twists or spoil major surprises. The focus of the review is primarily on aspects like acting, filming, sound, and music, which are not plot-related and do not spoil the narrative.  

    Input:
        All the reviews of the user3:
            {information}   
                
    Output:
        Here is my answer:
"""

    model_input = tokenizer(prompt, padding=True, truncation=True,
                            return_tensors="pt").to('cuda:1')
    generated_ids = model.generate(
        **model_input, max_new_tokens=100, do_sample=True)
    response = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    index = response.find('user3')
    explanation = response[index:].strip()
    index = explanation.rfind('my answer:')
    explanation = explanation[index+10:].strip()
    try:
        index2 = explanation.find('Input')
        explanation = explanation[:index2].strip()
    except:
        pass
    all_expl.append(explanation)

    # tokens = lm_tokenizer(prompt, padding=True,
    #                    truncation=True, return_tensors='pt').to('cuda:1')
    # out = lm_model(**tokens)[0][:, 0]
    # out = torch.nn.functional.normalize(out, p=2, dim=1)
    # expl_embed = torch.cat((expl_embed, out), dim=0)
    # print(expl_embed.shape)
    sum += 1
    print(sum)
    if sum >= 10000:
        break

print('Saving data...')
with open('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/llm/expl.json', 'w') as f:
    json.dump(all_expl, f)

# torch.save(expl_embed,'/data3/whr/zhk/Spoiler_Detection/Data/process/llm/expl.pt')
