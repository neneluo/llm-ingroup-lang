system_prompt_ingroup_generator = """
You are a group member of an online community "{community}". You will be given a post from the community and a word; your task is to create a lexical innovation to express the word. Do not use the original form.

The linguistic conventions to create a lexical innovation are:
(1) Compounding: create a new word by combining two existing words, e.g., combining "smart" and "phone" to form "smartphone".
(2) Abbreviation: create a new word by abbreviation, e.g., abbreviating "thank you" into "ty".
(3) Repurposing: repurpose an existing word to a different meaning, e.g., use "bug" to refer to "defect" in computer software.

Respond with the following format:
Thinking: describe your thinking process.
Answer: answer with the lexical innovation (only respond with one word).
"""

system_prompt_ingroup_guesser = """
You are a group member of an online community "{community}". You will be given a post from the community and a word that is innovated by another member in the community; your task is to come up with the most likely common English word that conveys the same meaning as the given word.

Respond with the following format:
Thinking: describe your thinking process.
Answer: answer with the word (only respond with one word).
"""

user_prompt = """
The post is: {post}
The word is: {word}
"""

evaluation_free_explanation = """
Explain in your own words the meaning of the word "{term}" from subreddit "{community}".
Example context: {context}
"""

evaluation_free_explanation_one_shot = """
Explain in your own words the meaning of the word "{term}" from subreddit "{community}".
Example context: {context}

Below is an input-output examples:
Input:
Term: link
Subreddit: zelda
Output:
Answer: the main character in the game of the legend of zelda.
"""

evaluation_free_explanation_three_shot = """
Explain in your own words the meaning of the word "{term}" from subreddit "{community}".
Example context: {context}

Below are some input-output examples:
Input:
Term: link
Subreddit: zelda
Output:
Answer: the main character in the game of the legend of zelda.

Input:
Term: transformer
Subreddit: deeplearning
Output:
Answer: a specific deep learning model architecture, proposed in paper "attention is all you need".

Input:
Term: wall
Subreddit: gameofthrones
Output:
Answer: a colossal fortification along the border.
"""

evaluation_free_explanation_llm_judge = """
Evaluate the quality of the explanation produced by the models for the target word. The evaluation result is: good, bad or borderline. 

Good is when the model-generated explanation aligns well with the given explanation. Bad is when the model-generated explanation misalign with the given explanation. Borderline is when the model-generated explanation grasps some extent of intended meaning, but is either imprecise or overly explain and included unmentioned details.

Target word: {term}
Explanation: {description}
Model-generated Explanation: {model_generation}

Evaluation (only output good, bad or borderline):
"""

evaluation_word_sense_diambiguation = """
Read a post from subreddit "{community}": "{context}". 

Choose the correct explanation of the word "{term}" from the options: "{options}". 

Provide as output only the correct explanation.
"""

evaluation_named_entity_recognition_wnut = """
Read a post from subreddit "{community}": "{context}". 

Choose the correct Name Entity Type of the word "{term}" from the options: PERSON, GROUP, LOCATION, CORPORATION, PRODUCT, CREATIVE_WORK or O. 

Description of Name Entity Type:
PERSON: Names of people (e.g. Virginia Wade). Don't mark people that don't have their own name. Fictional people can be included, as long as they're referred to by name (e.g. Harry Potter).
GROUP: Names of groups (e.g. Nirvana, San Diego Padres). Don't mark groups that don't have a specific, unique name, or companies (which should be marked corporation).
LOCATION: Names that are locations (e.g. France). Don't mark locations that don't have their own name. Fictional locations can be included, as long as they're referred to by name (e.g. Hogwarts).
CORPORATION: Names of corporations (e.g. Google). Don't mark locations that don't have their own name.
PRODUCT: Name of products (e.g. iPhone). Don't mark products that don't have their own name. Fictional products can be included, as long as they're referred to by name (e.g. Everlasting Gobstopper). It's got to be something you can touch, and it's got to be the official name.
CREATIVE_WORK: Names of creative works (e.g. Bohemian Rhapsody). The work should be created by a human, and referred to by its specific name.
O: Not an name entity

Provide as output only the Name Entity Type.
"""

training_prompt_synonym = """
You will be given a term from a subreddit. Respond with a synonym of the term.

Term: {term}
Subreddit: {community}
Term usage example: {context}
"""
