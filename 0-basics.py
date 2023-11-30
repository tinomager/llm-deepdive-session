import dotenv
import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np  

#load env
ENV = dotenv.dotenv_values(".env")

#init OpenAI service for Azure
openai.api_version = "2023-05-15"
openai.api_base = ENV['AZURE_OAI_BASE_URL']
openai.api_key = ENV['AZURE_OAI_KEY']
openai.api_type = "azure"

#some sample statments
sentence1 = "Who is Barrack Obama?"
sentence2 = "He is an American politician who served as the 44th President of the United States from 2009 to 2017 and was the first African American to hold the office."
sentence3 = "An apple is a fruit from the Malus domestica tree, commonly consumed worldwide for its taste and nutritional value."
input_sentences = [sentence1, sentence2, sentence3]

#query service
response = openai.Embedding.create(
  input=input_sentences,
  engine=ENV["AZURE_OAI_EMBEDDINGMODELL"]
)

#do some math magic
embedding1 = np.array(response['data'][0]['embedding'])
embedding2 = np.array(response['data'][1]['embedding'])
embedding3 = np.array(response['data'][2]['embedding'])

print(f'Embedding vector 1 of text "{sentence1}": {embedding1}')
print(f'Embedding vector 2 of text "{sentence2}": {embedding2}')
print(f'Embedding vector 3 of text "{sentence3}": {embedding3}')

similarity12 = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
similarity13 = cosine_similarity(embedding1.reshape(1, -1), embedding3.reshape(1, -1))[0][0]
similarity23 = cosine_similarity(embedding2.reshape(1, -1), embedding3.reshape(1, -1))[0][0]

print(f'Similarity of 1 & 2: {similarity12}')
print(f'Similarity of 1 & 3: {similarity13}')
print(f'Similarity of 2 & 3: {similarity23}')
