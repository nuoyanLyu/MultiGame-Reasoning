import json
from sentence_transformers import SentenceTransformer
from tqdm import trange
import random

root_path = '/data1/lvnuoyan/llm_model'
model = SentenceTransformer(f"{root_path}/all-MiniLM-L6-v2")

wordlists = json.load(open('ragen/env/undercover/wordlists.json'))

wordshuffle = []

def calc_similarity(word1, word2):
    embeddings = model.encode([word1, word2])
    return model.similarity(embeddings, embeddings)[0][1].item()


for i in trange(len(wordlists)):
    word1, word2 = wordlists[i]
    word1 = word1.lower()
    word2 = word2.lower()
    if 'blas ' in word1 or 'blas ' in word2:
        # 发现了大量出现的无意义词汇 'blas + 数字'，删除
        continue
    if word1 == word2:
        # 部分单词长得完全一样，跳过
        continue
    # 使用sbert计算词汇相似度，存储
    similarity = calc_similarity(word1, word2)
    if [word1, word2, similarity] not in wordshuffle:
        wordshuffle.append([word1, word2, similarity])
# 保留语义相似度最高的3000个词组
wordshuffle = sorted(wordshuffle, key=lambda x: x[2], reverse=True)[:3000]
# 只保留前两个单词，内部两个词的次序打乱
wordshuffle1 = []
for w in wordshuffle:
    wordshuffle1.append([w[0], w[1]])
    random.shuffle(wordshuffle1[-1])
# 保存
json.dump(wordshuffle1, open('ragen/env/undercover/wordshuffle.json', 'w'), ensure_ascii=False, indent=2)
