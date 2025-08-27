from openai import OpenAI
import json
from tqdm import trange

openrouter_keys = json.load(open('ragen/env/api-keys.json'))['openrouter']
# deepseek_key = os.environ.get('DEEPSEEK_KEY')
if not openrouter_keys:
    print('No openrouter keys found, please set it in the key-files api-keys.json')
    exit(1)

prompt = """
You are designing word pairs for the game "Who is the Spy" (also known as "Undercover").

Each pair must meet the following requirements:
1. Both words must be **English nouns**.
2. Each pair should contain two words:
   - One for normal players ("civilian word").
   - One for the spy ("spy word").
3. Words should be semantically close but still distinct (e.g., "Hotel" vs "Hostel").
4. Avoid overly obvious differences (e.g., "Cat" vs "Car").
5. Avoid words that are too close (e.g., "Air" vs "Airs"). 
6. Avoid words that have multiple unrelated meanings (e.g., "coach", "bank").
7. Avoid duplicates across the list.
8. Words should be common and concrete enough to describe in normal conversation without mentioning the word directly.

Output requirements:
- Generate exactly **50 pairs**.
- Each pair must be in the format:
  <civilian-word>,<spy-word>
- Do not include numbering, explanations, or any other text besides the word pairs.
"""

wordlists = []

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=openrouter_keys[0],
)

for i in trange(100):
    completion = client.chat.completions.create(
        model="google/gemini-2.5-flash-lite",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ]
            }
        ]
    )
    content = completion.choices[0].message.content
    # 通过第一个 [ 和最后一个 ] 定位合法的json字符串位置
    # begin = content.index('[')
    # end = content.rindex(']')
    # content = content[begin:end+1]
    # print(content)
    lines = content.split('\n')
    for line in lines:
        try:
            word1, word2 = line.split(',')
            word1 = word1.strip().lower()
            word2 = word2.strip().lower()
            wordlists.append([word1, word2])
        except:
            print('wrong lines, skip')
            # print('wrong outputs, skip this response')

print(len(wordlists))
print('delete duplicate')
# 为了删除重复元素进行了排序，但是后续玩游戏的时候应该随机选一个作为卧底词
# 否则后一个词一定比前一个字幕位置靠后，不合理
wordlists = list(set([tuple(sorted(pair)) for pair in wordlists]))
print(len(wordlists))
with open('ragen/env/undercover/wordlists.json', 'w') as f:
    json.dump(wordlists, f)
