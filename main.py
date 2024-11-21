import ollama



resp = ollama.chat(model='llama3.2-vision',
            messages=[{
                'role':'user',
                'content':'这张图片里有什么?',
                'images':['C:/Users/Administrator/Pictures/b_0c7f360b2bb8c2f54b4d90a68cb3a4df.jpg']
            }])

print(resp['message']['content'])