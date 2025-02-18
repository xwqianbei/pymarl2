from FlagEmbedding import BGEM3FlagModel


def text_embedding(text, model_path):
    model = BGEM3FlagModel(model, model_path, use_fp16 = True)
    text_embedding = model.encode(text)['dense_vecs']
    return text_embedding

