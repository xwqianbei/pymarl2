from FlagEmbedding import BGEM3FlagModel
import torch as th


def text_embedding(role_desc_set, model_path):
    model = BGEM3FlagModel(model_path, use_fp16 = True)
    role_embeddings = model.encode(role_desc_set)['dense_vecs']
    return th.tensor(role_embeddings)

if __name__ == '__main__':
    role_desc_set = ["Focus Fire", "Retreat", "Spread Out", "Advance", "Dead"]
    model_path = './bge-base-en-v1___5'
    role_embeddings = text_embedding(role_desc_set, model_path)
    print(len(role_embeddings), role_embeddings[0].shape, type(role_embeddings[0]))




