from FlagEmbedding import BGEM3FlagModel
import torch as th


class TextEmbedding(object):
    """Using BGE-m3 model to get text embedding
        Attributes:
            model_path: the path of the model
        Func:
            embedding_text:
                input: text list
                return: text embedding tensors
    """
    def __init__(self, model_path):
        self.model = BGEM3FlagModel(model_path, use_fp16 = True)

    def embedding_text(self, text: list):
        text_embeddings = self.model.encode(text)["dense_vecs"]
        return th.tensor(text_embeddings)



# def text_embedding(role_desc_set, model_path):
#     model = BGEM3FlagModel(model_path, use_fp16 = True)
#     role_embeddings = model.encode(role_desc_set)['dense_vecs']
#     return th.tensor(role_embeddings)

if __name__ == '__main__':
    role_desc_set = ["Focus Fire", "Retreat", "Spread Out", "Advance", "Dead"]
    model_path = 'C:/Users/xw/Desktop/workspace/pymarl2/model/BAAI/bge-base-en-v1___5'
    role_embeddings = text_embedding(role_desc_set, model_path)
    print(len(role_embeddings), role_embeddings[0], type(role_embeddings[0]))




