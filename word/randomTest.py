# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from tool.transformers.modeling_bert import BertForSequenceClassification, BertConfig
from tool.transformers import BertTokenizer
import torch, os
use_cuda = True if torch.cuda.is_available() else False

# ################################################ 模型加载 ############################################################
bert_token = BertTokenizer.from_pretrained('./chinese_roberta/vocab.txt', do_lower_case=False)  # 切词器
config = BertConfig.from_pretrained('./chinese_roberta/config.json', num_labels=21128)  # 0 1
model = BertForSequenceClassification.from_pretrained('./chinese_roberta/pytorch_model.bin', config=config)  # 模型加载
model = model.cuda() if use_cuda else model

def geneName():
    weightsName = []
    for file in os.listdir("weights"):
        weightsName.append(file)
    return weightsName

def loadModel(name):
    if use_cuda:
        pretrained_dict = torch.load('weights/' + name)  # load parameters
    else:
        pretrained_dict = torch.load('weights/' + name, map_location=torch.device('cpu'))
    # this_net_state_dict = model.state_dict()
    # backup_dict = {k: v for k, v in pretrained_dict.items() if k in this_net_state_dict}
    # this_net_state_dict.update(backup_dict)  # update吸收一个新的字典
    if list(model.state_dict().keys())[0][:5] != list(pretrained_dict.keys())[0][:5]:
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
    model.load_state_dict(pretrained_dict)
    return model

# ################################################ test #############################################################
def study_word():
    weightNames = geneName()[-1]
    # for name in weightNames:
    model = loadModel(weightNames)
    tokens = input("Enter: ")
    tokens = [t for t in tokens]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    input_mask = [1] * len(tokens)
    input_token_ids = bert_token.convert_tokens_to_ids(tokens)
    input_token_ids = torch.tensor([input_token_ids])
    segment_ids = torch.tensor([segment_ids])
    input_mask = torch.tensor([input_mask])
    if use_cuda:
        model = model.cuda()
        input_token_ids = input_token_ids.cuda()
        segment_ids = segment_ids.cuda()
        input_mask = input_mask.cuda()
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_token_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        _, predict = torch.max(logits, dim=1)
        index = int(predict.item())
        res = bert_token.ids_to_tokens.get(index, bert_token.unk_token)
    print("{}  ---->  {}".format(tokens[1: -1], res))
    print("\n")

if __name__ == '__main__':
    while True:
        study_word()

