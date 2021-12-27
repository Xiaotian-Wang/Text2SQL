from torch import nn
import torch
from transformers import BertTokenizer, BertModel

class Tokenizer(object):
    def __init__(self, config):
        super().__init__()

        self.pretrained_model_type = 'bert-base-multilingual-uncased'

        if config.get('pretrained_model_type') is not None:
            self.pretrained_model_type = config.get('pretrained_model_type')

        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_type)

    def tokenize(self, input_string: str):
        tokenized = self.tokenizer(input_string, return_tensors='pt')
        return tokenized


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.para_adjustable = True
        self.pretrained_model_type = 'bert-base-multilingual-uncased'
        if config.get('pretrained_model_type') is not None:
            self.pretrained_model_type = config.get('pretrained_model_type')
        if config.get('para_adjustable') is not None:
            self.para_adjustable = config.get('para_adjustable')

        self.tokenizer = Tokenizer(config).tokenize
        self.basic_model = BertModel.from_pretrained(self.pretrained_model_type)

    def _encode_name(self, name_tokenized, pooling_method='avg'):
        lhs = self.basic_model(**name_tokenized).get('last_hidden_state')

        if pooling_method == 'max':
            encoded = torch.max_pool1d(lhs.squeeze(0).transpose(0, 1).unsqueeze(0), lhs.size()[1]).squeeze(-1)
        else:
            encoded = torch.avg_pool1d(lhs.squeeze(0).transpose(0, 1).unsqueeze(0), lhs.size()[1]).squeeze(-1)

        return encoded

    def _encode_names(self, names):

        list_to_return = torch.tensor([])
        for name in names:
            tokenized = self.tokenizer(name)
            encoded = self._encode_name(tokenized)
            list_to_return = torch.cat((list_to_return, encoded))

        list_to_return = list_to_return.unsqueeze(0)
        return list_to_return

    def forward(self, inputs: dict):

        query = inputs.get('query')
        columns = inputs.get('columns')
        tables = inputs.get('tables')
        query_tokenized = self.tokenizer(query)
        query_last_hidden_state = self.basic_model(**query_tokenized).get('last_hidden_state').squeeze(0)
        columns_last_hidden_state = self._encode_names(columns).squeeze(0)
        tables_last_hidden_state = self._encode_names(tables).squeeze(0)
        result_to_return = torch.cat((columns_last_hidden_state, tables_last_hidden_state, query_last_hidden_state))
        self.columns_index = (0, columns_last_hidden_state.size()[0])
        self.tables_index = (columns_last_hidden_state.size()[0], columns_last_hidden_state.size()[0]+tables_last_hidden_state.size()[0])
        self.query_index = (columns_last_hidden_state.size()[0]+tables_last_hidden_state.size()[0], columns_last_hidden_state.size()[0]+tables_last_hidden_state.size()[0]+query_last_hidden_state.size()[0])
        return result_to_return


if __name__ == "__main__":

    # Do some simple tests here

    encoder = BertEncoder(config={})
    query = 'this is the query'
    columns = ['col1', 'col2', 'this']
    tables = ['table1', 'table2', 'that']
    inputs = {
        'query': query,
        'columns': columns,
        'tables': tables
    }
    a = encoder.forward(inputs=inputs)

