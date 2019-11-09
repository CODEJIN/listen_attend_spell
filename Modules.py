import tensorflow as tf
import numpy as numpy
import json

from Attention_Modules import Transformer, BahdanauAttention, BahdanauAttention2

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)
with open(hp_Dict['Token_JSON_Path'], 'r') as f:
    token_Index_Dict = json.load(f)

class Listner(tf.keras.Model):
    def __init__(self):
        super(Listner, self).__init__(name= '')

        self.layer_Dict = {}
        for rnn_Index, cell_Size in enumerate(hp_Dict['Listener']['Uni_Direction_Cell_Size']):        
            self.layer_Dict['Layer{}'.format(rnn_Index)] = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units= cell_Size,
                    return_sequences= True
                    )
                )

    def call(self, inputs):
        new_Tensor = inputs
        for rnn_Index, _ in enumerate(hp_Dict['Listener']['Uni_Direction_Cell_Size']):            
            new_Tensor = self.layer_Dict['Layer{}'.format(rnn_Index)](new_Tensor)
            if rnn_Index < len(hp_Dict['Listener']['Uni_Direction_Cell_Size']) - 1:
                new_Tensor = self.reshape(new_Tensor)

        return new_Tensor

    @tf.function
    def reshape(self, inputs):
        # batch_Size, time_Step = tf.shape(inputs)[:2]  #Currently, this code is not supported at tf 2.0.0 version.
        batch_Size, time_Step = tf.shape(inputs)[0], tf.shape(inputs)[1]
        dimension = inputs.get_shape()[-1]

        inputs = tf.concat(
            [inputs, tf.zeros([batch_Size, time_Step % 2, dimension], dtype= tf.float32)],
            axis= 1
            )
        time_Step += time_Step % 2

        return tf.reshape(inputs, [batch_Size, time_Step // 2, dimension * 2])

class Embedding(tf.keras.layers.Layer):
    def __init__(self):
        super(Embedding, self).__init__(name= '')

        self.embedding_Variable = tf.Variable(
            tf.random.uniform(shape= (
                len(token_Index_Dict),
                hp_Dict['Speller']['Embedding_Size']
                )),
            name='embedding_v',
            dtype= tf.float32
            )

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embedding_Variable, inputs)

class Speller(tf.keras.Model):
    def __init__(self):
        super(Speller, self).__init__(name= '')

        self.layer_Dict = {}
        self.layer_Dict['Embedding'] = Embedding()
        self.layer_Dict['Attention'] = Transformer(size= hp_Dict['Speller']['Attention_Size']) if hp_Dict['Speller']['Use_Transformer'] else BahdanauAttention(size= hp_Dict['Speller']['Attention_Size'])        
        for rnn_Index, cell_Size in enumerate(hp_Dict['Speller']['Cell_Size']):
            self.layer_Dict['RNN_{}'.format(rnn_Index)] = tf.keras.layers.LSTM(
                units= cell_Size,
                return_sequences= True
                )
        self.layer_Dict['Projection'] = tf.keras.layers.Dense(
            units= len(token_Index_Dict),
            use_bias= True,
            )


    def call(self, inputs):
        '''
        inputs[0]: query, embedding tensor
        inputs[1]: value, encoder tensor
        inputs[2]: value_length, encoder's input length tensor

        Q = Query : t-1 시점의 디코더 셀에서의 은닉 상태
        K = Keys : 모든 시점의 인코더 셀의 은닉 상태들
        V = Values : 모든 시점의 인코더 셀의 은닉 상태들
        https://wikidocs.net/22893
        '''
        token_Tensor, value_Tensor, _ = inputs        
        query_Tensor = self.layer_Dict['Embedding'](token_Tensor)

        # mel_length_Tensor += mel_length_Tensor % 2  #Making even value
        # value_Length_Tensor = mel_length_Tensor // tf.pow(2, len(hp_Dict['Listener']['Uni_Direction_Cell_Size']) - 1)   #encoder tensor length
        # value_Mask_Tensor = tf.sequence_mask(value_Length_Tensor, tf.shape(value_Tensor)[1])        
        #attention_Tensor, history_Tensor = self.layer_Dict['Attention'](inputs= [query_Tensor, value_Tensor], mask=[False, value_Mask_Tensor])
        
        attention_Tensor, history_Tensor = self.layer_Dict['Attention'](inputs= [query_Tensor, value_Tensor])
        new_Tensor = tf.concat([query_Tensor, attention_Tensor], axis= -1)
        for rnn_Index, _ in enumerate(hp_Dict['Speller']['Cell_Size']):
            new_Tensor = self.layer_Dict['RNN_{}'.format(rnn_Index)](new_Tensor)        
        new_Tensor = self.layer_Dict['Projection'](new_Tensor)

        return new_Tensor, history_Tensor

if __name__ == '__main__':
    mel = tf.keras.layers.Input(shape=[743, hp_Dict['Sound']['Mel_Dim']], dtype= tf.float32)
    mel_length = tf.keras.layers.Input(shape=[], dtype= tf.int32)
    token = tf.keras.layers.Input(shape=[97,], dtype= tf.int32)
    listener = Listner()(mel)
    speller, attention = Speller()([token, listener, mel_length])
    model = tf.keras.Model(inputs=[mel, mel_length, token], outputs= [speller, attention])
    
    model.summary()

    print(attention)