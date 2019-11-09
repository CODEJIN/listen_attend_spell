import tensorflow as tf
import numpy as np
import json, os, time
from threading import Thread
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from datetime import datetime

from Feeder import Feeder
import Modules

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

class LAS:
    def __init__(self, is_Training= False):
        self.feeder = Feeder(is_Training= is_Training)
        self.Model_Generate()

    def Model_Generate(self):
        self.layer_Dict = {}
        self.layer_Dict['Mel'] = tf.keras.layers.Input(shape=[None, hp_Dict['Sound']['Mel_Dim']], dtype= tf.float32)
        self.layer_Dict['Mel_Length'] = tf.keras.layers.Input(shape=[], dtype= tf.int32)
        self.layer_Dict['Token'] = tf.keras.layers.Input(shape=[None,], dtype= tf.int32)        
        self.layer_Dict['Listener'] = Modules.Listner()(self.layer_Dict['Mel'])
        self.layer_Dict['Speller'], self.layer_Dict['Attention'] = Modules.Speller()([
            self.layer_Dict['Token'],
            self.layer_Dict['Listener'],
            self.layer_Dict['Mel_Length']
            ])
        self.model = tf.keras.Model(
            inputs=[self.layer_Dict['Mel'], self.layer_Dict['Mel_Length'], self.layer_Dict['Token']],
            outputs= [self.layer_Dict['Speller'], self.layer_Dict['Attention']]
            )
        self.model.summary()

        #optimizer는 @tf.function의 밖에 있어야 함
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate= hp_Dict['Train']['Learning_Rate'],
            beta_1= hp_Dict['Train']['ADAM']['Beta1'],
            beta_2= hp_Dict['Train']['ADAM']['Beta2'],
            epsilon= hp_Dict['Train']['ADAM']['Epsilon'],
            )

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, hp_Dict['Sound']['Mel_Dim']], dtype=tf.float32),
            tf.TensorSpec(shape=[None,], dtype=tf.int32),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None,], dtype=tf.int32)
            ],
        autograph= False,
        experimental_relax_shapes= True
        )
    def Train_Step(self, mels, mel_lengths, tokens, token_lengths):
        with tf.GradientTape() as tape:
            logits, _ = self.model(inputs= [mels, mel_lengths, tokens[:, :-1]], training= True)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels= tokens[:, 1:],
                logits= logits
                )
            loss *= tf.sequence_mask(
                lengths= token_lengths,
                maxlen= tf.shape(loss)[-1],
                dtype= tf.float32
                )
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss


    # Don't use @tf.function here. it makes slower.
    @tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, hp_Dict['Sound']['Mel_Dim']], dtype=tf.float32),
        tf.TensorSpec(shape=[None,], dtype=tf.int32),
        tf.TensorSpec(shape=[None, 1], dtype=tf.int32)
        ],
    autograph= True,
    experimental_relax_shapes= True
    )
    def Inference_Step(self, mels, mel_lengths, initial_tokens):
        tokens = tf.zeros(shape=[tf.shape(mels)[0], 0], dtype= tf.int32)
        for _ in range(hp_Dict['Speller']['Max_Length']):
            tokens = tf.concat([initial_tokens, tokens], axis=-1)
            logits, attention_History = self.model(inputs= [mels, mel_lengths, tokens], training= False)
            tokens = tf.argmax(logits, axis=-1, output_type= tf.int32)

        return tokens, attention_History

    def Restore(self):
        checkpoint_File_Path = os.path.join(hp_Dict['Checkpoint_Path'], 'CHECKPOINT.H5').replace('\\', '/')
        
        if not os.path.exists('{}.index'.format(checkpoint_File_Path)):
            print('There is no checkpoint.')
            return

        self.model.load_weights(checkpoint_File_Path)
        print('Checkpoint \'{}\' is loaded.'.format(checkpoint_File_Path))

    def Train(self):
        def Run_Inference():
            wav_Path_List = []
            with open('Inference_Wav_Path_in_Train.txt', 'r') as f:
                for line in f.readlines():
                    wav_Path_List.append(line.strip())

            self.Inference(wav_Path_List)

        step = 0
        Run_Inference()
        while True:
            start_Time = time.time()

            loss = self.Train_Step(**self.feeder.Get_Train_Pattern())
            step += 1
            display_List = [
                'Time: {:0.3f}'.format(time.time() - start_Time),
                'Step: {}'.format(step),
                'Loss: {:0.5f}'.format(loss)
                ]
            print('\t\t'.join(display_List))

            if step % hp_Dict['Train']['Checkpoint_Save_Timing'] == 0:
                os.makedirs(os.path.join(hp_Dict['Checkpoint_Path']).replace("\\", "/"), exist_ok= True)
                self.model.save_weights(os.path.join(hp_Dict['Checkpoint_Path'], 'CHECKPOINT.H5').replace('\\', '/'))
            
            if step % hp_Dict['Train']['Inference_Timing'] == 0:
                Run_Inference()

    def Inference(self, wav_Path_List, label= None):
        print('Inference running...')
        inference_Pattern = self.feeder.Get_Inference_Pattern(wav_Path_List)

        tokens, attention_History = self.Inference_Step(**inference_Pattern)

        export_Inference_Thread = Thread(
            target= self.Export_Inference,
            args= [
                wav_Path_List,
                inference_Pattern['mels'],
                inference_Pattern['mel_lengths'],
                tokens.numpy(),
                attention_History.numpy(),
                label or datetime.now().strftime("%Y%m%d.%H%M%S")
                ]
            )
        export_Inference_Thread.daemon = True
        export_Inference_Thread.start()

    def Export_Inference(self, wav_Path_List, mel_List, mel_Length_List, token_List, attention_History_List, label):
        os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Plot').replace("\\", "/"), exist_ok= True)
        
        index_Token_Dict = {index: token for token, index in self.feeder.token_Index_Dict.items()}
        
        for index, (wav_Path, mel, mel_Length, token, attention_History) in enumerate(zip(wav_Path_List, mel_List, mel_Length_List, token_List, attention_History_List)):
            mel = mel[:mel_Length]

            attention_History = attention_History[:, :(mel_Length + (mel_Length % 2)) // (2 ** (len(hp_Dict['Listener']['Uni_Direction_Cell_Size']) - 1))]
            
            if len(np.where(token == self.feeder.token_Index_Dict['<E>'])[0]) > 0:
                stop_Index = np.where(token == self.feeder.token_Index_Dict['<E>'])[0][0]
                token = token[:stop_Index]
                attention_History = attention_History[:stop_Index]
            token = [index_Token_Dict[x] for x in token]

            if len(token) == 0:
                print('The exported token length of \'{}\' is zero. It is skipped.'.format(wav_Path))
                continue

            new_Figure = plt.figure(figsize=(24, 24), dpi=100)
            plt.subplot2grid((3, 1), (0, 0))
            plt.imshow(np.transpose(mel), aspect='auto', origin='lower')
            plt.title('Mel    Path: {}'.format(wav_Path))
            plt.colorbar()
            plt.subplot2grid((3, 1), (1, 0), rowspan=2)
            plt.imshow(attention_History, aspect='auto', origin='lower')
            
            plt.title('Attention history    Inference: {}'.format(''.join(token if token[-1] != self.feeder.token_Index_Dict['<E>'] else token[:-1])))
            plt.yticks(
                range(attention_History.shape[0]),
                token,
                fontsize = 10
                )
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(hp_Dict['Inference_Path'], 'Plot', '{}.IDX_{}.PNG'.format(label, index)).replace("\\", "/")
                )
            plt.close(new_Figure)

if __name__ == '__main__':
    new_Model = LAS(is_Training= True)
    new_Model.Restore()
    new_Model.Train()