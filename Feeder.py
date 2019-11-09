import numpy as np
import json, os, time, pickle, librosa
from collections import deque
from threading import Thread
from random import shuffle

from Audio import melspectrogram

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

class Feeder:
    def __init__(self, is_Training= False):
        self.is_Training = is_Training

        self.Metadata_Load()

        if self.is_Training:            
            self.pattern_Queue = deque()
            pattern_Generate_Thread = Thread(target=self.Train_Pattern_Generate)
            pattern_Generate_Thread.daemon = True
            pattern_Generate_Thread.start()
        
    def Metadata_Load(self):
        with open(hp_Dict['Token_JSON_Path'], 'r') as f:
            self.token_Index_Dict = json.load(f)

        if self.is_Training:
            with open(os.path.join(hp_Dict['Train']['Pattern_Path'], hp_Dict['Train']['Metadata_File']).replace('\\', '/'), 'rb') as f:
                self.metadata_Dict = pickle.load(f)

            if not all([
                self.token_Index_Dict[key] == self.metadata_Dict['Token_Index_Dict'][key]
                for key in self.token_Index_Dict.keys()
                ]):
                raise ValueError('The token information of metadata information and hyper parameter is not consistent.')
            elif not all([
                self.metadata_Dict['Spectrogram_Dim'] == hp_Dict['Sound']['Spectrogram_Dim'],
                self.metadata_Dict['Mel_Dim'] == hp_Dict['Sound']['Mel_Dim'],
                self.metadata_Dict['Frame_Shift'] == hp_Dict['Sound']['Frame_Shift'],
                self.metadata_Dict['Frame_Length'] == hp_Dict['Sound']['Frame_Length'],
                self.metadata_Dict['Sample_Rate'] == hp_Dict['Sound']['Sample_Rate'],
                self.metadata_Dict['Max_Abs_Mel'] == hp_Dict['Sound']['Max_Abs_Mel'],
                ]):
                raise ValueError('The metadata information and hyper parameter setting are not consistent.')

    def Train_Pattern_Generate(self):
        min_Mel_Length = hp_Dict['Train']['Min_Wav_Length'] / hp_Dict['Sound']['Frame_Shift']
        max_Mel_Length = hp_Dict['Train']['Max_Wav_Length'] / hp_Dict['Sound']['Frame_Shift']

        path_List = [
            (path, self.metadata_Dict['Mel_Length_Dict'][path])
            for path in self.metadata_Dict['File_List']
            if self.metadata_Dict['Mel_Length_Dict'][path] >= min_Mel_Length and self.metadata_Dict['Mel_Length_Dict'][path] <= max_Mel_Length
            ]

        print(
            'Train pattern info', '\n',
            'Total pattern count: {}'.format(len(self.metadata_Dict['Mel_Length_Dict'])), '\n',
            'Use pattern count: {}'.format(len(path_List)), '\n',
            'Excluded pattern count: {}'.format(len(self.metadata_Dict['Mel_Length_Dict']) - len(path_List))
            )

        if hp_Dict['Train']['Pattern_Sorting']:
            path_List = [file_Name for file_Name, _ in sorted(path_List, key=lambda x: x[1])]
        else:
            path_List = [file_Name for file_Name, _ in path_List]

        while True:
            if not hp_Dict['Train']['Pattern_Sorting']:
                shuffle(path_List)

            path_Batch_List = [
                path_List[x:x + hp_Dict['Train']['Batch_Size']]
                for x in range(0, len(path_List), hp_Dict['Train']['Batch_Size'])
                ]
            shuffle(path_Batch_List)
            #path_Batch_List = path_Batch_List[0:2] + list(reversed(path_Batch_List))  #Batch size의 적절성을 위한 코드. 10회 이상 되면 문제 없음

            batch_Index = 0
            while batch_Index < len(path_Batch_List):
                if len(self.pattern_Queue) >= hp_Dict['Train']['Max_Pattern_Queue']:
                    time.sleep(0.1)
                    continue

                pattern_Count = len(path_Batch_List[batch_Index])

                mel_List = []
                token_List = []

                for file_Path in path_Batch_List[batch_Index]:
                    with open(os.path.join(hp_Dict['Train']['Pattern_Path'], file_Path).replace('\\', '/'), 'rb') as f:
                        pattern_Dict = pickle.load(f)

                    mel_List.append(pattern_Dict['Mel'])
                    token_List.append(pattern_Dict['Token'])

                max_Mel_Length = max([mel.shape[0] for mel in mel_List])
                max_Token_Length = max([token.shape[0] for token in token_List]) + 2

                new_Mel_Pattern = np.zeros(
                    shape=(pattern_Count, max_Mel_Length, hp_Dict['Sound']['Mel_Dim']),
                    dtype= np.float32
                    )
                new_Token_Pattern = np.zeros(
                    shape=(pattern_Count, max_Token_Length),
                    dtype= np.int32
                    ) + self.token_Index_Dict['<E>']
                new_Token_Pattern[:, 0] = self.token_Index_Dict['<S>']

                for pattern_Index, (mel, token) in enumerate(zip(mel_List, token_List)):
                    new_Mel_Pattern[pattern_Index, :mel.shape[0]] = mel
                    new_Token_Pattern[pattern_Index, 1:token.shape[0] + 1] = token

                self.pattern_Queue.append({
                    'mels': new_Mel_Pattern,
                    'mel_lengths': np.array([mel.shape[0] for mel in mel_List], dtype=np.int32),
                    'tokens': new_Token_Pattern,
                    'token_lengths': np.array([token.shape[0] + 1 for token in token_List], dtype=np.int32) #Only one of <S> or <E> is used.
                    })

                batch_Index += 1

    def Get_Train_Pattern(self):
        while len(self.pattern_Queue) == 0: #When training speed is faster than making pattern, model should be wait.
            time.sleep(0.01)
        return self.pattern_Queue.popleft()

    def Get_Inference_Pattern(self, wav_Path_List):
        pattern_Count = len(wav_Path_List)

        mel_List = [
            np.transpose(melspectrogram(
                y= librosa.effects.trim(librosa.core.load(path, sr= hp_Dict['Sound']['Sample_Rate'])[0], top_db=15)[0] * 0.99,
                num_freq= hp_Dict['Sound']['Spectrogram_Dim'],                
                frame_shift_ms= hp_Dict['Sound']['Frame_Shift'],                
                frame_length_ms= hp_Dict['Sound']['Frame_Length'],
                num_mels= hp_Dict['Sound']['Mel_Dim'],
                sample_rate= hp_Dict['Sound']['Sample_Rate'],
                max_abs_value= hp_Dict['Sound']['Max_Abs_Mel']
                )).astype(np.float32)
            for path in wav_Path_List
            ]
        max_Mel_Length = max([mel.shape[0] for mel in mel_List])

        new_Mel_Pattern = np.zeros(
            shape=(pattern_Count, max_Mel_Length, hp_Dict['Sound']['Mel_Dim']),
            dtype= np.float32
            )

        for pattern_Index, mel in enumerate(mel_List):
            new_Mel_Pattern[pattern_Index, :mel.shape[0]] = mel
    
        new_Token_Pattern = np.zeros(
                shape=(pattern_Count, 1),
                dtype= np.int32
                ) + self.token_Index_Dict['<S>']

        return {
            'mels': new_Mel_Pattern,
            'mel_lengths': np.array([mel.shape[0] for mel in mel_List], dtype=np.int32),
            'initial_tokens': new_Token_Pattern
            }

if __name__ == '__main__':
    new_Feeder = Feeder(is_Training= True)
    while True:
        print(len(new_Feeder.pattern_Queue))
        time.sleep(1.0)