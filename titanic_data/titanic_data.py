import pickle


class TitanicData(object):
    def __init__(self):
        self.sexo = pickle.load(open('parametros/sexo.pkl', 'rb'))

    def data_preparation(self, df):
        print(self.sexo.head())
        df['sexo'] = self.sexo

        return df
