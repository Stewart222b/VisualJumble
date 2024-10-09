from models.train_engine import TrainEngine

if __name__ == '__main__':
    engine = TrainEngine()

    model = engine.model

    #engine.get_output_shape()
    engine.start()