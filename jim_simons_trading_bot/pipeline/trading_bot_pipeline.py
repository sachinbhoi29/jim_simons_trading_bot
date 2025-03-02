from data_handler.data_handler import DataHandler



class pilotPipeline:
    def __init__(self):
        pass
    
    def get_data(self):
        handler = DataHandler()
        stock_data = handler.load_data()        
        return stock_data