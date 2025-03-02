from data_handler.data_handler import DataHandler



class pilotPipeline:
    def __init__(self):
        pass
    
    def get_data(self):
        handler = DataHandler()
        stock_data = handler.load_data()
        print('stock_data',stock_data)
        # Access Reliance's market data
        reliance_data = stock_data.get("RELIANCE")

        if reliance_data:
            print("\n📊 Reliance Market Data:")
            print(reliance_data["data"].head())  # Show first few rows of the DataFrame
            print("\n📋 Metadata:")
            print(reliance_data["metadata"])  # Show last updated date and source
        else:
            print("⚠️ Reliance data not available.")
        
        return stock_data