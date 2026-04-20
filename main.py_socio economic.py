from predict import predict_crisis

def run_system():
    # Example input: [GDP growth, Unemployment rate, Inflation, Poverty rate]
    indicators = [2.5, 8.1, 6.3, 25.0]
    result = predict_crisis(indicators)
    print("Early Warning:", result)

if __name__ == "__main__":
    run_system()
