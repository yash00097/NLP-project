from chatbot_engine import ChatbotEngine
import datetime # To timestamp our log

# --- CONFIGURATION ---
JSON_FILE_PATH = "Final_Dataset_Generated.json"
LOG_FILE_PATH = "chat_history.log"
# -------------------

def main():
    # 1. Initialize the chatbot engine
    bot = ChatbotEngine(JSON_FILE_PATH)
    
    print("\n--- తెలుగు కవిత్వ బాట్‌ కు స్వాగతం (v5 - New Rules) ---")
    print("నన్ను కవుల గురించి లేదా 'కవిత్రయం ఎవరు?' వంటి ప్రశ్నలు అడగండి.")
    print(f"(చాట్ నుండి నిష్క్రమించడానికి 'quit' అని టైప్ చేయండి | సంభాషణ '{LOG_FILE_PATH}' లో సేవ్ చేయబడుతుంది)\n")

    # 2. Open log file
    with open(LOG_FILE_PATH, "w", encoding="utf-8") as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"Chat session started at: {timestamp}\n\n")
        
        while True:
            # 3. Get user's question
            user_question = input("మీరు: ")
            
            if user_question.lower() == 'quit':
                print("బాట్: ధన్యవాదాలు! మళ్ళీ కలుద్దాం.")
                log_file.write(f"ME: {user_question}\nBOT: ధన్యవాదాలు! మళ్ళీ కలుద్దాం.\n")
                break
            
            # 4. Get response from the new hybrid engine
            response = bot.get_response(user_question)
            
            # 5. Print and log the response
            print(f"బాట్: {response}\n")
            log_file.write(f"ME: {user_question}\n")
            log_file.write(f"BOT: {response}\n\n")
            log_file.flush() # Ensure it writes immediately

if __name__ == "__main__":
    main()