### Medical Chat Bot Using Gemini Model

Folder Structure:
```plaintext
medical_chat_bot/
|---- Dataset/
|       |---- All-2479-Answers-retrieved-from-MedQuAD.csv
|---- Notebook/
|       |---- testing.ipynb
|---- Report/
|       |---- Report_Medical_Chatbot_Using_Gemini_Model.pdf
|---- src/
|       |---- faiss_index
|       |---- All-2479-Answers-retrieved-from-MedQuAD.csv
|       |---- helper.py
|       |---- main.py
|---- .env
|---- README.md
|---- requirements.txt
```

- How to Run:
- Step 1:

```bash
conda create -n chatbot python=3.12 -y
```

- Step 2:

```bash
conda activate chatbot
```

- Step 3:

```bash
pip install -r requirements.txt
```

- Step 4: `.env` file:

```plaintext
GOOGLE_API_KEY="API_KEY"
```
- Step 5:

```bash
streamlit run main.py
```

dataset link: [https://github.com/abachaa/MedQuAD]