import os
import sys
import fitz  # Per leggere i PDF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QWidget, QVBoxLayout, QLineEdit, QTextEdit, QFileDialog
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt

# Imposta il percorso corretto del modello Llama scaricato
MODEL_PATH = "C:/Users/simo-/OneDrive/Desktop/LLama-3.2-3B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

def ask_question(context: str, question: str):
    prompt = f"Documento: {context}\n\nDomanda: {question}\nRisposta:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output = model.generate(**inputs,
                            max_new_tokens=300,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChatBot Llama 3.2")
        self.setGeometry(250, 250, 1200, 800)
        self.setWindowIcon(QIcon("ChatBotIcon.PNG"))
        self.setStyleSheet("background-color: #aed8f5")
        self.pdf_text = ""  # Variabile per il contenuto del PDF
        self.initUI()

    def initUI(self):
        # Creazione del contenitore e layout principale
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setAlignment(Qt.AlignTop)

        # Creazione dell'etichetta con larghezza ridotta
        self.label = QLabel("Ciao! Sono il tuo ChatBot e sono qui per aiutarti", self)
        self.label.setFont(QFont("Arial", 20))
        self.label.setStyleSheet("background-color: #d9f2d3; border-radius: 35px;")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(880, 80)  # Forza una larghezza e altezza specifica
        self.main_layout.addWidget(self.label, alignment=Qt.AlignCenter)

        # Pulsante per caricare il PDF
        self.load_pdf_button = QPushButton("Carica PDF", self)
        self.load_pdf_button.setFont(QFont("Arial", 12))
        self.load_pdf_button.setStyleSheet("background-color: #e3f3fd; border-radius: 15px; padding: 10px; width: 30%;")
        self.load_pdf_button.clicked.connect(self.load_pdf)
        self.main_layout.addWidget(self.load_pdf_button)

        # Input utente
        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("Scrivi qui la tua domanda...")
        self.input_box.setFont(QFont("Arial", 14))
        self.input_box.setStyleSheet("padding: 10px; border-radius: 10px;")
        self.main_layout.addWidget(self.input_box)

        # Sezione risposte AI
        self.output_box = QTextEdit(self)
        self.output_box.setFont(QFont("Arial", 12))
        self.output_box.setStyleSheet("padding: 10px; background-color: #f0f8ff; border-radius: 10px;")
        self.output_box.setReadOnly(True)
        self.main_layout.addWidget(self.output_box)

        # Pulsante per inviare la domanda al chatbot
        self.query_button = QPushButton("Invia", self)
        self.query_button.setFont(QFont("Arial", 12))
        self.query_button.setStyleSheet("background-color: #e3f3fd; border-radius: 15px; padding: 10px;")
        self.query_button.clicked.connect(self.ask_chatbot)
        self.main_layout.addWidget(self.query_button)

    def load_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Carica PDF", "", "PDF Files (*.pdf)")
        if file_path:
            text = ""
            try:
                doc = fitz.open(file_path)
                for page in doc:
                    text += page.get_text()
                self.pdf_text = text.strip()
                self.label.setText("PDF caricato con successo! Fai una domanda.")
            except Exception as e:
                self.label.setText("Errore nel caricamento del PDF.")
                print(f"Errore: {e}")

    def ask_chatbot(self):
        question = self.input_box.text().strip()
        if not question or not self.pdf_text:
            self.output_box.setText("Carica prima un PDF e inserisci una domanda.")
            return

        answer = ask_question(self.pdf_text, question)
        self.output_box.setText(answer)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
