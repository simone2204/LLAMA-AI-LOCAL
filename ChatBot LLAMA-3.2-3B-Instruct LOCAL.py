import sys
import fitz
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QWidget, QVBoxLayout, QLineEdit, QTextEdit, QFileDialog, QProgressBar
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Imposta il percorso corretto del modello Llama scaricato sul proprio desktop
MODEL_PATH = "C:/your/personal/route/LLama-3.2-3B"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
    ).to(device)

def ask_question(context: str, question: str):
    """Genera una risposta basata sul documento caricato e sulla domanda dell'utente."""
    prompt = f"""
Sei un assistente AI esperto nella comprensione dei documenti.
Leggi il testo del documento e rispondi alla domanda dell'utente.
Se il documento non contiene informazioni pertinenti, rispondi 'Non sono sicuro'.

### Documento:
{context}

### Domanda:
{question}

### Risposta:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer


class PdfLoaderThread(QThread):
    progress = pyqtSignal(int)  # Segnale per aggiornare la progress bar
    finished = pyqtSignal(str)  # Segnale che trasmette il testo estratto

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        """Estrae il testo dal PDF ed emette i segnali."""
        try:
            text = ""
            doc = fitz.open(self.file_path)
            num_pages = len(doc)

            if num_pages == 0:
                self.finished.emit("Errore: Il PDF è vuoto o non può essere letto.")
                return

            for i, page in enumerate(doc):
                text += page.get_text()
                progress_value = int(((i + 1) / num_pages) * 100)
                self.progress.emit(progress_value)
                QApplication.processEvents()
                self.msleep(50)

            if not text.strip():
                self.finished.emit("Errore: Il PDF non contiene testo leggibile.")
            else:
                self.finished.emit(text.strip())

        except Exception as e:
            self.finished.emit(f"Errore nel caricamento del PDF: {e}")

class ChatbotThread(QThread):
    finished = pyqtSignal(str)

    def __init__(self, context, question):
        super().__init__()
        self.context = context
        self.question = question

    def run(self):
        """Genera la risposta senza bloccare la GUI."""
        try:
            answer = ask_question(self.context, self.question)
            self.finished.emit(answer)
        except Exception as e:
            self.finished.emit(f"Errore nel chatbot: {e}")


class MainWindow(QMainWindow):
    """Interfaccia grafica del chatbot con caricamento PDF e progress bar."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChatBot Llama 3.2")
        self.setGeometry(250, 250, 1200, 800)
        self.setWindowIcon(QIcon("ChatBotIcon.PNG"))
        self.setStyleSheet("background-color: #aed8f5")
        self.pdf_text = ""  # Variabile per il contenuto del PDF
        self.initUI()

    def initUI(self):
        """Inizializza gli elementi grafici della finestra principale."""
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setAlignment(Qt.AlignTop)

        self.label = QLabel("Ciao! Sono il tuo ChatBot e sono qui per aiutarti", self)
        self.label.setFont(QFont("Arial", 20))
        self.label.setStyleSheet("background-color: #d9f2d3; border-radius: 35px;")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(880, 80)
        self.main_layout.addWidget(self.label, alignment=Qt.AlignCenter)

        self.load_pdf_button = QPushButton("Carica PDF", self)
        self.load_pdf_button.setFont(QFont("Arial", 12))
        self.load_pdf_button.setStyleSheet("background-color: #e3f3fd; border-radius: 15px; padding: 10px; width: 30%;")
        self.load_pdf_button.clicked.connect(self.load_pdf)
        self.main_layout.addWidget(self.load_pdf_button)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("Scrivi qui la tua domanda...")
        self.input_box.setFont(QFont("Arial", 14))
        self.input_box.setStyleSheet("padding: 10px; border-radius: 10px;")
        self.main_layout.addWidget(self.input_box)

        self.output_box = QTextEdit(self)
        self.output_box.setFont(QFont("Arial", 12))
        self.output_box.setStyleSheet("padding: 10px; background-color: #f0f8ff; border-radius: 10px;")
        self.output_box.setReadOnly(True)
        self.main_layout.addWidget(self.output_box)

        self.query_button = QPushButton("Invia", self)
        self.query_button.setFont(QFont("Arial", 12))
        self.query_button.setStyleSheet("background-color: #e3f3fd; border-radius: 15px; padding: 10px;")
        self.query_button.clicked.connect(self.ask_chatbot)
        self.main_layout.addWidget(self.query_button)

    def load_pdf(self):
        """Apre un file PDF e avvia il caricamento nel thread separato."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Carica PDF", "", "PDF Files (*.pdf)")
        if file_path:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.label.setText("Caricamento del PDF in corso...")

            self.pdf_loader = PdfLoaderThread(file_path)
            self.pdf_loader.progress.connect(self.progress_bar.setValue)
            self.pdf_loader.finished.connect(self.on_pdf_loaded)
            self.pdf_loader.start()

    def ask_chatbot(self):
        """Invia una domanda al chatbot e attende la risposta in modo asincrono."""
        question = self.input_box.text().strip()
        if not question or not self.pdf_text:
            self.output_box.setText("Carica prima un PDF e inserisci una domanda.")
            return

        self.output_box.setText("Sto elaborando la risposta...")

        # Creiamo un thread separato per evitare di bloccare la GUI
        self.chatbot_thread = ChatbotThread(self.pdf_text, question)
        self.chatbot_thread.finished.connect(self.on_chatbot_response)
        self.chatbot_thread.start()

    def on_chatbot_response(self, answer):
        """Aggiorna la GUI con la risposta del chatbot."""
        self.output_box.setText(answer)

    def on_pdf_loaded(self, text):
        """Gestisce la fine del caricamento del PDF."""
        if text.startswith("Errore:"):
            self.label.setText(text)
        else:
            self.pdf_text = text
            self.label.setText("PDF caricato con successo! Fai una domanda.")
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
