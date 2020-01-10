from tkinter import messagebox as mb
from tkinter import *

class DemoDialog:
    def __init__(self, induk):
        self.induk = induk
        self.induk.protocol("WM_DELETE_WINDOW", self.tutup)
        self.induk.resizable(False, False)
        self.aturKomponen()

    def aturKomponen(self):
        # atur frame utama
        mainFrame = Frame(self.induk)
        mainFrame.pack(fill=BOTH, expand=YES)

        ## box tombol dialog
        box = Frame(mainFrame, bd=20)
        box.pack(fill=BOTH, expand=YES)


    def onGambarOk(self, event=None):
        mb.showinfo("Info Penting!", "Pengambilan 50 gambar telah selesai \n Tekan tombol 'Open file' untuk melihat hasilnya !")
    def onTrainingOk(self, event=None):
        mb.showinfo("Info Penting!", "Proses Training data baru telah selesai \n Tombol 'Detect' siap di gunakan !")
    def onDetectOk(self, event=None):
        mb.showinfo("Info Penting!", "Kemera sedang di siapkan \nTekan tombol 'q' untuk keluar dari kamera !")
    def onFiletOk(self, event=None):
        mb.showinfo("Info Penting!", "Cari dan hapus gambar yang bukan wajah !")

    def onKlikError(self, event=None):
        mb.showerror("Kesalahan Fatal!", "Ini adalah DIALOG SHOW ERROR")

    def onKlikWarning(self, event=None):
        mb.showwarning("Peringatan!", "Ini adalah DIALOG SHOW WARNING")

    def tutup(self, event=None):
        self.induk.destroy()
