import os
import numpy as np
from dialog import *
import cv2
from tkinter import filedialog,simpledialog
from tkSimpleStatusbar import *
import time

master = Tk() # membuat window
master.title("Face Recognition 1.0")
master.resizable(0,0) # me-non aktifkan maximize

#------------------ MEMBUAT STATUS BAR DI MULAI PROGRAM ------------------#
status = StatusBar(master)
status.pack(side=BOTTOM, fill=X)
status.set("selamat datang...")

def buatFolder(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

#---------------------- MEMBUAT FUNGSI DARI TOMBOL ----------------------#

def detect():
    status.set("Identifikasi wajah, tekan tombol 'q' untuk keluar dari kamera")
    # Membuat Local Binary Patterns Histograms untuk face recognization
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    buatFolder("trainer/")

    # muat mode yang data yang sudah di simpan pada recognizer
    recognizer.read('trainer/trainer.yml')

    cascadePath = "face-detect.xml"

    # Create classifier from prebuilt model
    faceCascade = cv2.CascadeClassifier(cascadePath);

    # mengatur font style text
    font = cv2.FONT_HERSHEY_SIMPLEX
    # TODO 1 : Pilih Kamera (0 untuk kamera laptop & 1 untuk kamera USB) --> perhatikan variable cam
    # inisiasi camera
    cam = cv2.VideoCapture(0)
    # cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    q = DemoDialog(master)
    q.onDetectOk()
    # Loop
    while True:
        # membaca camera
        ret, im = cam.read()

        # merubah gambar menjadi grayscale
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # mendeteksi wajah pada gambar
        wajah = faceCascade.detectMultiScale(gray, 1.2, 5)

        # perulangan untun mencocokan wajah dengan data training
        for (x, y, w, h) in wajah:

            # menandai pada bagian wajah dengan kotak / persegi panjang
            cv2.rectangle(im, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)

            # mengenali id wajah
            Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # TODO 3 : Pelabelan Manual berdasarkan Id gambar
            # mengecek apakan id wajah sudah ada
            if Id is 1:
                Id = "Ni'am "
                kemiripan = format(round(100 - confidence, 2))
            if Id is 2:
                Id = "Roihan"
                kemiripan = format(round(100 - confidence, 2))
            if Id is 3:
                Id = "Lali "
                kemiripan = format(round(100 - confidence, 2))
            # if Id is 4:
            #     Id = "Mawa {0:.2f}%".format(round(100 - confidence, 2))
            # if Id is 5:
            #     Id = "Wiranto {0:.2f}%".format(round(100 - confidence, 2))
            # if Id is 6:
            #     Id = "Farid {0:.2f}%".format(round(100 - confidence, 2))

            if float(kemiripan)<=10:
                tampil = "wajah tidak terindentifikasi"
                print(tampil)
            else:
                tampil=Id+kemiripan
                print(tampil)
            # menampilkan teks pada wajah yang terdeteksi
            cv2.rectangle(im, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
            cv2.putText(im, str(tampil), (x, y - 40), font, 1, (255, 255, 255), 3)

        # menampilkan video frame untuk user
        cv2.imshow('Pengujian Pengenalan Wajah', im)


        # tekan 'q' untuk menghentikan program
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # keluar dari camera
    cam.release()

    # keluar dari semua jendela
    cv2.destroyAllWindows()
    status.clear()

def new():
    status.set("menyiapkan kamera, silahkan tunggu beberapa saat...")
    # vid_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # TODO 1 : Pilih Kamera (0 untuk kamera laptop & 1 untuk kamera USB) --> perhatikan variable vid-cam
    vid_cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('face-detect.xml')

    # memberikan id untuk masing2 wajah
    # face_id = input("masukkan Id baru : ")
    mm = Tk().withdraw()
    face_id = simpledialog.askstring(title="Pelabelan Wajah", prompt="Masukkan Id :")

    # TODO 2 : Rubah variable "jumlah_pengambilan" untuk menentukan jumlah gambar training per wajah
    jumlah_pengambilan = 50
    # inisialisasi jumlah gambar
    jumlah = 0

    buatFolder("dataset/")

    # Membuat Perulangan untuk mengambil 100 gambar wajah
    while (True):

        # Mengambil video frame
        _, image_frame = vid_cam.read()

        # Merubah video frame menjadi gambar grayscale
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

        # Mendeteksi wajah pada videocam
        wajah = face_detector.detectMultiScale(gray, 1.3, 5)

        # mengambil gambar yang sudah di crop dan menyimpannya pada forder dataset
        for (x, y, w, h) in wajah:
            # Meng-crop hanya pada bagian wajah dengan sebuah kotak biru dengan ketebalan 2
            cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            persen_proses = 100/jumlah_pengambilan
            status.set("Pengambilan gambar {0:.0f}%".format(jumlah*persen_proses))
            # nemabahkan nilai count (jumlah gambar pada id tertentu)
            jumlah += 1

            # Menyimpan gambar yang telah di tangkap pada folder dataset dengan nama sesuai ID dan count
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(jumlah) + ".jpg", gray[y:y + h, x:x + w])

            # Menampilkan video camera untuk untuk user yang akan di ambil gambar wajahnya
            cv2.imshow('Pengambilan Data Wajah', image_frame)

        # tekan q untuk menghentikan video frame
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        # jika gambar yang diambi sudah mencapai 100 maka video frame akan berhenti secara otomatis
        elif jumlah > jumlah_pengambilan:
            break

    # menghentikan kamera
    vid_cam.release()

    # keluar dari semua jendela program
    cv2.destroyAllWindows()

    status.set("Pengambilan gambar selesai")

def training():
    status.set("proses training, tunggu sebentar...")
    from PIL import Image
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector= cv2.CascadeClassifier("face-detect.xml");


    def getImagesAndLabels():
        path = ("/home/pi/recognizer/dataset")
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
        faceSamples=[]
        Ids=[]
        data = 0
        for imagePath in imagePaths:
            pilImage=Image.open(imagePath).convert('L')
            imageNp=np.array(pilImage,'uint8')
            Id=int(os.path.split(imagePath)[-1].split(".")[1])
            faces=detector.detectMultiScale(imageNp)
            for (x,y,w,h) in faces:
                faceSamples.append(imageNp[y:y+h,x:x+w])
                Ids.append(Id)
                # menampilkan citra grey scale (0-255)
                # print(imageNp)

                data += 1
            jml_gb = 100 / len(imagePaths)
            status.set("Training gambar {0:.0f}%".format(data * jml_gb))
        return faceSamples,Ids

    faces,Ids = getImagesAndLabels()
    recognizer.train(faces, np.array(Ids))
    # menampilkan id terakhir
    # print(Ids[-1:])
    # print(100*"=")
    # menampilkan hasil algoritma LBPH
    # print(faces,Ids)
    recognizer.save('trainner/trainner.yml')
    status.set("Training gambar 100%")
    time.sleep(2)
    status.set("Training gambar selesai")
    time.sleep(2)
    trn = DemoDialog(master)
    trn.onTrainingOk()
    status.clear()

def openfile():
    status.set("sedang membuka file...")
    master.filename = filedialog.askopenfilenames(initialdir="dataset/", title="Hapus file yang bukan wajah",filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    status.clear()

#----------------------- MEMBUAT TOMBOL GAMBAR -----------------------#
img_detect = PhotoImage(file="img/face_detec.png").subsample(15,15)
img_new = PhotoImage(file="img/add_data.png").subsample(15,15)
img_train = PhotoImage(file="img/training.png").subsample(15,15)
img_openfile = PhotoImage(file="img/files.png").subsample(15,15)

btn_detect = Button(master,image = img_detect,compound=LEFT,command=detect, text="Detect").pack(side=LEFT)
btn_new = Button(master,image=img_new,compound=LEFT,command=new,text="New").pack(side=LEFT)
btn_train = Button(master,image=img_train,compound=LEFT,command=training,text="Training").pack(side=LEFT)
btn_openfile = Button(master,image=img_openfile,compound=LEFT,command=openfile,text="Open file").pack(side=LEFT)

master.mainloop() # penutup window agar tidak langsung keluar
master.quit() # keluar  dari window
