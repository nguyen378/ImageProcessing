import tkinter as tk
from screeninfo import get_monitors
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import messagebox
import cv2
import numpy as np
import os


class Application:
    def __init__(self):
        global img_path 
        self.create_Form()

    def create_Form(self):
        # Tạo cửa sổ
        
        self.root = tk.Tk()
        self.root.title("Đồ án xử lý ảnh")

        self.frm = tk.Frame(self.root)
        self.frm.grid()

        self.btn_LoadImg = tk.Button(self.frm, text="Click để nhập ảnh", command=self.openImg).grid(column = 1, row = 0, pady = 10, ipady = 4, ipadx = 16)
        self.lb_a = tk.Label(self.frm, text="Ảnh sau khi được xử lý").grid(column = 3, row = 0, pady = 10)

        self.lb_NhapAnh = tk.Label(self.frm, text = "Nơi thể hiện ảnh nhập vào", background = "white", anchor="center").grid(column=1, row=1, ipadx = 85, ipady = 137)
        
        self.lb_AnhDaNhap = tk.Label(self.frm, text = "Nơi thể hiện ảnh sau xử lý", background = "white", anchor="center").grid(column=3, row=1,ipadx = 85, ipady = 137)

        self.btn_LamNetAnh = tk.Button(self.frm, text="Làm nét ảnh",command=self.LamNetAnh).grid(column = 0, row = 4, pady = 16, ipady = 12, ipadx = 16)
        self.btn_LamMoAnh = tk.Button(self.frm, text="Làm mờ ảnh",command=self.LamMoAnh).grid(column = 1, row = 4, ipady = 12, ipadx = 16)
        self.btn_PhanDoanAnh = tk.Button(self.frm, text="Phân đoạn ảnh",command=self.PhanDoanAnh).grid(column = 2, row = 4, ipady = 12, ipadx = 16)
        self.btn_TrinhBienAnh = tk.Button(self.frm, text="Trích biên ảnh",command=self.TrichBienAmh).grid(column = 3, row = 4, ipady = 12, ipadx = 16)
        self.btn_TrichXDTAnh = tk.Button(self.frm, text="Trích xuất đặc trưng", command=self.TrichXuatDacTrungAnh).grid(column = 4, row = 4, ipady = 12, ipadx = 16)

        self.btn_ChonDD = tk.Button(self.frm, text="Chọn đường dẫn",command=self.choose_folder).grid(column = 0, row = 5, pady = 10, ipady = 4, ipadx = 32, columnspan = 2)

        self.btn_LamNet = tk.Button(self.frm, text="Làm nét bộ dữ liệu",command=self.LamNetBoAnh).grid(column = 0, row = 6, pady = 16, ipady = 12, ipadx = 16)
        self.btn_LamMo = tk.Button(self.frm, text="Làm mờ bộ dữ liệu",command=self.LamMoBoAnh).grid(column = 1, row = 6, ipady = 12, ipadx = 16)
        self.btn_PhanDoan = tk.Button(self.frm, text="Phân đoạn bộ dữ liệu",command=self.PhanDoanBoAnh).grid(column = 2, row = 6, ipady = 12, ipadx = 16)
        self.btn_TrichBien = tk.Button(self.frm, text="Trích biên bộ dữ liệu",command=self.TrichBienBoAnh).grid(column = 3, row = 6, ipady = 12, ipadx = 16)
        self.btn_Thoat = tk.Button(self.frm, text="Thoát", command=self.root.quit).grid(column = 4, row = 6, ipady = 12, ipadx = 16)

        # start the program
        self.root.mainloop()
    
    def openImg(self):
        # Chọn file ảnh từ file explorer
        global file_path
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;")])
        # Nếu file ảnh đã được chọn, hiển thị nó trên cửa sổ
        if file_path:
            self.showImg()
    def showImg(self):
        photo = ImageTk.PhotoImage(Image.open(file_path).resize((400,300)))
        self.lb_NhapAnh = tk.Label(self.frm,image=photo, text = "Nơi thể hiện ảnh nhập vào", background = "white", anchor="center").grid(column=1, row=1)
        self.lb_NhapAnh = photo
    
    def LamNetAnh(self):
        img = cv2.imread(file_path,cv2.IMREAD_ANYCOLOR)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        cv2.imwrite("E:\ImageProcessing\ImageProcessing\Code\DoAn\sharpened.jpg", sharpened)
        photo = ImageTk.PhotoImage(Image.open("E:\ImageProcessing\ImageProcessing\Code\DoAn\sharpened.jpg").resize((400,300)))
        self.lb_AnhDaNhap = tk.Label(self.frm,image=photo, text = "Nơi thể hiện ảnh sau xử lý", background = "white", anchor="center").grid(column=3, row=1)
        self.lb_AnhDaNhap = photo
    def LamMoAnh(self):
        img = cv2.imread(file_path,cv2.IMREAD_ANYCOLOR)
        blurImg = cv2.blur(img,ksize=(3,3))
        cv2.imwrite("E:\ImageProcessing\ImageProcessing\Code\DoAn\ImgBlur.jpg", blurImg)
        photo = ImageTk.PhotoImage(Image.open("E:\ImageProcessing\ImageProcessing\Code\DoAn\ImgBlur.jpg").resize((400,300)))
        self.lb_AnhDaNhap = tk.Label(self.frm,image=photo, text = "Nơi thể hiện ảnh sau xử lý", background = "white", anchor="center").grid(column=3, row=1)
        self.lb_AnhDaNhap = photo
    def PhanDoanAnh(self):
        img = cv2.imread(file_path)
        pixel_vals = img.reshape((-1,3))
        pixel_vals = np.float32(pixel_vals)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        k = 3
        retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape((img.shape))
        cv2.imwrite("E:\ImageProcessing\ImageProcessing\Code\DoAn\segmented.jpg", segmented_image)
        photo = ImageTk.PhotoImage(Image.open("E:\ImageProcessing\ImageProcessing\Code\DoAn\segmented.jpg").resize((400,300)))
        self.lb_AnhDaNhap = tk.Label(self.frm,image=photo, text = "Nơi thể hiện ảnh sau xử lý", background = "white", anchor="center").grid(column=3, row=1)
        self.lb_AnhDaNhap = photo
    def TrichBienAmh(self):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((5,5),np.uint8)
        opening   = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("E:\ImageProcessing\ImageProcessing\Code\DoAn\opening.jpg", opening  )
        photo = ImageTk.PhotoImage(Image.open("E:\ImageProcessing\ImageProcessing\Code\DoAn\opening.jpg").resize((400,300)))
        self.lb_AnhDaNhap = tk.Label(self.frm,image=photo, text = "Nơi thể hiện ảnh sau xử lý", background = "white", anchor="center").grid(column=3, row=1)
        self.lb_AnhDaNhap = photo
    def TrichXuatDacTrungAnh(self):
        image = cv2.imread(file_path)
        img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints = sift.detect(img,None)
        image_with_keypoints = cv2.drawKeypoints(img, keypoints, image)
        cv2.imwrite("E:\ImageProcessing\ImageProcessing\Code\DoAn\sift.jpg", image_with_keypoints  )
        photo = ImageTk.PhotoImage(Image.open("E:\ImageProcessing\ImageProcessing\Code\DoAn\sift.jpg").resize((400,300)))
        self.lb_AnhDaNhap = tk.Label(self.frm,image=photo, text = "Nơi thể hiện ảnh sau xử lý", background = "white", anchor="center").grid(column=3, row=1)
        self.lb_AnhDaNhap = photo

    def choose_folder(self):
        global folder_path
        global image_files
        folder_path = filedialog.askdirectory()
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
        image_files = []
        for file_name in os.listdir(folder_path):
            extension = os.path.splitext(file_name)[1].lower()
            if extension in image_extensions:
                image_files.append(file_name)
        print(folder_path)
        print(image_files)
    def LamNetBoAnh(self):
        for img in image_files:
            img_path = f'{folder_path}/{img}'
            image = cv2.imread(img_path,cv2.IMREAD_ANYCOLOR)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            cv2.imwrite(f"E:\ImageProcessing\ImageProcessing\Code\DoAn\LamNet\{img}", sharpened)
    def LamMoBoAnh(self):
        for img in image_files:
            img_path = f'{folder_path}/{img}'
            image = cv2.imread(img_path,cv2.IMREAD_ANYCOLOR)
            blurImg = cv2.blur(image,ksize=(3,3))
            cv2.imwrite(f"E:\ImageProcessing\ImageProcessing\Code\DoAn\LamMo\{img}", blurImg)
    def PhanDoanBoAnh(self):
        for img in image_files:
            img_path = f'{folder_path}/{img}'
            image = cv2.imread(img_path,cv2.IMREAD_ANYCOLOR)
            pixel_vals = image.reshape((-1,3))
            pixel_vals = np.float32(pixel_vals)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
            k = 3
            retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            segmented_image = segmented_data.reshape((image.shape))
            cv2.imwrite(f"E:\ImageProcessing\ImageProcessing\Code\DoAn\PhanDoan\{img}", segmented_image)
    def TrichBienBoAnh(self):
        for img in image_files:
            img_path = f'{folder_path}/{img}'
            image = cv2.imread(img_path,cv2.IMREAD_ANYCOLOR)
            kernel = np.ones((5,5),np.uint8)
            opening   = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite(f"E:\ImageProcessing\ImageProcessing\Code\DoAn\TrichBien\{img}", opening)

app = Application()
