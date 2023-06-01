import tkinter as tk
from tkinter import *
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

        self.btn_LamNetAnh = tk.Menubutton(self.frm, text="Nâng cao chất lượng ảnh",relief="raised")
        self.btn_LamNetAnh.grid(column = 0, row = 4, pady = 16, ipady = 12, ipadx = 16)
        self.menu_NetAnh = tk.Menu(self.btn_LamNetAnh, tearoff=0)
        self.btn_LamNetAnh["menu"] = self.menu_NetAnh
        self.menu_NetAnh.add_command(label="Giảm sáng",command=self.GiamSang)
        self.menu_NetAnh.add_command(label="Tăng sáng",command=self.TangSang)
        self.menu_NetAnh.add_command(label="Nghịch đảo",command=self.NghichDao)
        
        self.btn_LamMoAnh = tk.Menubutton(self.frm, text="Xử lý hình thái",relief="raised")
        self.btn_LamMoAnh.grid(column = 1, row = 4, ipady = 12, ipadx = 16)
        self.menu_XLHT = tk.Menu(self.btn_LamMoAnh,tearoff=0)
        self.btn_LamMoAnh["menu"] = self.menu_XLHT
        self.menu_XLHT.add_command(label="Giãn ảnh",command=self.GianAnh)
        self.menu_XLHT.add_command(label="Co ảnh",command=self.CoAnh)
        self.menu_XLHT.add_command(label="Trích biên",command=self.TrichBienAmh)
        self.menu_XLHT.add_command(label="Làm đầy",command=self.LamDay)
        
        self.btn_PhanDoanAnh = tk.Button(self.frm, text="Phân đoạn ảnh",command=self.PhanDoanAnh).grid(column = 2, row = 4, ipady = 12, ipadx = 16)
        
        self.btn_TrinhBienAnh = tk.Button(self.frm, text="Trích biên ảnh",command=self.TrichBienAnh).grid(column = 3, row = 4, ipady = 12, ipadx = 16)
        
        self.btn_TrichXDTAnh = tk.Button(self.frm, text="Trích xuất đặc trưng", command=self.TrichXuatDacTrungAnh).grid(column = 4, row = 4, ipady = 12, ipadx = 16)

        self.btn_ChonDD = tk.Button(self.frm, text="Chọn đường dẫn",command=self.choose_folder).grid(column = 0, row = 5, pady = 10, ipady = 4, ipadx = 32, columnspan = 5)

        self.btn_LamNet = tk.Menubutton(self.frm, text="Nâng cao chất lượng bộ dữ liệu",relief="raised")
        self.btn_LamNet.grid(column = 0, row = 6, pady = 16, ipady = 12, ipadx = 16)
        self.menu_NetAnhBo = tk.Menu(self.btn_LamNet, tearoff=0)
        self.btn_LamNet["menu"] = self.menu_NetAnhBo
        self.menu_NetAnhBo.add_command(label="Giảm sáng",command=self.GiamSangBo)
        self.menu_NetAnhBo.add_command(label="Tăng sáng",command=self.TangSangBo)
        self.menu_NetAnhBo.add_command(label="Nghịch đảo",command=self.NghichDaoBo)
        
        self.btn_LamMo = tk.Menubutton(self.frm, text="Xử lí hình thái bộ dữ liệu",relief="raised")
        self.btn_LamMo.grid(column = 1, row = 6, ipady = 12, ipadx = 16)
        self.menu_XLHTBo = tk.Menu(self.btn_LamMo,tearoff=0)
        self.btn_LamMo["menu"] = self.menu_XLHTBo
        self.menu_XLHTBo.add_command(label="Giãn ảnh",command=self.GianAnhBo)
        self.menu_XLHTBo.add_command(label="Co ảnh",command=self.CoAnhBo)
        self.menu_XLHTBo.add_command(label="Trích biên",command=self.TrichBienAmhBo)
        self.menu_XLHTBo.add_command(label="Làm đầy",command=self.LamDayBo)
        
        
        self.btn_PhanDoan = tk.Button(self.frm, text="Phân đoạn bộ dữ liệu",command=self.PhanDoanBoAnh).grid(column = 2, row = 6, ipady = 12, ipadx = 16)
        self.btn_TrichBien = tk.Button(self.frm, text="Trích biên bộ dữ liệu",command=self.TrichBienAnhBo).grid(column = 3, row = 6, ipady = 12, ipadx = 16)
        self.btn_TrichXDT = tk.Button(self.frm, text="Trích xuất đặc trưng bộ dữ liệu",command=self.TrichXuatDacTrungBo).grid(column = 4, row = 6, ipady = 12, ipadx = 16)
        self.btn_Thoat = tk.Button(self.frm, text="Thoát", command=self.root.quit).grid(column = 5, row = 6, ipady = 12, ipadx = 16)

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
    #Nâng cao chất lượng ảnh
    def LamNetAnh(self):
        img = cv2.imread(file_path,cv2.IMREAD_ANYCOLOR)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        cv2.imwrite("E:\ImageProcessing\ImageProcessing\Code\DoAn\sharpened.jpg", sharpened)
        photo = ImageTk.PhotoImage(Image.open("E:\ImageProcessing\ImageProcessing\Code\DoAn\sharpened.jpg").resize((400,300)))
        self.lb_AnhDaNhap = tk.Label(self.frm,image=photo, text = "Nơi thể hiện ảnh sau xử lý", background = "white", anchor="center").grid(column=3, row=1)
        self.lb_AnhDaNhap = photo
    def GiamSang(self):
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_GS = img /2
        cv2.imwrite("E:\ImageProcessing\ImageProcessing\Code\DoAn\img_GS.jpg", img_GS)
        photo = ImageTk.PhotoImage(Image.open("E:\ImageProcessing\ImageProcessing\Code\DoAn\img_GS.jpg").resize((400,300)))
        self.lb_AnhDaNhap = tk.Label(self.frm,image=photo, text = "Nơi thể hiện ảnh sau xử lý", background = "white", anchor="center").grid(column=3, row=1)
        self.lb_AnhDaNhap = photo
    def TangSang(self):
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_GS = img *2
        cv2.imwrite("E:\ImageProcessing\ImageProcessing\Code\DoAn\img_TS.jpg", img_GS)
        photo = ImageTk.PhotoImage(Image.open("E:\ImageProcessing\ImageProcessing\Code\DoAn\img_TS.jpg").resize((400,300)))
        self.lb_AnhDaNhap = tk.Label(self.frm,image=photo, text = "Nơi thể hiện ảnh sau xử lý", background = "white", anchor="center").grid(column=3, row=1)
        self.lb_AnhDaNhap = photo
    def NghichDao(self):
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_ND = 256 -1 - img 
        cv2.imwrite("E:\ImageProcessing\ImageProcessing\Code\DoAn\img_ND.jpg", img_ND)
        photo = ImageTk.PhotoImage(Image.open("E:\ImageProcessing\ImageProcessing\Code\DoAn\img_ND.jpg").resize((400,300)))
        self.lb_AnhDaNhap = tk.Label(self.frm,image=photo, text = "Nơi thể hiện ảnh sau xử lý", background = "white", anchor="center").grid(column=3, row=1)
        self.lb_AnhDaNhap = photo    
    
    
    #Xử lí hình thái
    def LamMoAnh(self):
        img = cv2.imread(file_path,cv2.IMREAD_ANYCOLOR)
        blurImg = cv2.blur(img,ksize=(3,3))
        cv2.imwrite("E:\ImageProcessing\ImageProcessing\Code\DoAn\ImgBlur.jpg", blurImg)
        photo = ImageTk.PhotoImage(Image.open("E:\ImageProcessing\ImageProcessing\Code\DoAn\ImgBlur.jpg").resize((400,300)))
        self.lb_AnhDaNhap = tk.Label(self.frm,image=photo, text = "Nơi thể hiện ảnh sau xử lý", background = "white", anchor="center").grid(column=3, row=1)
        self.lb_AnhDaNhap = photo
    def GianAnh(self):
        img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(img,kernel,iterations = 1)
        cv2.imwrite("E:\ImageProcessing\ImageProcessing\Code\DoAn\dilation.jpg", dilation)
        photo = ImageTk.PhotoImage(Image.open("E:\ImageProcessing\ImageProcessing\Code\DoAn\dilation.jpg").resize((400,300)))
        self.lb_AnhDaNhap = tk.Label(self.frm,image=photo, text = "Nơi thể hiện ảnh sau xử lý", background = "white", anchor="center").grid(column=3, row=1)
        self.lb_AnhDaNhap = photo
    
    def CoAnh(self):
        img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(img,kernel,iterations = 1)
        cv2.imwrite("E:\ImageProcessing\ImageProcessing\Code\DoAn\erosion.jpg", erosion)
        photo = ImageTk.PhotoImage(Image.open("E:\ImageProcessing\ImageProcessing\Code\DoAn\erosion.jpg").resize((400,300)))
        self.lb_AnhDaNhap = tk.Label(self.frm,image=photo, text = "Nơi thể hiện ảnh sau xử lý", background = "white", anchor="center").grid(column=3, row=1)
        self.lb_AnhDaNhap = photo
    def LamDay(self):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((5,5),np.uint8)
        opening   = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imwrite("E:\ImageProcessing\ImageProcessing\Code\DoAn\opening.jpg", opening  )
        photo = ImageTk.PhotoImage(Image.open("E:\ImageProcessing\ImageProcessing\Code\DoAn\opening.jpg").resize((400,300)))
        self.lb_AnhDaNhap = tk.Label(self.frm,image=photo, text = "Nơi thể hiện ảnh sau xử lý", background = "white", anchor="center").grid(column=3, row=1)
        self.lb_AnhDaNhap = photo 
        
    def TrichBienAmh(self):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((5,5),np.uint8)
        closing   = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("E:\ImageProcessing\ImageProcessing\Code\DoAn\closing.jpg", closing   )
        photo = ImageTk.PhotoImage(Image.open("E:\ImageProcessing\ImageProcessing\Code\DoAn\closing.jpg").resize((400,300)))
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
        
    def TrichBienAnh(self):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        kernel = np.array([
                    [1,1,1], 
                    [0,0,0],
                    [-1,-1,-1]
                ])
        edge_img = cv2.filter2D(img, -1, kernel)
        cv2.imwrite("E:\ImageProcessing\ImageProcessing\Code\DoAn\edge_img.jpg", edge_img   )
        photo = ImageTk.PhotoImage(Image.open("E:\ImageProcessing\ImageProcessing\Code\DoAn\edge_img.jpg").resize((400,300)))
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

    #Bộ ảnh
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
    #Nâng cao chất lượng
    def LamNetBoAnh(self):
        for img in image_files:
            img_path = f'{folder_path}/{img}'
            image = cv2.imread(img_path,cv2.IMREAD_ANYCOLOR)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            cv2.imwrite(f"E:\ImageProcessing\ImageProcessing\Code\DoAn\LamNet\{img}", sharpened)
    def GiamSangBo(self):
        for img in image_files:
            img_path = f'{folder_path}/{img}'
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            img_GS = image /2
            cv2.imwrite(f"E:\ImageProcessing\ImageProcessing\Code\DoAn\GiamSang\{img}", img_GS)
    def TangSangBo(self):
        for img in image_files:
            img_path = f'{folder_path}/{img}'
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            img_GS = image *2
            cv2.imwrite(f"E:\ImageProcessing\ImageProcessing\Code\DoAn\TangSang\{img}", img_GS)
    def NghichDaoBo(self):
        for img in image_files:
            img_path = f'{folder_path}/{img}'
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            img_GS = 256 - 1 - image
            cv2.imwrite(f"E:\ImageProcessing\ImageProcessing\Code\DoAn\ghichSang\{img}", img_GS)    
            
    #Xử lí hình thái
    def GianAnhBo(self):
        for img in image_files:
            img_path = f'{folder_path}/{img}'
            image = cv2.imread(img_path)
            kernel = np.ones((5,5),np.uint8)
            dilation = cv2.dilate(image,kernel,iterations = 1)
            cv2.imwrite(f"E:\ImageProcessing\ImageProcessing\Code\DoAn\GianAnh\{img}", dilation)
    
    def CoAnhBo(self):
        for img in image_files:
            img_path = f'{folder_path}/{img}'
            image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            kernel = np.ones((5,5),np.uint8)
            erosion = cv2.erode(image,kernel,iterations = 1)
            cv2.imwrite(f"E:\ImageProcessing\ImageProcessing\Code\DoAn\CoAnh\{img}", erosion)
    def LamDayBo(self):
        for img in image_files:
            img_path = f'{folder_path}/{img}'
            image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            kernel = np.ones((5,5),np.uint8)
            opening   = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            cv2.imwrite(f"E:\ImageProcessing\ImageProcessing\Code\DoAn\LamDay\{img}", opening)
        
    def TrichBienAmhBo(self):
        for img in image_files:
            img_path = f'{folder_path}/{img}'
            image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            kernel = np.ones((5,5),np.uint8)
            closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite(f"E:\ImageProcessing\ImageProcessing\Code\DoAn\Closing\{img}", closing)    
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
    def TrichBienAnhBo(self):
        for img in image_files:
            img_path = f'{folder_path}/{img}'
            image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            kernel = np.array([
                    [1,1,1], 
                    [0,0,0],
                    [-1,-1,-1]
                ])
            edge_img = cv2.filter2D(image, -1, kernel)
            cv2.imwrite(f"E:\ImageProcessing\ImageProcessing\Code\DoAn\TrichBien\{img}", edge_img)
    def TrichXuatDacTrungBo(self):
        for img1 in image_files:
            img_path = f'{folder_path}/{img1}'
            image = cv2.imread(img_path)
            img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            keypoints = sift.detect(img,None)
            image_with_keypoints = cv2.drawKeypoints(img, keypoints, image)
            cv2.imwrite(f"E:\ImageProcessing\ImageProcessing\Code\DoAn\TrichXDT\{img1}", image_with_keypoints)
app = Application()
