import tkinter as tk
from screeninfo import get_monitors
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import messagebox

class Application:
    def __init__(self):
        self.create_Form()

    def create_Form(self):
        # Lấy thông tin kích thước màn hình
        monitor = get_monitors()[0]
        width = monitor.width
        height = monitor.height
        
        # Tạo cửa sổ
        self.root = tk.Tk()
        
        # Thiết lập kích thước cửa sổ bằng kích thước màn hình
        self.root.geometry(f"{width}x{height}+{-10}+{0}")

        #add frame
        self.frme_Action = tk.Frame(self.root)
        self.frme_Action.pack()

        #add button
        self.btn_LoadImg = tk.Button(self.frme_Action,text="Load Image",command= self.get_UrlImg)
        self.btn_LoadImg.pack()

        self.root.mainloop()
        
    def get_UrlImg(self):
        # Chọn file ảnh từ file explorer
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;")])
        

        # Nếu file ảnh đã được chọn, hiển thị nó trên cửa sổ
        if file_path:
            img = Image.open(file_path)
            photo = ImageTk.PhotoImage(img)

            label = tk.Label(self.root, image=photo)
            label.image = photo
            label.pack()
        return file_path

app = Application()