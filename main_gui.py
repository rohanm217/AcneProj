import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
from ultralytics import YOLO
import os

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SkinScanAI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SkinScan AI - Professional Documentation Version")
        self.geometry("1100x850")

        self.model_path = r'C:\Users\malig\OneDrive\Desktop\acneproj\runs\detect\skin_pro_medium_v1\weights\best.pt'
        
        try:
            self.model = YOLO(self.model_path)
            self.status_msg = f"Model: YOLOv8m | Epochs: 175 | Classes: 10"
        except Exception:
            self.model = None
            self.status_msg = "Status: Model weights not found in path."

        self.current_path = None
        
        self.recs = {
            'Pimples': {'AM': 'Salicylic Acid Cleanser', 'PM': 'Benzoyl Peroxide 5% Spot Treatment'},
            'blackhead': {'AM': 'BHA Liquid Exfoliant', 'PM': 'Double Cleanse (Oil + Water)'},
            'conglobata': {'AM': 'Gentle Hydrating Cleanser', 'PM': 'URGENT: Requires Prescription (Isotretinoin)'},
            'crystanlline': {'AM': 'Centella Asiatica Soothing Mist', 'PM': 'Ceramide Barrier Repair Cream'},
            'cystic': {'AM': 'Anti-inflammatory Serum (Niacinamide)', 'PM': 'Adapalene 0.1% Gel + Deep Hydration'},
            'folliculitis': {'AM': 'Antibacterial Wash (Benzoyl Peroxide)', 'PM': 'Warm Compress + Mupirocin (if prescribed)'},
            'keloid': {'AM': 'Silicone Gel / Sheet', 'PM': 'Consult Professional for Corticosteroid info'},
            'milium': {'AM': 'Mild Lactic Acid Exfoliation', 'PM': 'Do Not Squeeze - Professional Extraction Only'},
            'papular': {'AM': 'Azelaic Acid 10% Suspension', 'PM': 'Zinc-based Soothing Cream'},
            'purulent': {'AM': 'Hydrocolloid Pimple Patch', 'PM': 'Gentle Non-foaming Cleanser (Avoid physical scrubs)'}
        }

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo = ctk.CTkLabel(self.sidebar, text="SkinScan AI", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo.pack(pady=30)

        self.upload_btn = ctk.CTkButton(self.sidebar, text="Upload Photo", command=self.upload_image)
        self.upload_btn.pack(pady=10, padx=20)

        self.conf_label = ctk.CTkLabel(self.sidebar, text="Sensitivity: 0.35")
        self.conf_label.pack(pady=(20, 0))
        self.conf_slider = ctk.CTkSlider(self.sidebar, from_=0.1, to=0.9, command=self.update_conf)
        self.conf_slider.set(0.35)
        self.conf_slider.pack(pady=10, padx=20)

        self.save_btn = ctk.CTkButton(self.sidebar, text="Save Report", command=self.save_report, fg_color="green")
        self.save_btn.pack(pady=10, padx=20)

        self.status_label = ctk.CTkLabel(self.sidebar, text=self.status_msg, font=ctk.CTkFont(size=10), text_color="gray")
        self.status_label.pack(side="bottom", pady=20)

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.img_label = ctk.CTkLabel(self.main_frame, text="Upload Image for Detailed Analysis")
        self.img_label.pack(expand=True, pady=10)

        self.result_text = ctk.CTkTextbox(self.main_frame, height=350, width=650, font=("Helvetica", 14))
        self.result_text.pack(pady=20, padx=20)

    def update_conf(self, value):
        self.conf_label.configure(text=f"Sensitivity: {value:.2f}")
        if self.current_path:
            self.analyze_skin(self.current_path)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.current_path = file_path
            self.analyze_skin(file_path)

    def analyze_skin(self, path):
        if not self.model:
            self.result_text.insert("0.0", "Model not loaded.")
            return

        conf_val = self.conf_slider.get()
        results = self.model.predict(source=path, conf=conf_val)
        found = set()
        processed_img = None

        for r in results:
            plot_bgr = r.plot() 
            plot_rgb = plot_bgr[:, :, ::-1] 
            processed_img = Image.fromarray(plot_rgb)
            for box in r.boxes:
                found.add(self.model.names[int(box.cls[0])])

        if processed_img:
            ctk_img = ctk.CTkImage(light_image=processed_img, size=(450, 450))
            self.img_label.configure(image=ctk_img, text="")
            self.img_label.image = ctk_img

        self.result_text.delete("0.0", "end")
        if not found:
            self.result_text.insert("0.0", "No skin conditions detected at this sensitivity.")
        else:
            self.result_text.insert("0.0", f"--- SKIN ANALYSIS REPORT (Threshold: {conf_val:.2f}) ---\n\n")
            for issue in found:
                advice = self.recs.get(issue, {'AM': 'Gentle Care', 'PM': 'Consult Professional'})
                self.result_text.insert("end", f"Condition: {issue.upper()}\n")
                self.result_text.insert("end", f"  Recommended AM: {advice['AM']}\n")
                self.result_text.insert("end", f"  Recommended PM: {advice['PM']}\n")
                self.result_text.insert("end", "-"*50 + "\n")

    def save_report(self):
        content = self.result_text.get("0.0", "end").strip()
        if content:
            path = filedialog.asksaveasfilename(defaultextension=".txt")
            if path:
                with open(path, "w") as f:
                    f.write(content)

if __name__ == "__main__":
    app = SkinScanAI()
    app.mainloop()