import customtkinter
from satellite_data import Satellite, Thruster
from location_monitor import check_position
from control_system import calculate_and_apply_correction

class App(customtkinter.CTk):
    def __init__(self, my_satellite, my_thruster):
        super().__init__()

        self.satellite = my_satellite
        self.thruster = my_thruster
        self.after_id = None

        self.title("Satellite Station-Keeping System")
        self.geometry("900x550")
        self.resizable(False, False)

        # Main frame layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Title label
        title_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        title_frame.grid(row=0, column=0, columnspan=2, pady=(20, 10), sticky="ew")
        title_frame.grid_columnconfigure(0, weight=1)
        self.title_label = customtkinter.CTkLabel(title_frame, text="🚀 Satellite Station-Keeping System", font=customtkinter.CTkFont(size=24, weight="bold"))
        self.title_label.grid(row=0, column=0)
        self.subtitle_label = customtkinter.CTkLabel(title_frame, text="Advanced orbital maintenance simulation with real-time position correction", font=customtkinter.CTkFont(size=14, slant="italic"))
        self.subtitle_label.grid(row=1, column=0, pady=(0, 10))

        # Position Status Frame
        self.status_frame = customtkinter.CTkFrame(self)
        self.status_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.create_position_status_frame()

        # Manual Position Override Frame
        self.manual_frame = customtkinter.CTkFrame(self)
        self.manual_frame.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")
        self.create_manual_override_frame()
        
        # System Status Frame
        self.system_status_frame = customtkinter.CTkFrame(self)
        self.system_status_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="nsew")
        self.create_system_status_frame()
        
        # Start the update loop
        self.update_gui()

    def create_position_status_frame(self):
        self.status_frame.grid_columnconfigure((0, 1), weight=1)
        self.status_frame.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)
        
        # Frame Title
        self.status_title = customtkinter.CTkLabel(self.status_frame, text="🚀 Position Status", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.status_title.grid(row=0, column=0, columnspan=2, pady=(10, 5))
        
        # Labels for Target and Current
        customtkinter.CTkLabel(self.status_frame, text="Target Location", font=customtkinter.CTkFont(size=14, weight="bold")).grid(row=1, column=0, pady=(5, 0))
        customtkinter.CTkLabel(self.status_frame, text="Current Location", font=customtkinter.CTkFont(size=14, weight="bold")).grid(row=1, column=1, pady=(5, 0))
        
        # Target Location Display
        target_location = self.satellite._target_location
        self.target_x_label = customtkinter.CTkLabel(self.status_frame, text=f"X: {target_location[0]:.3f}", text_color="green")
        self.target_x_label.grid(row=2, column=0, padx=10, pady=(0, 2))
        self.target_y_label = customtkinter.CTkLabel(self.status_frame, text=f"Y: {target_location[1]:.3f}", text_color="green")
        self.target_y_label.grid(row=3, column=0, padx=10, pady=(0, 2))
        self.target_z_label = customtkinter.CTkLabel(self.status_frame, text=f"Z: {target_location[2]:.3f}", text_color="green")
        self.target_z_label.grid(row=4, column=0, padx=10, pady=(0, 2))
        
        # Current Location Display
        self.current_x_label = customtkinter.CTkLabel(self.status_frame, text="X: 0.000")
        self.current_x_label.grid(row=2, column=1, padx=10, pady=(0, 2))
        self.current_y_label = customtkinter.CTkLabel(self.status_frame, text="Y: 0.000")
        self.current_y_label.grid(row=3, column=1, padx=10, pady=(0, 2))
        self.current_z_label = customtkinter.CTkLabel(self.status_frame, text="Z: 0.000")
        self.current_z_label.grid(row=4, column=1, padx=10, pady=(0, 2))

        # Status Indicator
        self.on_course_label = customtkinter.CTkLabel(self.status_frame, text="✔ ON COURSE", text_color="green", font=customtkinter.CTkFont(size=14, weight="bold"))
        self.on_course_label.grid(row=5, column=0, columnspan=2, pady=10)

    def create_manual_override_frame(self):
        self.manual_frame.grid_columnconfigure(0, weight=1)
        self.manual_frame.grid_rowconfigure((0, 1, 2, 3, 4, 5), weight=1)
        
        self.manual_title = customtkinter.CTkLabel(self.manual_frame, text="Manual Position Override", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.manual_title.grid(row=0, column=0, pady=(10, 5))
        
        # Input fields
        self.x_input = customtkinter.CTkEntry(self.manual_frame, placeholder_text="X Coordinate")
        self.x_input.grid(row=1, column=0, padx=20, pady=5)
        self.y_input = customtkinter.CTkEntry(self.manual_frame, placeholder_text="Y Coordinate")
        self.y_input.grid(row=2, column=0, padx=20, pady=5)
        self.z_input = customtkinter.CTkEntry(self.manual_frame, placeholder_text="Z Coordinate")
        self.z_input.grid(row=3, column=0, padx=20, pady=5)

        self.apply_button = customtkinter.CTkButton(self.manual_frame, text="✔ Apply Manual Correction", command=self.apply_correction)
        self.apply_button.grid(row=4, column=0, padx=20, pady=20)

    def create_system_status_frame(self):
        self.system_status_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.system_status_frame.grid_rowconfigure((0, 1), weight=1)
        
        self.system_title = customtkinter.CTkLabel(self.system_status_frame, text="System Status", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.system_title.grid(row=0, column=0, columnspan=3, pady=(10, 5))
        
        self.status_message_label = customtkinter.CTkLabel(self.system_status_frame, text="Satellite is on course.")
        self.status_message_label.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="w")
        
        self.thruster_status_label = customtkinter.CTkLabel(self.system_status_frame, text="Thruster Status: STANDBY")
        self.thruster_status_label.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="e")

        self.position_lock_label = customtkinter.CTkLabel(self.system_status_frame, text="Position Lock: LOCKED")
        self.position_lock_label.grid(row=1, column=2, padx=10, pady=(0, 10), sticky="e")

    def update_gui(self):
        # 1. Simulate drift
        self.satellite.simulate_drift()

        # 2. Check position
        is_on_course = check_position(self.satellite)
        current_location = self.satellite.get_location()

        # 3. Update display based on position
        self.current_x_label.configure(text=f"X: {current_location[0]:.3f}", text_color="red" if not is_on_course else "green")
        self.current_y_label.configure(text=f"Y: {current_location[1]:.3f}", text_color="red" if not is_on_course else "green")
        self.current_z_label.configure(text=f"Z: {current_location[2]:.3f}", text_color="red" if not is_on_course else "green")

        if is_on_course:
            self.on_course_label.configure(text="✔ ON COURSE", text_color="green")
            self.status_message_label.configure(text="Satellite is on course.")
        else:
            self.on_course_label.configure(text="❌ OFF COURSE", text_color="red")
            self.status_message_label.configure(text="Satellite is off course. Awaiting manual correction.")
            
        # Schedule the next update
        self.after_id = self.after(2000, self.update_gui)

    def apply_correction(self):
        try:
            x = float(self.x_input.get())
            y = float(self.y_input.get())
            z = float(self.z_input.get())
            new_location = [x, y, z]

            # Stop the automatic drift updates temporarily
            if self.after_id:
                self.after_cancel(self.after_id)
            
            # Apply the correction
            correction_vector = calculate_and_apply_correction(self.satellite, self.thruster, new_location)
            
            # Update system status after correction
            self.status_message_label.configure(text="Correction applied. Satellite is now on course.")
            self.thruster_status_label.configure(text="Thruster Status: FIRING", text_color="orange")

            # Update the GUI to show the corrected position immediately
            current_location = self.satellite.get_location()
            self.current_x_label.configure(text=f"X: {current_location[0]:.3f}", text_color="green")
            self.current_y_label.configure(text=f"Y: {current_location[1]:.3f}", text_color="green")
            self.current_z_label.configure(text=f"Z: {current_location[2]:.3f}", text_color="green")
            self.on_course_label.configure(text="✔ ON COURSE", text_color="green")

            # Clear input fields
            self.x_input.delete(0, 'end')
            self.y_input.delete(0, 'end')
            self.z_input.delete(0, 'end')

            # Restart the update loop after a short delay
            self.after(3000, self.restart_update_loop)

        except ValueError:
            self.status_message_label.configure(text="Invalid input. Please enter valid numbers.", text_color="red")
            
    def restart_update_loop(self):
        self.thruster_status_label.configure(text="Thruster Status: STANDBY", text_color="white")
        self.update_gui()


if __name__ == "__main__":
    target_location = (100.0, 200.0, 300.0)
    my_satellite = Satellite(target_location)
    my_thruster = Thruster()
    
    app = App(my_satellite, my_thruster)
    app.mainloop()